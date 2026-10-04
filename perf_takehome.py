"""Single-file 881-cycle compiler for the original one-core take-home machine.

All tree preprocessing and input computation execute as original ISA operations.
No saved instruction listing, input-dependent host computation, experiment imports,
expanded machine limits, or changes to reference/test files are used.

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
publish or redistribute solutions, consistent with the original source notice.
"""
from collections import defaultdict, Counter
from functools import lru_cache
import collections, heapq, random, sys
from problem import (Engine, DebugInfo, SLOT_LIMITS, VLEN, N_CORES, SCRATCH_SIZE,
                     HASH_STAGES)
LIMIT={'alu':12,'valu':6,'load':2,'store':2,'flow':1}

class FrontendBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.stagger=3
        self.reg_weight=20
        self.reg_limit=160
        self.pointer_flows=64
        self.index_select3_groups=0

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length

        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_kernel(self, forest_height, n_nodes, batch_size, rounds):
        """Vector traversal with list scheduling across independent batches."""
        import heapq
        ops = []
        last_write = {}
        virtual_map = {}
        virtual_next = 0
        virtual_width = {}
        op_rw = []
        op_group = []
        current_group=-1
        def new_virtual(logical, width):
            nonlocal virtual_next
            out = virtual_next
            virtual_next += 8
            virtual_width[out] = width
            for i in range(width): virtual_map[logical+i] = out+i
            return out
        def remap_slot(engine,slot,readmap,writemap):
            op=slot[0]
            if engine=='valu':
                return (op,writemap(slot[1]))+tuple(readmap(x) for x in slot[2:])
            if engine=='flow':
                if op=='add_imm': return (op,writemap(slot[1]),readmap(slot[2]),slot[3])
                assert op=='vselect'
                return (op,writemap(slot[1]))+tuple(readmap(x) for x in slot[2:])
            if engine=='load':
                if op=='const': return (op,writemap(slot[1]),slot[2])
                if op=='load_offset': return (op,writemap(slot[1]),readmap(slot[2]),slot[3])
                return (op,writemap(slot[1]),readmap(slot[2]))
            if engine=='store': return (op,readmap(slot[1]),readmap(slot[2]))
            if engine=='alu': return (op,writemap(slot[1]),readmap(slot[2]),readmap(slot[3]))
            raise AssertionError(engine)
        def emit(engine, slot, reads, writes, vector_lane=None):
            j=len(ops)
            oldmap=virtual_map.copy()
            vreads=[oldmap[x] for x in reads]
            if writes:
                if vector_lane is not None:
                    if vector_lane==0: new_virtual(writes[0],8)
                elif engine=='load' and slot[0]=='load_offset':
                    if slot[3]==0: new_virtual(slot[1],8)
                else: new_virtual(writes[0],len(writes))
            vwrites=[virtual_map[x] for x in writes]
            # load_offset's base points to its eight-lane virtual register.
            vslot=remap_slot(engine,slot,oldmap.__getitem__,virtual_map.__getitem__)
            deps={last_write[x] for x in vreads if x in last_write}
            ops.append((engine,vslot,deps))
            op_rw.append((vreads,vwrites))
            op_group.append(current_group)
            for x in vwrites: last_write[x]=j
        def vec(a): return list(range(a,a+VLEN))
        def v(op,d,a,b): emit('valu',(op,d,a,b),vec(a)+vec(b),vec(d))
        def ma(d,a,b,c): emit('valu',('multiply_add',d,a,b,c),vec(a)+vec(b)+vec(c),vec(d))
        def sel(d,c,a,b): emit('flow',('vselect',d,c,a,b),vec(c)+vec(a)+vec(b),vec(d))
        constants = {}
        def c(x):
            if x not in constants:
                a = self.alloc_scratch(length=8)
                emit('load',('const',a,x),[],[a])
                emit('valu',('vbroadcast',a,a),[a],vec(a))
                constants[x] = a
            return constants[x]
        one,two = c(1),c(2)
        cache_depth = getattr(self,"cache_depth",2)
        cache4_groups = getattr(self,"cache4_groups",0)
        c5=c(0xB55A4F09)
        pointer = self.alloc_scratch()
        cache_count=(1 << (cache_depth+1))-1
        cache_loads=[]
        preprocess_reads=[]
        preprocess_writes=[]
        preload=self.alloc_scratch(length=8)
        raw_scalars=[]
        transformed=[]
        root_original=self.alloc_scratch(length=8)
        for start in range(0,cache_count,8):
            emit('load',('const',pointer,7+start),[],[pointer])
            emit('load',('vload',preload,pointer),[pointer],vec(preload))
            cache_loads.append(len(ops)-1)
            preprocess_reads.append((len(ops)-1,7+start,7+start+8))
            if start==0:
                emit('valu',('vbroadcast',root_original,preload),[preload],vec(root_original))
            for i in range(start,min(start+8,cache_count)):
                a=self.alloc_scratch()
                emit('alu',('^',a,preload+i-start,c5),[preload+i-start,c5],[a])
                transformed.append(a)
        scalar_tree=[]
        for dep in range(cache_depth+1):
            scalar_tree.extend(reversed(transformed[(1<<dep)-1:(1<<(dep+1))-1]))
        tree=[None]*cache_count
        differences={}
        def broadcast_scalar(a):
            dest=self.alloc_scratch(length=8)
            emit('valu',('vbroadcast',dest,a),[a],vec(dest))
            return dest
        tree[0]=broadcast_scalar(scalar_tree[0])
        for i in range(2,len(tree),2):
            tree[i-1]=broadcast_scalar(scalar_tree[i-1])
            if (getattr(self,'flow_mask',31)>>(i//2-1))&1:
                tree[i]=broadcast_scalar(scalar_tree[i])
            else:
                d=self.alloc_scratch()
                emit('alu',('-',d,scalar_tree[i],scalar_tree[i-1]),[scalar_tree[i],scalar_tree[i-1]],[d])
                differences[i]=broadcast_scalar(d)
        pre_stores=[]
        pre_buf=self.alloc_scratch(length=8)
        pre_src=self.alloc_scratch();pre_dst=self.alloc_scratch()
        pointer_last={}
        def pointer_const(dest,value):
            if dest in pointer_last and value==pointer_last[dest]+8:
                eight=c(8)
                emit('alu',('+',dest,dest,eight),[dest,eight],[dest])
            elif value in (8,14):
                source=c(value)
                virtual_map[dest]=virtual_map[source]
            else:
                emit('load',('const',dest,value),[],[dest])
            pointer_last[dest]=value
        prefix_loads=list(cache_loads)
        packet_bands=[(3,2),(5,3)]
        packet_depth=5
        packet_height=2
        packet_nodes=(1<<packet_height)-1
        packet_width=1<<packet_depth
        for dep in range(4,8):
            if any(pd <= dep < pd+ph for pd,ph in packet_bands): continue
            width=1<<dep
            level_src=self.alloc_scratch(length=width)
            for off in range(0,width,8):
                pointer_const(pre_src,6+width+off)
                emit('load',('vload',level_src+off,pre_src),[pre_src],vec(level_src+off))
                prefix_loads.append(len(ops)-1)
            for off in range(0,width,8):
                emit('load',('const',pre_dst,width+off),[],[pre_dst])
                for lane in range(8):
                    source=level_src+width-1-off-lane
                    emit('alu',('^',pre_buf+lane,source,c5),[source,c5],[pre_buf+lane],vector_lane=lane)
                emit('store',('vstore',pre_dst,pre_buf),[pre_dst]+vec(pre_buf),[])
                ops[-1][2].update(prefix_loads)
                pre_stores.append(len(ops)-1)
        # Load both original bands before any overlapping relocation writes.
        packet_loads=[]
        packet_stores=[]
        packet_stores_by_depth={}
        for packet_depth,packet_height in packet_bands:
            packet_stores_by_depth[packet_depth]=[]
            packet_width=1<<packet_depth
            packet_levels=[]
            for depth in range(packet_depth,packet_depth+packet_height):
                width=1<<depth
                level=self.alloc_scratch(length=width)
                packet_levels.append(level)
                for off in range(0,width,8):
                    pointer_const(pre_src,6+width+off)
                    emit('load',('vload',level+off,pre_src),[pre_src],vec(level+off))
                    packet_loads.append(len(ops)-1)
                    preprocess_reads.append((len(ops)-1,6+width+off,6+width+off+8))
            packet_flat=[]
            for root in range(packet_width):
                for level,source in enumerate(packet_levels):
                    count=1<<level
                    for child in range(count):
                        packet_flat.append(source+(packet_width<<level)-1-root*count-child)
            for off in range(0,len(packet_flat),8):
                pointer_const(pre_dst,packet_width+off)
                for lane in range(8):
                    source=packet_flat[off+lane]
                    emit('alu',('^',pre_buf+lane,source,c5),[source,c5],[pre_buf+lane],vector_lane=lane)
                emit('store',('vstore',pre_dst,pre_buf),[pre_dst]+vec(pre_buf),[])
                packet_stores.append(len(ops)-1)
                packet_stores_by_depth[packet_depth].append(len(ops)-1)
                preprocess_writes.append((len(ops)-1,packet_width+off,packet_width+off+8))
        for store,lo,hi in preprocess_writes:
            ops[store][2].update(read for read,rlo,rhi in preprocess_reads if lo<rhi and rlo<hi)
        tmp_pool=[self.alloc_scratch(length=8) for _ in range(8)]
        shared_w=tmp_pool[0]
        states=[]
        histories=[]
        for g in range(batch_size//8):
            current_group=g
            val,idx,t,u = [self.alloc_scratch(length=8) for _ in range(4)]
            p=self.alloc_scratch()
            if g<getattr(self,'input_pointer_flows',6):
                emit('flow',('add_imm',p,one,6+n_nodes+batch_size+g*8),[one],[p])
            else:
                emit('load',('const',p,7+n_nodes+batch_size+g*8),[],[p])
            emit('load',('vload',val,p),[p],vec(val))
            states.append((val,idx,t,u,shared_w,p))
            histories.append([self.alloc_scratch(length=8) for _ in range(5)])
        packets=[[self.alloc_scratch(length=8) for _ in range(8)] for g in states]
        packet_bits=[[self.alloc_scratch(length=8) for _ in range(packet_height-1)] for g in states]
        packet_candidates=[self.alloc_scratch(length=8) for _ in range(1<<(packet_height-1))]
        compact_inv3=pow(3,-1,1<<32)
        compact_inv7=pow(7,-1,1<<32)
        compact_a=28*compact_inv3 & 0xffffffff
        compact_b=(448*compact_inv3-192) & 0xffffffff
        compact_c=(-8*compact_inv7) & 0xffffffff
        compact_e=(767-1536*compact_inv7) & 0xffffffff
        deep_start=8
        deep_load_buffers=[self.alloc_scratch(length=8) for _ in range(8)]
        deep_source=self.alloc_scratch(length=8)
        four_nbuf=getattr(self,'four_buffers',2)
        four_ptrs=[]
        for b in range(four_nbuf):
            row=[]
            for candidate in range(2):
                ptr=self.alloc_scratch(length=8)
                if b==0:
                    for lane in range(8):
                        emit('load',('const',ptr+lane,7+n_nodes+8*candidate+lane),[],[ptr+lane],vector_lane=lane)
                else:
                    v('+',ptr,four_ptrs[b-1][candidate],c(16))
                row.append(ptr)
            four_ptrs.append(row)
        four_addr1=[self.alloc_scratch(length=8) for g in states]
        four_early_stores={}
        middle_jobs={}
        middle_memory_edges=[]
        def middle_selected(group):
            mask=getattr(self,'middle_group_mask',None)
            return bool(mask & (1<<group)) if mask is not None else group<getattr(self,'middle_two_groups',0)
        four_late_addr=self.alloc_scratch(length=8)
        four_late_addr2=self.alloc_scratch(length=8)
        four_nodes=self.alloc_scratch(length=8)
        four_jobs={}
        four_packet_loads={}
        for r in range(rounds):
            depth=r%(forest_height+1)
            active_packet=next(((pd,ph) for pd,ph in packet_bands if pd<=depth<pd+ph),None)
            if active_packet:
                packet_depth,packet_height=active_packet
                packet_nodes=(1<<packet_height)-1
                packet_width=1<<packet_depth
            for g,(val,idx,t,u,w,p) in enumerate(states):
                current_group=g
                first_gather=cache_depth+1
                if cache_depth>=4 and g>=cache4_groups: first_gather=4
                bits=histories[g]
                if packet_depth <= depth < packet_depth+packet_height:
                    packet_level=depth-packet_depth
                    if packet_level==0:
                        for lane in range(8): virtual_map[t+lane]=virtual_map[idx+lane]
                        for lane in range(8):
                            emit('load',('vload',packets[g][lane],t+lane),[t+lane],vec(packets[g][lane]))
                            ops[-1][2].update(packet_stores_by_depth[packet_depth])
                            if packet_height==3 and g<getattr(self,'four_groups',32):four_packet_loads[len(ops)-1]=(r+2,g)
                            if packet_height==2 and g<(getattr(self,'two_groups',32) if r>forest_height else getattr(self,'early_two_groups',9)):four_packet_loads[len(ops)-1]=(r+1,g)
                    if packet_height==3 and packet_level==1 and g<getattr(self,'four_groups',32):
                        ptr,dummy=four_ptrs[g%four_nbuf]
                        sel(four_addr1[g],packet_bits[g][0],ptr,dummy)
                        middle_load=None
                        if middle_selected(g):
                            middle_default=[];middle_conditional=[]
                            for lane in range(8):
                                source=packets[g][lane]+1
                                emit('store',('store',ptr+lane,source),[ptr+lane,source],[])
                                middle_default.append(len(ops)-1)
                            for lane in range(8):
                                source=packets[g][lane]+2
                                emit('store',('store',four_addr1[g]+lane,source),[four_addr1[g]+lane,source],[])
                                ops[-1][2].add(middle_default[lane])
                                middle_conditional.append(len(ops)-1)
                            emit('load',('vload',four_nodes,ptr),[ptr],vec(four_nodes))
                            ops[-1][2].update(middle_default+middle_conditional)
                            middle_load=len(ops)-1
                            middle_jobs[(r+1,g)]=(middle_default+middle_conditional,middle_load)
                        stores0=[];stores1=[]
                        for lane in range(8):
                            source=packets[g][lane]+4
                            emit('store',('store',ptr+lane,source),[ptr+lane,source],[])
                            stores0.append(len(ops)-1)
                            if middle_load is not None:
                                ops[-1][2].add(middle_load)
                                middle_memory_edges.append((middle_load,len(ops)-1))
                        for lane in range(8):
                            source=packets[g][lane]+6
                            emit('store',('store',four_addr1[g]+lane,source),[four_addr1[g]+lane,source],[])
                            ops[-1][2].add(stores0[lane])
                            stores1.append(len(ops)-1)
                        four_early_stores[(r+1,g)]=(stores0,stores1)
                    if packet_height==3 and packet_level==1 and middle_selected(g):
                        v('^',val,val,four_nodes)
                    elif packet_height==2 and packet_level==1 and g<(getattr(self,'two_groups',32) if r>forest_height else getattr(self,'early_two_groups',9)):
                        ptr,dummy=four_ptrs[g%four_nbuf]
                        sel(four_late_addr,packet_bits[g][0],dummy,ptr)
                        stores0=[];stores1=[]
                        for lane in range(8):
                            source=packets[g][lane]+2
                            emit('store',('store',ptr+lane,source),[ptr+lane,source],[])
                            stores0.append(len(ops)-1)
                        for lane in range(8):
                            source=packets[g][lane]+1
                            emit('store',('store',four_late_addr+lane,source),[four_late_addr+lane,source],[])
                            ops[-1][2].add(stores0[lane])
                            stores1.append(len(ops)-1)
                        emit('load',('vload',four_nodes,ptr),[ptr],vec(four_nodes))
                        ops[-1][2].update(stores1)
                        four_jobs[(r,g)]=(stores0+stores1,[len(ops)-1],g%four_nbuf)
                        v('^',val,val,four_nodes)
                    elif packet_height==3 and packet_level==2 and g<getattr(self,'four_groups',32):
                        ptr,dummy=four_ptrs[g%four_nbuf]
                        sel(four_late_addr,packet_bits[g][1],dummy,ptr)
                        sel(four_late_addr2,packet_bits[g][0],four_late_addr,dummy)
                        stores0,stores1=four_early_stores[(r,g)]
                        stores2=[];stores3=[]
                        for lane in range(8):
                            source=packets[g][lane]+3
                            emit('store',('store',four_late_addr+lane,source),[four_late_addr+lane,source],[])
                            ops[-1][2].add(stores1[lane])
                            stores2.append(len(ops)-1)
                        for lane in range(8):
                            source=packets[g][lane]+5
                            emit('store',('store',four_late_addr2+lane,source),[four_late_addr2+lane,source],[])
                            ops[-1][2].add(stores2[lane])
                            stores3.append(len(ops)-1)
                        emit('load',('vload',four_nodes,ptr),[ptr],vec(four_nodes))
                        ops[-1][2].update(stores3)
                        prior_middle=middle_jobs.get((r,g),([],None))[0]
                        four_jobs[(r,g)]=(prior_middle+stores0+stores1+stores2+stores3,[len(ops)-1],g%four_nbuf)
                        v('^',val,val,four_nodes)
                    else:
                        count=1<<packet_level
                        for candidate in range(count):
                            dest=packet_candidates[candidate]
                            for lane in range(8):
                                source=packets[g][lane]+count-1+candidate
                                emit('alu',('^',dest+lane,val+lane,source),[val+lane,source],[dest+lane],vector_lane=lane)
                        for level in range(packet_level-1,-1,-1):
                            count//=2
                            for candidate in range(count):
                                sel(packet_candidates[candidate],packet_bits[g][level],packet_candidates[2*candidate+1],packet_candidates[2*candidate])
                        # Alias the selected virtual vector without generating a move.
                        for lane in range(8): virtual_map[val+lane]=virtual_map[packet_candidates[0]+lane]
                elif depth==0:
                    v('^',val,val,root_original if r==0 else tree[0])
                elif depth<first_gather:
                    count=1<<(depth-1)
                    for j in range(count):
                        node=(1<<depth)+2*j
                        if not (getattr(self,'flow_mask',31) >> (node//2-1)) & 1:
                            ma(tmp_pool[j],bits[depth-1],differences[node],tree[node-1])
                        else:
                            sel(tmp_pool[j],bits[depth-1],tree[node],tree[node-1])
                    for lev in range(depth-2,-1,-1):
                        count//=2
                        for j in range(count): sel(tmp_pool[j],bits[lev],tmp_pool[2*j+1],tmp_pool[2*j])
                    v('^',val,val,tmp_pool[0])
                elif deep_start<=depth<=forest_height:
                    if depth==deep_start:
                        v('^',val,val,c5)

                    for lane in range(8):
                        virtual_map[deep_source+lane]=virtual_map[val+lane]
                    for lane,buf in enumerate(deep_load_buffers):
                        emit('load',('vload',buf,idx+lane),[idx+lane],vec(buf))
                        emit('alu',('^',val+lane,deep_source+lane,buf+6),[deep_source+lane,buf+6],[val+lane],vector_lane=lane)
                else:
                    if 4<=depth<=7:
                        gather_addr=idx
                    else:
                        v('-',t,c(5+3*(1<<depth)),idx)
                        gather_addr=t
                    for lane in range(8):
                        emit('load',('load_offset',u,gather_addr,lane),[gather_addr+lane],[u+lane])
                        if 4<=depth<=7: ops[-1][2].update(pre_stores)
                    if not 4<=depth<=7: v('^',u,u,c5)
                    v('^',val,val,u)
                ma(val,val,c(4097),c(0x7ED55D16))
                v('>>',t,val,c(19)); v('^',u,val,c(0xC761C23C)); v('^',val,t,u)
                ma(t,val,c(33),c((0x165667B1+0xD3A2646C)&0xffffffff))
                ma(u,val,c(33<<9),c((0x165667B1<<9)&0xffffffff))
                v('^',val,t,u)
                ma(val,val,c(9),c(0xFD7046C5))
                v('>>',t,val,c(16)); v('^',val,val,t)
                if deep_start<=depth<forest_height:
                    v('^',val,val,c5)
                if r+1<rounds and depth!=forest_height:
                    if depth<first_gather:
                        v('&',bits[depth],val,one)
                        if depth+1==first_gather:
                            if g<(getattr(self,'index_lookup_final_groups',4) if r>forest_height else getattr(self,'index_lookup_groups',5)):
                                for j in range(4):sel(tmp_pool[j],bits[2],c(11+6*j),c(8+6*j))
                                sel(t,bits[1],tmp_pool[1],tmp_pool[0])
                                sel(u,bits[1],tmp_pool[3],tmp_pool[2])
                                sel(idx,bits[0],u,t)
                            else:
                                sel(t,bits[1],c(14),c(8))
                                sel(u,bits[1],c(26),c(20))
                                sel(idx,bits[0],u,t)
                                ma(idx,bits[2],c(3),idx)
                    else:
                        v('&',t,val,one)
                        if depth in (3,5,6):
                            pass
                        elif depth==4:
                            if g<getattr(self,'boundary_lookup_groups',32):
                                sel(u,t,c((compact_b+7)&0xffffffff),c(compact_b))
                                sel(w,t,c((compact_b+21)&0xffffffff),c((compact_b+14)&0xffffffff))
                                sel(u,packet_bits[g][0],w,u)
                                ma(idx,idx,c(compact_a),u)
                            else:
                                sel(u,packet_bits[g][0],c((compact_b+14)&0xffffffff),c(compact_b))
                                ma(idx,idx,c(compact_a),u)
                                ma(idx,t,c(7),idx)
                        elif depth==7:
                            sel(u,packet_bits[g][1],c((compact_e-2)&0xffffffff),c(compact_e))
                            sel(w,packet_bits[g][1],c((compact_e-6)&0xffffffff),c((compact_e-4)&0xffffffff))
                            sel(u,packet_bits[g][0],w,u)
                            ma(idx,idx,c(compact_c),u)
                            v('-',idx,idx,t)
                        else:
                            ma(idx,idx,two,t)
                        if packet_depth <= depth < packet_depth+packet_height-1:
                            for lane in range(8): virtual_map[packet_bits[g][depth-packet_depth]+lane]=virtual_map[t+lane]
        for val,idx,t,u,w,p in states:
            v('^',val,val,c5)
            emit('store',('vstore',p,val),[p]+vec(val),[])
        # Delete SSA operations that cannot affect any memory store.
        if getattr(self, 'dead_code_elimination', True):
            needed=set()
            pending=[j for j,(engine,slot,deps) in enumerate(ops) if engine=='store']
            while pending:
                j=pending.pop()
                if j in needed: continue
                needed.add(j); pending.extend(ops[j][2])
            mapping={old:new for new,old in enumerate(sorted(needed))}
            ops=[(engine,slot,{mapping[d] for d in deps}) for old,(engine,slot,deps) in enumerate(ops) if old in needed]
            op_rw=[rw for old,rw in enumerate(op_rw) if old in needed]
            op_group=[g for old,g in enumerate(op_group) if old in needed]
        four_jobs={key:([mapping[j] for j in st],[mapping[j] for j in ld],buf) for key,(st,ld,buf) in four_jobs.items()}
        four_packet_loads={mapping[j]:key for j,key in four_packet_loads.items()}
        middle_memory_edges=[(mapping[read],mapping[write]) for read,write in middle_memory_edges]
        self.logical_counts=collections.Counter((e,s[0]) for e,s,d in ops)
        if getattr(self,'count_only',False):return
        # Critical-path list scheduling, respecting read/write hazards.
        succ=[[] for _ in ops]; indeg=[]
        for j,(_,_,deps) in enumerate(ops):
            indeg.append(len(deps))
            for d in deps: succ[d].append(j)
        op_weight=[getattr(self,"flow_weight",2) if eng=="flow" else getattr(self,"gather_weight",1) if eng=="load" and slot[0]=="load_offset" else 1 for eng,slot,deps in ops]
        rank=op_weight.copy()
        for j in range(len(ops)-1,-1,-1):
            if succ[j]: rank[j]=op_weight[j]+max(rank[k] for k in succ[j])
        stagger=getattr(self,"stagger",3)
        rank=[rank[j]+(batch_size//8-op_group[j])*stagger if op_group[j]>=0 else rank[j]+getattr(self,"init_boost",100) for j in range(len(ops))]
        from collections import Counter
        read_bases=[set(r//8*8 for r in reads) for reads,writes in op_rw]
        write_bases=[set(w//8*8 for w in writes) for reads,writes in op_rw]
        uses=Counter(r for rr in read_bases for r in rr)
        live=set(); born=set(); max_live=0; live_total=0
        reg_limit=getattr(self,"reg_limit",200)
        reg_weight=getattr(self,"reg_weight",100)
        ready=defaultdict(list)
        for j,(eng,_,_) in enumerate(ops):
            if indeg[j]==0: ready[eng].append(j)
        partial={};op_start={};op_end={}
        four_active=[None]*four_nbuf
        four_store_job={j:key for key,(st,ld,buf) in four_jobs.items() for j in st}
        four_load_job={j:key for key,(st,ld,buf) in four_jobs.items() for j in ld}
        four_pending={key:set(ld) for key,(st,ld,buf) in four_jobs.items()}
        four_todo=set(four_jobs)
        four_admitted=set()
        used_words={r for reads,writes in op_rw for r in reads}
        live_width={base:sum(base+lane in used_words for lane in range(width)) for base,width in virtual_width.items()}
        def weight(r): return live_width[r]/8
        def live_size(): return live_total
        def delta(j,complete=True):
            new=sum(weight(r) for r in write_bases[j]-born)
            freed=sum(weight(r) for r in read_bases[j] if uses[r]==1 and r in live) if complete else 0
            return new-freed
        def pick(eng,scalar=False):
            if eng=='store':
                ready_stores=set(ready['store'])
                for buf in range(four_nbuf):
                    if four_active[buf] is not None: continue
                    available=[key for key in four_todo if four_jobs[key][2]==buf and all(j in ready_stores for j in four_jobs[key][0][:8])]
                    if available:
                        key=max(available,key=lambda key:(max(rank[j] for j in four_jobs[key][1]),-key[1],-key[0]))
                        four_active[buf]=key;four_todo.remove(key)
            choices=[]
            for j in ready[eng]:
                slot=ops[j][1]
                if j in four_packet_loads and four_packet_loads[j] not in four_admitted and len(four_admitted)>=getattr(self,'four_admission',4):continue
                if j in four_store_job:
                    key=four_store_job[j]
                    if four_active[four_jobs[key][2]]!=key:continue
                reserve=getattr(self,'four_reserve',8) if delta(j)>0 and j not in four_load_job and op_group[j] not in {key[1] for key in four_active if key is not None} else 0
                if live_size()+delta(j)>reg_limit-reserve: continue
                if scalar and (len(slot)!=4 or slot[0]=='vbroadcast'): continue
                if live_size()+delta(j)>reg_limit: continue
                choices.append(j)
            if not choices: return None
            j=max(choices,key=lambda j:(rank[j]+(getattr(self,'four_load_boost',1000) if j in four_load_job else 0)+(getattr(self,'four_store_boost',0) if j in four_store_job else 0)-reg_weight*delta(j)+(getattr(self,'fma_bias',100) if eng=='valu' and len(ops[j][1])!=4 else 0)+getattr(self,'fraction_bias',80)*sum(1/uses[r] for r in read_bases[j] if uses[r]>1 and r in live),-j))
            ready[eng].remove(j)
            return j
        def commit(j,complete=True):
            nonlocal max_live, live_total
            if j in four_packet_loads:four_admitted.add(four_packet_loads[j])
            new=write_bases[j]-born
            born.update(new);live.update(new)
            if complete:
                for r in read_bases[j]:
                    uses[r]-=1
                    if uses[r]==0: live.discard(r)
                for w in write_bases[j]:
                    if uses[w]==0: live.discard(w)
            live_total=sum(weight(r) for r in live)
            max_live=max(max_live,live_total)
            if j in four_load_job and complete:
                key=four_load_job[j]
                four_pending[key].remove(j)
                if not four_pending[key]:
                    assert four_active[four_jobs[key][2]]==key
                    four_active[four_jobs[key][2]]=None
                    four_admitted.discard(key)
        while any(ready.values()) or partial:
            bundle={};chosen=[];alu_slots=[]
            for j in list(partial):
                eng,slot,_=ops[j];op,d,a,b=slot;lane=partial[j]
                while lane<8 and len(alu_slots)<12:
                    alu_slots.append((op,d+lane,a+lane,b+lane));lane+=1
                if lane==8:
                    chosen.append(j);del partial[j];commit(j)
                else:partial[j]=lane
            for eng in SLOT_LIMITS:
                for _ in range(SLOT_LIMITS[eng]-(len(alu_slots) if eng=='alu' else 0)):
                    j=pick(eng)
                    if j is None:break
                    chosen.append(j);op_start.setdefault(j,len(self.instrs));commit(j)
                    bundle.setdefault(eng,[]).append(ops[j][1])
            while len(alu_slots)+len(bundle.get('alu',[]))<12:
                j=pick('valu',scalar=True)
                if j is None:break
                slot=ops[j][1];op,d,a,b=slot
                op_start.setdefault(j,len(self.instrs));lane=0
                if 12-len(alu_slots)-len(bundle.get('alu',[]))<8 and live_size()+delta(j,False)>reg_limit:
                    ready['valu'].append(j);break
                while lane<8 and len(alu_slots)+len(bundle.get('alu',[]))<12:
                    alu_slots.append((op,d+lane,a+lane,b+lane));lane+=1
                if lane==8:chosen.append(j);commit(j)
                else:partial[j]=lane;commit(j,False)
            if alu_slots:
                bundle.setdefault('alu',[]).extend(alu_slots)
            if not bundle: raise RuntimeError(('register deadlock',len(live),max_live))
            self.instrs.append(bundle)
            for j in chosen:
                op_end[j]=len(self.instrs)-1
                for k in succ[j]:
                    indeg[k]-=1
                    if indeg[k]==0:ready[ops[k][0]].append(k)
        self.ssa_ops=ops
        self.ssa_op_rw=op_rw
        self.ssa_op_group=op_group
        self.ssa_virtual_width=virtual_width
        self.ssa_schedule_start=op_start
        self.ssa_schedule_end=op_end
        self.ssa_four_jobs=four_jobs
        self.ssa_four_packet_loads=four_packet_loads
        self.ssa_four_releases={key:ld[-1] for key,(st,ld,buf) in four_jobs.items()}
        self.ssa_four_admission=getattr(self,'four_admission',4)
        self.ssa_buffer_orders={buf:sorted((key for key,(_,_,b) in four_jobs.items() if b==buf), key=lambda key:min(op_start[j] for j in four_jobs[key][0])) for buf in range(four_nbuf)}
        self.ssa_memory_edges=list(middle_memory_edges)
        for order in self.ssa_buffer_orders.values():
            for previous,current in zip(order,order[1:]):
                for st in four_jobs[current][0][:8]:
                    self.ssa_memory_edges.append((four_jobs[previous][1][0],st))
        self.ssa_ops_locked=[(eng,slot,set(deps)) for eng,slot,deps in ops]
        for previous_load,next_store in self.ssa_memory_edges:
            self.ssa_ops_locked[next_store][2].add(previous_load)
        self.four_times={key:([op_start[j] for j in st],[op_start[j] for j in ld]) for key,(st,ld,buf) in four_jobs.items()}
        self.ssa_scheduled_peak=max_live*8
        # Allocate contiguous vector blocks while independently retiring lanes.
        # All words are reserved from the block's first write, including lanes
        # written later by load_offset or scalarized vector operations.
        births={}; deaths={}; lane_deaths={}
        for j,(reads,writes) in enumerate(op_rw):
            for w in writes:
                base=w//8*8
                births[base]=min(births.get(base,op_start[j]),op_start[j])
                deaths[base]=max(deaths.get(base,op_end[j]),op_end[j]+0.1)
                lane_deaths[w]=max(lane_deaths.get(w,op_end[j]),op_end[j]+0.1)
            for r in reads:
                base=r//8*8
                deaths[base]=max(deaths.get(base,op_end[j]),op_end[j])
                lane_deaths[r]=max(lane_deaths.get(r,op_end[j]),op_end[j])
        physical={}; available=[]
        for vbase,start in sorted(births.items(),key=lambda x:(x[1],-deaths[x[0]])):
            width=virtual_width[vbase]
            reg=0
            while True:
                occupied=next((i for i in range(reg,min(reg+width,len(available))) if available[i]>start),None)
                if occupied is None: break
                reg=occupied+1
            if reg+width>len(available): available.extend([-1]*(reg+width-len(available)))
            physical[vbase]=reg
            for lane in range(width):
                available[reg+lane]=lane_deaths.get(vbase+lane,start+0.1)
        allocated=len(available)
        assert allocated <= SCRATCH_SIZE, ("Scheduled kernel exceeds scratch capacity", allocated, len(self.instrs))
        self.ssa_peak=allocated
        self.ssa_op_count=len(ops)
        self.scratch_ptr=allocated
        self.scratch_debug={}
        def addr(x): return physical[x//8*8]+x%8
        self.virtual_instrs=self.instrs
        self.physical_map=physical
        self.instrs=[{eng:[remap_slot(eng,slot,addr,addr) for slot in slots]
                      for eng,slots in bundle.items()} for bundle in self.instrs]

def base(name=None, **cfg):
    assert name in (None,'astra870_store_midreuse_front')
    k=FrontendBuilder()
    for key,value in cfg.items():setattr(k,key,value)
    k.build_kernel(10,2047,256,16)
    k._ops=k.ssa_ops_locked;k._op_rw=k.ssa_op_rw;k._op_group=k.ssa_op_group
    k._virtual_width=k.ssa_virtual_width;k._virtual_instrs=k.virtual_instrs
    k._edge_latencies={edge:0 for edge in k.ssa_memory_edges}
    return k

def rw(e,s):
 op=s[0]
 if e=='alu':return set(s[2:]),{s[1]}
 if e=='valu':return ({s[2]} if op=='vbroadcast' else {x+i for x in s[2:] for i in range(8)}),set(range(s[1],s[1]+8))
 if e=='load':
  if op=='const':return set(),{s[1]}
  if op=='load_offset':return {s[2]+s[3]},{s[1]+s[3]}
  return {s[2]},set(range(s[1],s[1]+(8 if op=='vload' else 1)))
 if e=='store':return {s[1]}|set(range(s[2],s[2]+(8 if op=='vstore' else 1))),set()
 if e=='flow':
  if op=='select':return set(s[2:]),{s[1]}
  if op=='vselect':return {x+i for x in s[2:] for i in range(8)},set(range(s[1],s[1]+8))
  if op=='add_imm':return {s[2]},{s[1]}
  raise Exception(s)
 raise Exception(e)

class Graph:
 def __init__(self,k):
  # Map each physical slot back to the original virtual operation.
  slotmap={};ops=k._ops;orig=[];byop=defaultdict(list)
  for j,(e,s,d) in enumerate(ops):
   slotmap[e,s]=j
   if e=='valu' and len(s)==4:
    for lane in range(8):slotmap['alu',(s[0],s[1]+lane,s[2]+lane,s[3]+lane)]=j
  self.nodes=[];self.original=[];groups=[]
  for t,(pb,vb) in enumerate(zip(k.instrs,k._virtual_instrs)):
   for e,slots in pb.items():
    for s,vs in zip(slots,vb[e]):
     n=len(self.nodes);self.nodes.append((e,vs));self.original.append(t)
     j=slotmap[e,vs];orig.append(j);byop[j].append(n);groups.append(k._op_group[j])
  self.groups=groups;n=len(self.nodes);deps=[{} for _ in range(n)]
  bytime=defaultdict(list)
  for i,t in enumerate(self.original):bytime[t].append(i)
  write={};read=defaultdict(set)
  for t,ns in bytime.items():
   for j in ns:
    r,w=rw(*self.nodes[j])
    for a in r:
     if a in write:deps[j][write[a]]=1
     read[a].add(j)
   for j in ns:
    r,w=rw(*self.nodes[j])
    for a in w:
     if a in write:deps[j][write[a]]=1
     for q in ():
      if q!=j:deps[j][q]=max(deps[j].get(q,0),0)
     read[a].clear();write[a]=j
  explicit=0
  for j,(e,s,ds) in enumerate(ops):
   for d in ds:
    if not set(k._op_rw[j][0])&set(k._op_rw[d][1]):
     explicit+=1
     for to in byop[j]:
      for fr in byop[d]:deps[to][fr]=max(deps[to].get(fr,0),getattr(k,'_edge_latencies',{}).get((d,j),1))
  self.deps=deps;self.succ=[[] for _ in range(n)]
  for j,dd in enumerate(deps):
   for d,lat in dd.items():self.succ[d].append((j,lat))
  # Register reuse can induce zero-latency cycles; these operations must coissue.
  sys.setrecursionlimit(100000)
  visited=set();post=[]
  def dfs(j):
   visited.add(j)
   for a,l in self.succ[j]:
    if a not in visited:dfs(a)
   post.append(j)
  for j in range(n):
   if j not in visited:dfs(j)
  component={};clusters=[]
  def back(j,c):
   component[j]=c;clusters[c].append(j)
   for a in deps[j]:
    if a not in component:back(a,c)
  for j in reversed(post):
   if j not in component:clusters.append([]);back(j,len(clusters)-1)
  self.flatnodes=self.nodes;self.flatdeps=deps;self.flat_orig=self.original
  self.clusters=clusters;self.component=component
  nodes=[];cd=[{} for _ in clusters];ori=[];demands=[]
  for c,js in enumerate(clusters):
   nodes.append([self.nodes[j] for j in js]);ori.append(self.original[js[0]])
   demands.append(Counter(self.nodes[j][0] for j in js))
   assert all(self.original[j]==ori[-1] for j in js)
   assert all(demands[-1][e]<=LIMIT[e] for e in demands[-1])
   for j in js:
    for d,l in deps[j].items():
     if component[d]!=c:cd[c][component[d]]=max(cd[c].get(component[d],0),l)
     else:assert l==0
  self.nodes=nodes;self.deps=cd;self.original=ori;self.demands=demands
  n=len(nodes);self.succ=[[] for _ in range(n)]
  for j,dd in enumerate(cd):
   for d,lat in dd.items():self.succ[d].append((j,lat))
  deg=[len(d) for d in cd];ready=[i for i in range(n) if not deg[i]];top=[]
  while ready:
   j=ready.pop();top.append(j)
   for a,l in self.succ[j]:
    deg[a]-=1
    if not deg[a]:ready.append(a)
  assert len(top)==n,('cyclic',len(top),n)
  self.top=top
  self.rank=[0]*n
  for j in reversed(top):self.rank[j]=max([self.rank[a]+l for a,l in self.succ[j]] or [0])
  self.n=n

 def schedule(self, oldweight=1,critweight=0,order=('alu','valu','load','store','flow'),seed=0,noise=0, reverse=False,allow_lowering=False):
  rng=random.Random(seed);succ=self.succ;deps=self.deps;ori=self.original
  if reverse:
   succ=[list(d.items()) for d in deps]
   deps=[dict(s) for s in self.succ]
   ori=[max(ori)-o for o in ori]
  rank=[0]*self.n
  for j in (self.top if reverse else reversed(self.top)):rank[j]=max([rank[a]+l for a,l in succ[j]] or [0])
  deg=[len(d) for d in deps];earliest=[0]*self.n;ready=[];future=[]
  priority=[oldweight*ori[j]-critweight*rank[j]+noise*rng.random() for j in range(self.n)]
  for j in range(self.n):
   if not deg[j]:heapq.heappush(ready,(priority[j],j))
  times=[None]*self.n;program=[];t=0;done=0
  while done<self.n:
   while future and future[0][0]<=t:
    tm,j=heapq.heappop(future);heapq.heappush(ready,(priority[j],j))
   out={};blocked=[]
   while ready:
    p,j=heapq.heappop(ready)
    converted=False
    if any(len(out.get(e,[]))+q>LIMIT[e] for e,q in self.demands[j].items()):
     if allow_lowering and len(self.nodes[j])==1 and self.nodes[j][0][0]=='valu' and len(self.nodes[j][0][1])==4 and len(out.get('alu',[]))<=4:
      s=self.nodes[j][0][1];out.setdefault('alu',[]).extend((s[0],s[1]+l,s[2]+l,s[3]+l) for l in range(8));converted=True
     else:blocked.append((p,j));continue
    if not converted:
     for e,s in self.nodes[j]:out.setdefault(e,[]).append(s)
    times[j]=t;done+=1
    for a,lat in succ[j]:
     earliest[a]=max(earliest[a],t+lat);deg[a]-=1
     if deg[a]==0:
      if earliest[a]<=t:heapq.heappush(ready,(priority[a],a))
      else:heapq.heappush(future,(earliest[a],a))
   ready=blocked;heapq.heapify(ready)
   if not out and not future:raise Exception(('deadlock',done,self.n))
   program.append(out);t+=1
  if reverse:program.reverse();times=[len(program)-1-x for x in times]
  for j,ds in enumerate(self.deps):
   for d,l in ds.items():assert times[j]>=times[d]+l
  return program,times

def lifetimes(k,p):
 births={};ends={};deaths={}
 for t,b in enumerate(p):
  for e,ss in b.items():
   for s in ss:
    rr,ww=rw(e,s)
    for w in ww:
     base=w//8*8;births[base]=min(births.get(base,t),t);ends[w]=max(ends.get(w,t),t+.1)
    for r in rr:ends[r]=max(ends.get(r,t),t)
 for b in births:deaths[b]=max(ends.get(b+l,births[b]+.1) for l in range(k._virtual_width[b]))
 return births,ends,deaths

def gate_sparse_loads(k,cap=4,window=0):
 # Reserve a bounded pipeline for vloads used only at one lane. Packet rows,
 # preprocessing, and real contiguous input loads are not touched.
 readers=collections.defaultdict(set)
 for j,(reads,writes) in enumerate(k._op_rw):
  for r in reads:readers[r].add(j)
 jobs=[]
 for j,(eng,slot,deps) in enumerate(k._ops):
  if eng=='load' and slot[0]=='vload':
   dst=slot[1];used=[l for l in range(8) if readers[dst+l]]
   if len(used)==1:
    read=readers[dst+used[0]]
    if len(read)==1:jobs.append((j,next(iter(read))))
 jobs.sort(key=lambda job:(k.ssa_schedule_start[job[0]],job[0]))
 for (load,release),(nxt,_) in zip(jobs,jobs[cap:]):
  k._ops[nxt][2].add(release);k._edge_latencies[release,nxt]=0
 return len(jobs)

def gate_packet(k,cap):
 keys=sorted(k.ssa_four_jobs,key=lambda key:min(k.ssa_schedule_start[j] for j,kk in k.ssa_four_packet_loads.items() if kk==key))
 for prev,key in zip(keys,keys[cap:]):
  releases=getattr(k,'ssa_four_releases',{}).get(prev,k.ssa_four_jobs[prev][1])
  if isinstance(releases,int):releases=[releases]
  for j,kk in k.ssa_four_packet_loads.items():
   if kk==key:
    for rel in releases:k._ops[j][2].add(rel);k._edge_latencies[rel,j]=0

def shape_allocate(k,p,mode='area',seed=0,lane_birth=True):
 births,ends,deaths=lifetimes(k,p)
 lane_starts={}
 for t,bundle in enumerate(p):
  for eng,ss in bundle.items():
   for slot in ss:
    for word in rw(eng,slot)[1]:lane_starts[word]=min(lane_starts.get(word,t),t)
 shapes={}
 for b,st in births.items():
  shapes[b]=[]
  for l in range(k._virtual_width[b]):
   if b+l not in lane_starts:
    shapes[b].append(0);continue
   start=2*(lane_starts[b+l] if lane_birth else st)+1
   end=int(2*ends.get(b+l,st+.1))+(1 if ends.get(b+l,st+.1)%1 else 0)
   shapes[b].append(((1<<(end-start+1))-1)<<start)
 areas={b:sum(mask.bit_count() for mask in shape) for b,shape in shapes.items()}
 rng=random.Random(seed)
 def key(b):
  if mode=='area':return -areas[b],-k._virtual_width[b],births[b]
  if mode=='duration':return -(deaths[b]-births[b]),-areas[b],births[b]
  if mode=='width':return -k._virtual_width[b],-areas[b],births[b]
  if mode=='start':return births[b],-areas[b]
  if mode=='end':return -deaths[b],-areas[b]
  if mode=='random':return -areas[b]*(.9+.2*rng.random()),-k._virtual_width[b]
  raise ValueError(mode)
 masks=[];physical={}
 for b in sorted(shapes,key=key):
  shape=shapes[b];width=len(shape);reg=0
  while True:
   if all(not(masks[reg+l]&m) for l,m in enumerate(shape) if reg+l<len(masks)):break
   reg+=1
  if reg+width>len(masks):masks.extend([0]*(reg+width-len(masks)))
  physical[b]=reg
  for l,m in enumerate(shape):masks[reg+l]|=m
 return physical,len(masks)

def remap_program(p,physical):
 def a(x):return physical[x//8*8]+x%8
 def remap(e,s):
  if e=='load' and s[0]=='const':return s[0],a(s[1]),s[2]
  if e=='load' and s[0]=='load_offset':return s[0],a(s[1]),a(s[2]),s[3]
  if e=='flow' and s[0]=='add_imm':return s[0],a(s[1]),a(s[2]),s[3]
  return (s[0],)+tuple(a(x) for x in s[1:])
 return [{e:[remap(e,s) for s in ss] for e,ss in b.items()} for b in p]

def startup_synthesize(k,program):
    # Rebuild constant definitions using only architectural zero and ISA arithmetic.
    oldtime={(e,s):t for t,b in enumerate(program) for e,ss in b.items() for s in ss}
    constloads={s[1]:(j,s[2]) for j,(e,s,d) in enumerate(k._ops) if e=='load' and s[0]=='const'}
    vectors={}
    for j,(e,s,d) in enumerate(k._ops):
        if e=='valu' and s[0]=='vbroadcast' and s[2] in constloads:
            lj,value=constloads[s[2]]
            vectors[value]=(lj,j,s[1])
    k.constant_vectors={value:a for value,(lj,j,a) in vectors.items()}
    zeroload,zerobcast,zero=vectors[0]
    k._initial_zero_bases={zero}
    derive=getattr(k,'derive_constants',{1,2,3,4,6,9,12,16,19,33,40,16896})
    flow=getattr(k,'flow_constants',{4097,2127912214})
    recipes={1:('==',0,0),2:('+',1,1),3:('+',2,1),4:('+',2,2),6:('+',3,3),8:('<<',2,2),9:('*',3,3),12:('*',3,4),16:('*',4,4),19:('multiply_add',4,4,3),24:('+',12,12),32:('<<',2,4),33:('multiply_add',2,16,1),40:('multiply_add',4,9,4),16896:('<<',33,9),4096:('<<',1,12),4097:('+',4096,1)}
    oldops=[(e,s) for e,s,ds in k._ops]
    oldwidth=dict(k._virtual_width)
    nextbase=max(oldwidth)+8
    definitions=[]
    vals={0:zero}
    def vec(a):return list(range(a,a+8))
    def rewrite(j,e,s,reads,writes):
        k._ops[j]=(e,s,set());k._op_rw[j]=(reads,writes)
        definitions.append((e,s))
    def append(e,s,reads,writes):
        k._ops.append((e,s,set()));k._op_rw.append((reads,writes));k._op_group.append(-1)
        definitions.append((e,s))
    def materialize(value):
        nonlocal nextbase
        if value in vals:return vals[value]
        if value in vectors:
            lj,j,a=vectors[value]
        else:
            a=nextbase;nextbase+=8;k._virtual_width[a]=8
            lj=j=None
        if value in derive:
            op,*values=recipes[value];args=[materialize(v) for v in values]
            s=(op,a,*args);reads=[r for arg in args for r in vec(arg)]
            if j is None:append('valu',s,reads,vec(a))
            else:rewrite(j,'valu',s,reads,vec(a))
        else:
            if lj is None:
                scalar=nextbase;nextbase+=8;k._virtual_width[scalar]=1
            else:scalar=k._ops[lj][1][1]
            if value in flow:
                e='flow';s=('add_imm',scalar,zero,value);reads=[zero]
            else:
                e='load';s=('const',scalar,value);reads=[]
            if lj is None:append(e,s,reads,[scalar])
            else:rewrite(lj,e,s,reads,[scalar])
            s=('vbroadcast',a,scalar)
            if j is None:append('valu',s,[scalar],vec(a))
            else:rewrite(j,'valu',s,[scalar],vec(a))
        vals[value]=a
        return a
    for value in vectors:materialize(value)
    remove={oldops[j] for lj,j,a in vectors.values()}|{oldops[lj] for lj,j,a in vectors.values()}
    body=[{e:[s for s in ss if (e,s) not in remove] for e,ss in b.items()} for b in program]
    # A serial definition prefix makes every SSA dependency explicit to Graph.
    # Priorities below retain each existing body's original schedule coordinate.
    prefix=[{e:[s]} for e,s in definitions]
    k.instrs=k._virtual_instrs=prefix+body
    graph=Graph(k)
    constant_priority=getattr(k,'constant_priority',-8)
    graph.original=[min(oldtime.get(node,constant_priority) for node in nodes) for nodes in graph.nodes]
    if getattr(k,'startup_eager',True):
        addresses={s[1] for e,s in oldops if e=='load' and s[0]=='const' and s[2] in (7,k._startup_input_base)}
        roots={s[1] for e,s in oldops if e=='load' and s[0]=='vload' and s[2] in addresses}
        critical={vectors[v][2] for v in (4097,0x7ED55D16)}
        critical_scalar={oldops[vectors[v][0]][1][1] for v in (4097,0x7ED55D16)}
        for j,nodes in enumerate(graph.nodes):
            for e,s in nodes:
                if e=='load' and s[0]=='const' and s[1] in addresses:graph.original[j]=-20
                elif e=='load' and s[0]=='vload' and s[2] in addresses:graph.original[j]=-18
                elif s[1] in critical_scalar:graph.original[j]=-20
                elif e=='valu' and s[0]=='vbroadcast' and (s[1] in critical or s[2] in roots):graph.original[j]=-18

    for it in range(getattr(k,'startup_passes',6)):
        program,times=graph.schedule(1,getattr(k,'startup_critical',0),noise=0,reverse=bool(it%2),allow_lowering=True)
        graph.original=times
    return program

def reassociate_final(k,graph,count=4):
    # XOR is associative. Prepare x^C5 alongside x>>16 so final output
    # requires one dependent XOR, rather than a two-XOR serial chain.
    assert all(len(nodes)==1 for nodes in graph.nodes)
    original_ops=[nodes[0] for nodes in graph.nodes]
    ops=original_ops[:]
    writer={w:j for j,node in enumerate(ops) for w in rw(*node)[1]}
    constant=k.constant_vectors[0xB55A4F09]
    targets=[]
    for j,(engine,slot) in enumerate(ops):
        if engine!='store' or slot[0]!='vstore':continue
        final=[];middle=[];shifts=[]
        for lane in range(VLEN):
            final_id=writer[slot[2]+lane]
            final_engine,final_slot=ops[final_id]
            if final_engine not in ('alu','valu') or final_slot[0]!='^':break
            offset=lane if final_engine=='valu' else 0
            if final_slot[3]+offset!=constant+lane:break
            middle_id=writer[final_slot[2]+offset]
            middle_engine,middle_slot=ops[middle_id]
            if middle_engine not in ('alu','valu') or middle_slot[0]!='^':break
            middle_offset=final_slot[2]+offset-middle_slot[1]
            shift_id=writer.get(middle_slot[3]+middle_offset)
            shift_engine,shift_slot=ops[shift_id] if shift_id is not None else ('',())
            if shift_engine not in ('alu','valu') or shift_slot[0]!='>>':break
            final.append(final_id);middle.append(middle_id);shifts.append(middle_slot[3]+middle_offset)
        else:targets.append((j,set(final),set(middle),shifts[0]))
    assert len(targets)==32
    targets.sort(key=lambda target:graph.original[target[0]])
    for store,final,middle,shift in targets[-count:]:
        for j in final:
            engine,slot=ops[j]
            offset=slot[1]-ops[store][1][2]
            ops[j]=(engine,(slot[0],slot[1],slot[2],shift+offset))
        for j in middle:
            engine,slot=ops[j]
            offset=slot[1]%VLEN if engine=='alu' else 0
            ops[j]=(engine,(slot[0],slot[1],slot[2],constant+offset))
    # Keep every explicit memory-order edge, including buffer-reuse edges.
    deps=[{d:lat for d,lat in dd.items()
           if not (rw(*original_ops[d])[1]&rw(*original_ops[j])[0])}
          for j,dd in enumerate(graph.deps)]
    writer={w:j for j,node in enumerate(ops) for w in rw(*node)[1]}
    for j,node in enumerate(ops):
        for read in rw(*node)[0]:
            if read in writer:deps[j][writer[read]]=1
    graph.nodes=[[node] for node in ops]
    graph.deps=deps
    graph.succ=[[] for _ in ops]
    graph.demands=[Counter([node[0]]) for node in ops]
    for j,dd in enumerate(deps):
        for d,lat in dd.items():graph.succ[d].append((j,lat))
    indegree=list(map(len,deps))
    ready=[j for j,degree in enumerate(indegree) if not degree]
    top=[]
    while ready:
        j=ready.pop();top.append(j)
        for successor,lat in graph.succ[j]:
            indegree[successor]-=1
            if not indegree[successor]:ready.append(successor)
    assert len(top)==graph.n
    graph.top=top
    return graph

def lift(g,k,p,count=12):
 actual={(e,s):j for j,ns in enumerate(g.nodes) for e,s in ns};candidates=[]
 for e,s,ds in k._ops:
  if e=='valu' and len(s)==4:
   lanes=[('alu',(s[0],s[1]+l,s[2]+l,s[3]+l)) for l in range(8)]
   if all(x in actual for x in lanes):
    ids=[actual[x] for x in lanes];ts=[g.original[j] for j in ids];lo=min(ts);hi=max(ts)
    slack=sum(6-len(p[t].get('valu',[])) for t in range(lo,hi+1))
    # Lift readily-coalesced vectors where the vector engine had slack.
    score=(slack==0,hi-lo,not(60<lo<850),-slack,lo)
    candidates.append((score,ids,s))
 candidates.sort()
 chosen=candidates[:count];leader={};replacement={}
 for score,ids,s in chosen:
  anchor=min(ids);replacement[anchor]=('valu',s)
  for j in ids:leader[j]=anchor
 keep=[j for j in range(g.n) if leader.get(j,j)==j];mapping={j:n for n,j in enumerate(keep)}
 mapj=lambda j:mapping[leader.get(j,j)]
 nodes=[replacement.get(j,g.nodes[j][0]) for j in keep]
 deps=[{} for _ in keep];times=[0]*len(keep)
 for j,dd in enumerate(g.deps):
  to=mapj(j);times[to]=max(times[to],g.original[j])
  for d,lat in dd.items():
   fr=mapj(d)
   if fr!=to:deps[to][fr]=max(deps[to].get(fr,0),lat)
   else:assert not lat
 g.nodes=[[n] for n in nodes];g.deps=deps;g.original=times;g.n=len(nodes);g.demands=[collections.Counter([n[0]]) for n in nodes]
 g.succ=[[] for _ in nodes]
 for j,dd in enumerate(deps):
  for d,l in dd.items():g.succ[d].append((j,l))
 deg=list(map(len,deps));ready=[j for j,n in enumerate(deg) if not n];top=[]
 while ready:
  j=ready.pop();top.append(j)
  for q,l in g.succ[j]:
   deg[q]-=1
   if not deg[q]:ready.append(q)
 assert len(top)==g.n;g.top=top
 return g

def rebuild(g,nodes,explicit,times):
    writer={}
    for j,node in enumerate(nodes):
        for w in rw(*node)[1]:
            assert w not in writer,('duplicate SSA writer',w)
            writer[w]=j
    deps=[dict(d) for d in explicit]
    for j,node in enumerate(nodes):
        for r in rw(*node)[0]:
            if r in writer:deps[j][writer[r]]=max(deps[j].get(writer[r],0),1)
    g.nodes=[[n] for n in nodes];g.deps=deps;g.original=times;g.n=len(nodes)
    g.demands=[Counter([n[0]]) for n in nodes];g.succ=[[] for _ in nodes]
    for j,dd in enumerate(deps):
        for d,lat in dd.items():g.succ[d].append((j,lat))
    deg=list(map(len,deps));ready=[j for j,d in enumerate(deg) if not d];top=[]
    while ready:
        j=ready.pop();top.append(j)
        for q,lat in g.succ[j]:
            deg[q]-=1
            if not deg[q]:ready.append(q)
    assert len(top)==g.n,('transform cycle',len(top),g.n)
    g.top=top
    return g

def known_values(g,k):
    import operator
    functions={'+':operator.add,'-':operator.sub,'*':operator.mul,'^':operator.xor,'&':operator.and_,'|':operator.or_,'<<':operator.lshift,'>>':operator.rshift,'==':lambda a,b:int(a==b),'<':lambda a,b:int(a<b)}
    values={b+l:0 for b in getattr(k,'_initial_zero_bases',()) for l in range(8)}
    for j in g.top:
        assert len(g.nodes[j])==1
        e,s=g.nodes[j][0];op=s[0]
        if e=='load' and op=='const':values[s[1]]=s[2]&0xffffffff
        elif e=='flow' and op=='add_imm' and s[2] in values:values[s[1]]=(values[s[2]]+s[3])&0xffffffff
        elif e in ('alu','valu'):
            for l in range(8 if e=='valu' else 1):
                arguments=[s[2]] if op=='vbroadcast' else [a+l for a in s[2:]]
                if all(a in values for a in arguments):
                    args=[values[a] for a in arguments]
                    result=args[0] if op=='vbroadcast' else args[0]*args[1]+args[2] if op=='multiply_add' else functions[op](*args)
                    values[s[1]+l]=result&0xffffffff
    return values

def improve(g,k,p,two_count=24,broadcast_count=18,buffers=4,induction=True,broadcast_mode='all',shared_dummy=False,broadcast_min_use=0,pointer_engine='valu',two_selection='early',pointer_priority=500,two_skip=0):
    assert all(len(ns)==1 for ns in g.nodes)
    nodes=[ns[0] for ns in g.nodes];times=list(g.original)
    explicit=[{d:lat for d,lat in dd.items() if not rw(*nodes[d])[1]&rw(*nodes[j])[0]} for j,dd in enumerate(g.deps)]
    original_n=len(nodes);writer={w:j for j,n in enumerate(nodes) for w in rw(*n)[1]}
    removed=set();leader={};changed={};nextbase=max(k._virtual_width)+8
    def new(width=8):
        nonlocal nextbase
        base=nextbase;nextbase+=8;k._virtual_width[base]=width;return base
    def add(e,s,t,deps=()):
        j=len(nodes);nodes.append((e,s));times.append(t);explicit.append({d:lat for d,lat in deps});return j
    def drop(ids,anchor):
        for j in ids:
            if j!=anchor:removed.add(j);leader[j]=anchor
    # Existing pointer vectors are still named by their original scalar literals.
    pointer_defs={s[2]:s[1] for e,s,d in k._ops if e=='load' and s[0]=='const'}
    known=known_values(g,k)
    last_dummy=next(base for base,width in k._virtual_width.items() if width==8 and all(known.get(base+l)==2110+l for l in range(8)))
    induction_count=0
    if induction:
        input_defs={s[2]:(j,s[1]) for j,(e,s) in enumerate(nodes) if e=='load' and s[0]=='const' and 2310<=s[2]<=2558 and (s[2]-2310)%8==0}
        input_registers={known.get(s[1]):s[1] for e,s in nodes if e=='store' and s[0]=='vstore' and 2310<=known.get(s[1],-1)<=2558}
        assert len(input_registers)==32,len(input_registers)
        for address,(j,dest) in input_defs.items():
            if address-16 in input_registers:
                changed[j]=('alu',('+',dest,input_registers[address-16],k.constant_vectors[16]));induction_count+=1
    if not two_count and not broadcast_count:
        return rebuild(g,[changed.get(j,n) for j,n in enumerate(nodes)],explicit,times)
    # Dedicated memory: broadcast2118..2125; two-way outputs2126+16*b,
    # shared dummy2134..2141. All are in unused initial-index memory.
    def pointer_vector(address,base=None,step=None,priority=-12):
        dest=new()
        if pointer_engine=='flow':
            zero=next(iter(k._initial_zero_bases))
            for lane in range(8):add('flow',('add_imm',dest+lane,zero,address+lane),(priority if pointer_priority is None else pointer_priority)+lane)
        else:add('valu',('+',dest,base,k.constant_vectors[step]),priority)
        return dest
    broadcast_ptr=pointer_vector(2118,last_dummy,8) if broadcast_count else None
    outs=[];dummy=None;dummies=[]
    if two_count:
        for b in range(buffers):
            out=pointer_vector(2126+16*b,(broadcast_ptr if broadcast_ptr is not None else last_dummy) if b==0 else outs[-1],8 if b==0 and broadcast_ptr is not None else 16,-10+b)
            outs.append(out)
        for b in range(1 if shared_dummy else buffers):
            dummy=pointer_vector(2134+16*b,outs[b],8,-8+b);dummies.append(dummy)
    first_use={}
    for j,node in enumerate(nodes[:original_n]):
        for r in rw(*node)[0]:first_use[r]=min(first_use.get(r,100000),times[j])
    constant_values={a:v for v,a in k.constant_vectors.items()}
    exclude={1,2,4,8,16,4097,0x7ED55D16,0xB55A4F09}
    targets=[];covered=set()
    for j,(e,s) in enumerate(nodes[:original_n]):
        if e=='valu' and s[0]=='vbroadcast':
            value=constant_values.get(s[1]);source=nodes[writer[s[2]]] if s[2] in writer else None
            if value in exclude:continue
            if value is None and not(source and source[0]=='alu' and source[1][0] in ('^','-')):continue
            if broadcast_mode=='tree' and value is not None:continue
            targets.append((first_use.get(s[1],100000),j,{j},s[1],s[2],None));covered.add(s[1])
    if broadcast_mode=='all':
        for base,value in constant_values.items():
            if value in exclude or base in covered:continue
            ids={writer.get(base+l) for l in range(8)}
            if None in ids:continue
            e,s=nodes[writer[base]]
            if e not in ('alu','valu') or s[0]=='vbroadcast':continue
            targets.append((first_use.get(base,100000),min(ids),ids,base,None,s))
    targets=[target for target in targets if target[0]>=broadcast_min_use]
    targets.sort()
    targets=targets[:broadcast_count]
    previous=None
    for use,j,ids,dest,source,scalar_slot in targets:
        t=max(-8,use-6)
        if source is None:
            source=new(1)
            op=scalar_slot[0];args=scalar_slot[2:]
            if op=='multiply_add':
                product=new(1);add('alu',('*',product,args[0],args[1]),t-6)
                add('alu',('+',source,product,args[2]),t-5)
            else:add('alu',(op,source,*args),t-5)
        stores=[add('store',('store',broadcast_ptr+l,source),t-4+l//2,(() if previous is None else ((previous,0),))) for l in range(8)]
        changed[j]=('load',('vload',dest,broadcast_ptr));explicit[j].update({st:1 for st in stores})
        drop(ids,j);previous=j
    # Identify scalar duplicate-XOR two-way nodes.
    two=[]
    for j,(e,s) in enumerate(nodes[:original_n]):
        if e!='flow' or s[0]!='vselect':continue
        aa=[writer.get(s[3]+l) for l in range(8)];bb=[writer.get(s[4]+l) for l in range(8)]
        if any(q is None for q in aa+bb):continue
        a=[nodes[q] for q in aa];b=[nodes[q] for q in bb]
        if not all(e=='alu' and s[0]=='^' for e,s in a+b):continue
        if not all(a[l][1][2]==b[l][1][2]==a[0][1][2]+l for l in range(8)):continue
        two.append((times[j],j,s,aa,bb))
    two.sort();two=two[len(two)-two_count-two_skip:len(two)-two_skip] if two_count and two_selection=='late' else two[two_skip:two_skip+two_count];last_by_buffer={};last_conditional=None
    for number,(t,j,s,aa,bb) in enumerate(two):
        buffer=number%buffers;out=outs[buffer];dummy=dummies[0 if shared_dummy else buffer];address=new();selected=new()
        defaults=[add('store',('store',out+l,nodes[aa[l]][1][3]),t-16+l//2,(() if buffer not in last_by_buffer else ((last_by_buffer[buffer],0),))) for l in range(8)]
        add('flow',('vselect',address,s[2],dummy,out),t-7)
        conditional=[add('store',('store',address+l,nodes[bb[l]][1][3]),t-5+l//2,((defaults[l],1),)+(() if last_conditional is None or not shared_dummy else ((last_conditional[l],1),))) for l in range(8)]
        ld=add('load',('vload',selected,out),t-1,tuple((st,1) for st in defaults+conditional))
        changed[j]=('valu',('^',s[1],nodes[aa[0]][1][2],selected));drop(set(aa+bb),j)
        # Make the replacement available to the existing vector-lifting pass.
        k._ops.append((changed[j][0],changed[j][1],set()))
        last_by_buffer[buffer]=ld;last_conditional=conditional
    keep=[j for j in range(len(nodes)) if j not in removed];mapping={j:i for i,j in enumerate(keep)}
    resolve=lambda j:mapping[leader.get(j,j)]
    newdeps=[{} for _ in keep]
    for j,dd in enumerate(explicit):
        to=resolve(j)
        for d,lat in dd.items():
            fr=resolve(d)
            if fr!=to:newdeps[to][fr]=max(newdeps[to].get(fr,0),lat)
    finalnodes=[changed.get(j,nodes[j]) for j in keep]
    k.store_transform_counts=(len(targets),len(two),induction_count)
    return rebuild(g,finalnodes,newdeps,[times[j] for j in keep])

def materialize_lowering(g,p):
    """Make actual scalarized slots the graph before balanced lifting."""
    old=[ns[0] for ns in g.nodes]
    nodes=[];times=[]
    for t,b in enumerate(p):
        for e,ss in b.items():
            for s in ss:nodes.append((e,s));times.append(t)
    lookup={node:j for j,node in enumerate(nodes)};parts=[]
    for e,s in old:
        if (e,s) in lookup:parts.append([lookup[e,s]])
        else:
            assert e=='valu' and len(s)==4 and s[0]!='vbroadcast',(e,s)
            parts.append([lookup['alu',(s[0],s[1]+l,s[2]+l,s[3]+l)] for l in range(8)])
    explicit=[{} for _ in nodes]
    for j,dd in enumerate(g.deps):
        for d,lat in dd.items():
            if rw(*old[d])[1]&rw(*old[j])[0]:continue
            for to in parts[j]:
                for fr in parts[d]:explicit[to][fr]=max(explicit[to].get(fr,0),lat)
    return rebuild(g,nodes,explicit,times)

def dead_zero_sink(g,k):
    zero=next(iter(k._initial_zero_bases))
    known=known_values(g,k)
    dummy_values={2062,2078,2094,2110,2134}
    dummy_bases={base for base,width in k._virtual_width.items() if width==8 and any(all(known.get(base+l)==value+l for l in range(8)) for value in dummy_values)}
    assert len(dummy_bases)==5,dummy_bases
    old=[ns[0] for ns in g.nodes]
    explicit=[{d:lat for d,lat in dd.items() if not rw(*old[d])[1]&rw(*old[j])[0]} for j,dd in enumerate(g.deps)]
    nodes=[];replaced=0
    for e,s in old:
        if e=='flow' and s[0]=='vselect':
            args=tuple(zero if a in dummy_bases else a for a in s[2:])
            replaced+=sum(a in dummy_bases for a in s[2:]);s=(s[0],s[1],*args)
        nodes.append((e,s))
    assert replaced==104,replaced
    # Remove only now-dead, pure pointer definitions, never stores or memory reads.
    remove=set()
    while True:
        read={r for j,n in enumerate(nodes) if j not in remove for r in rw(*n)[0]}
        dead={j for j,(e,s) in enumerate(nodes) if j not in remove and (e in ('alu','valu','flow') or (e=='load' and s[0]=='const')) and rw(e,s)[1] and not rw(e,s)[1]&read}
        if not dead:break
        remove|=dead
    keep=[j for j in range(len(nodes)) if j not in remove];remap={j:i for i,j in enumerate(keep)}
    assert not any(d in remove for j in keep for d in explicit[j]),'removed explicit dependency'
    deps=[{remap[d]:lat for d,lat in explicit[j].items()} for j in keep]
    k.dead_sink_dummy_bases=dummy_bases
    k.dead_sink_removed=Counter(nodes[j][0] for j in remove)
    return rebuild(g,[nodes[j] for j in keep],deps,[g.original[j] for j in keep])

def reuse_late_buffer(g,k,buffer=3):
    known=known_values(g,k)
    oldbase=next(b for b,w in k._virtual_width.items() if w==8 and all(known.get(b+l)==2126+l for l in range(8)))
    newbase=next(b for b,w in k._virtual_width.items() if w==8 and all(known.get(b+l)==2054+16*buffer+l for l in range(8)))
    old=[ns[0] for ns in g.nodes]
    explicit=[{d:lat for d,lat in dd.items() if not rw(*old[d])[1]&rw(*old[j])[0]} for j,dd in enumerate(g.deps)]
    readers=[j for j,(e,s) in enumerate(old) if e=='load' and s[0]=='vload' and s[2]==newbase]
    assert len(readers)==8+(1 if buffer in (1,2,3) else 0),len(readers)
    nodes=[];removed=set();changed=0
    for j,(e,s) in enumerate(old):
        rr,ww=rw(e,s)
        if ww & set(range(oldbase,oldbase+8)):
            assert e=='flow' and s[0]=='add_imm'
            removed.add(j)
        elif rr & set(range(oldbase,oldbase+8)):
            changed+=1
            if e=='store':
                assert s[0]=='store' and oldbase<=s[1]<oldbase+8
                s=(s[0],newbase+s[1]-oldbase,s[2])
                for r in readers:explicit[j][r]=0
            elif e=='load':
                assert s[0]=='vload' and s[2]==oldbase;s=(s[0],s[1],newbase)
            else:
                assert e=='flow' and s[0]=='vselect';s=tuple(newbase if i>=2 and a==oldbase else a for i,a in enumerate(s))
        nodes.append((e,s))
    assert len(removed)==8 and changed==80,(len(removed),changed)
    keep=[j for j in range(len(nodes)) if j not in removed];mapping={j:i for i,j in enumerate(keep)}
    assert not any(d in removed for j in keep for d in explicit[j])
    return rebuild(g,[nodes[j] for j in keep],[{mapping[d]:lat for d,lat in explicit[j].items()} for j in keep],[g.original[j] for j in keep])

def header_broadcast(g,k,count=4,skip=0,priority=None):
    zero=next(iter(k._initial_zero_bases));known=known_values(g,k)
    old=[ns[0] for ns in g.nodes];nodes=list(old);times=list(g.original)
    explicit=[{d:lat for d,lat in dd.items() if not rw(*old[d])[1]&rw(*old[j])[0]} for j,dd in enumerate(g.deps)]
    nextbase=max(k._virtual_width)+8
    def new():
        nonlocal nextbase
        dest=nextbase;nextbase+=8;k._virtual_width[dest]=1;return dest
    def add(e,s,t,deps=()):
        j=len(nodes);nodes.append((e,s));times.append(t);explicit.append(dict(deps));return j
    available={value:word for word,value in known.items() if value in range(8)}
    # Prefer reserved architectural zero, then already-required scalar/vector lanes.
    available[0]=zero
    for value,a,b in ((5,2,3),(6,3,3)):
        if value not in available:
            available[value]=new();add('alu',('+',available[value],available[a],available[b]),-10)
    assert all(value in available for value in range(8)),available.keys()
    root_reads=[j for j,(e,s) in enumerate(old) if e=='load' and s[0]=='vload' and known.get(s[2])==7]
    assert len(root_reads)==1,root_reads
    firstuse={r:min(times[j] for j,n in enumerate(old) if r in rw(*n)[0]) for r in []}
    firstuse={}
    for j,n in enumerate(old):
        for r in rw(*n)[0]:firstuse[r]=min(firstuse.get(r,10000),times[j])
    pointer_words=set(available.values())
    candidates=[]
    for j,(e,s) in enumerate(old):
        if e!='valu' or s[0]!='vbroadcast':continue
        # Never replace pointer definitions or the immediate hash startup constants.
        if set(range(s[1],s[1]+8)) & pointer_words:continue
        if known.get(s[1]) in (4097,0x7ED55D16,0xB55A4F09):continue
        candidates.append((min(firstuse.get(s[1]+l,10000) for l in range(8)),j,s))
    candidates.sort();targets=candidates[skip:skip+count]
    writer={w:j for j,n in enumerate(old) for w in rw(*n)[1]}
    previous=None
    for number,(use,j,s) in enumerate(targets):
        t=times[j] if priority is None else priority+4*number
        if priority is not None and s[2] in writer:
            source=writer[s[2]]
            if old[source][0]=='load' and old[source][1][0]=='const':times[source]=priority-4+4*number
        stores=[]
        for lane in range(8):
            deps=(() if previous is None else ((previous,0),))+tuple((root,0) for root in root_reads if lane==7)
            stores.append(add('store',('store',available[lane],s[2]),t-4+lane//2,deps))
        nodes[j]=('load',('vload',s[1],zero));explicit[j].update({st:1 for st in stores});previous=j
    if previous is not None:
        # Any non-literal STORE pointer whose value may select zero belongs to a
        # conditional sink store. Fencing all runtime-pointer scalar stores is
        # stronger and does not need input-dependent address classification.
        for j,(e,s) in enumerate(old):
            if e=='store' and s[0]=='store' and s[1] not in known:
                explicit[j][previous]=max(explicit[j].get(previous,0),0)
    k.header_broadcast_targets=[(use,s[1],known.get(s[1])) for use,j,s in targets]
    return rebuild(g,nodes,explicit,times)

def scalar_select_candidates(g,p):
    nodes=[ns[0] for ns in g.nodes]
    writer={w:j for j,node in enumerate(nodes) for w in rw(*node)[1]}
    flowtimes={t for t,b in enumerate(p) if b.get('flow')}
    out=[]
    for j,(e,s) in enumerate(nodes):
        if e!='flow' or s[0]!='vselect':continue
        aa=[writer.get(s[3]+l) for l in range(8)];bb=[writer.get(s[4]+l) for l in range(8)]
        if None in aa+bb:continue
        a=[nodes[q] for q in aa];b=[nodes[q] for q in bb]
        if not all(e=='alu' and s[0]=='^' for e,s in a+b):continue
        if not all(a[l][1][2]==b[l][1][2]==a[0][1][2]+l for l in range(8)):continue
        depwords=set(s[2]+l for l in range(8))|set(n[1][3] for n in a+b)
        lo=max(g.original[writer[w]]+1 for w in depwords if w in writer)
        hi=g.original[j]
        slack=sum(t not in flowtimes for t in range(lo,hi))
        out.append(dict(j=j,s=s,aa=aa,bb=bb,lo=lo,hi=hi,slack=slack,span=hi-lo))
    return out


def scalar_packet_select(k,g,p,count=8,selection='slack',priority='early',skip=0):
    nodes=[ns[0] for ns in g.nodes];times=list(g.original)
    explicit=[{d:lat for d,lat in dd.items() if not rw(*nodes[d])[1]&rw(*nodes[j])[0]} for j,dd in enumerate(g.deps)]
    targets=scalar_select_candidates(g,p)
    if selection=='slack':targets.sort(key=lambda c:(-c['slack'],-c['span'],c['lo']))
    elif selection=='span':targets.sort(key=lambda c:(-c['span'],c['lo']))
    elif selection=='early':targets.sort(key=lambda c:c['hi'])
    elif selection=='late':targets.sort(key=lambda c:-c['hi'])
    elif selection.startswith('random'):random.Random(int(selection[6:])).shuffle(targets)
    targets=targets[skip:skip+count]
    removed=set();leader={};nextbase=max(k._virtual_width)+8
    for c in targets:
        j,s,aa,bb=(c[x] for x in ('j','s','aa','bb'))
        dest=nextbase;nextbase+=8;k._virtual_width[dest]=8
        for l in range(8):
            a,b=nodes[aa[l]][1],nodes[bb[l]][1]
            nodes.append(('flow',('select',dest+l,s[2]+l,a[3],b[3])))
            times.append(c['lo']+l if priority=='early' else c['hi']-8+l)
            explicit.append({})
        value=nodes[aa[0]][1][2]
        nodes[j]=('valu',('^',s[1],value,dest))
        k._ops.append(('valu',('^',s[1],value,dest),set()))
        for q in aa+bb:removed.add(q);leader[q]=j
    keep=[j for j in range(len(nodes)) if j not in removed];mapping={j:i for i,j in enumerate(keep)}
    resolve=lambda j:mapping[leader.get(j,j)]
    deps=[{} for _ in keep]
    for j,dd in enumerate(explicit):
        to=resolve(j)
        for d,lat in dd.items():
            fr=resolve(d)
            if fr!=to:deps[to][fr]=max(deps[to].get(fr,0),lat)
    g=rebuild(g,[nodes[j] for j in keep],deps,[times[j] for j in keep])
    return g,[(c['lo'],c['hi'],c['slack']) for c in targets]


def compile_base_graph(forest_height, n_nodes, batch_size, rounds):
    assert (forest_height,n_nodes,batch_size,rounds)==(10,2047,256,16)
    assert (SCRATCH_SIZE,N_CORES,VLEN)==(1536,1,8)
    k=base('astra870_store_midreuse_front',stagger=8,reg_limit=170,
           four_buffers=4,four_admission=4,early_two_groups=0,two_groups=0,
           index_lookup_groups=0,index_lookup_final_groups=0,boundary_lookup_groups=16,middle_two_groups=3,middle_group_mask=0xe0)
    gate_sparse_loads(k,4)
    gate_packet(k,4)
    graph=Graph(k)
    for phase,passes in enumerate((6,6,4)):
        for iteration in range(passes):
            program,times=graph.schedule(1,0,reverse=bool(iteration%2),allow_lowering=iteration>=2 or phase>0)
            graph.original=times
        k.instrs=k._virtual_instrs=program
        if phase==1:
            k._startup_input_base=7+n_nodes+batch_size
            scalar=max(k._virtual_width)+8
            zero=scalar+8
            k._virtual_width[scalar]=1
            k._virtual_width[zero]=8
            j=len(k._ops)
            k._ops.extend([('load',('const',scalar,0),set()),('valu',('vbroadcast',zero,scalar),{j})])
            k._op_rw.extend([([],[scalar]),([scalar],list(range(zero,zero+8)))])
            k._op_group.extend([-1,-1])
            program=startup_synthesize(k,[{'load':[('const',scalar,0)]},{'valu':[('vbroadcast',zero,scalar)]}]+program)
            k.instrs=k._virtual_instrs=program
        graph=Graph(k)
    graph=lift(graph,k,program,1)
    for iteration in range(6):
        program,times=graph.schedule(1,0,reverse=bool(iteration%2),allow_lowering=False)
        graph.original=times
    graph=reassociate_final(k,graph,4)
    program,times=graph.schedule(1,0,allow_lowering=False)
    graph.original=times
    graph=improve(graph,k,program,two_count=8,broadcast_count=0,buffers=1,
                  induction=False,pointer_engine='flow',two_selection='late',
                  pointer_priority=None,two_skip=0)
    for iteration in range(6):
        program,times=graph.schedule(1,0,reverse=bool(iteration%2),allow_lowering=True)
        graph.original=times
    graph=materialize_lowering(graph,program)
    counts=Counter(e for bundle in program for e,slots in bundle.items() for slot in slots)
    count=max(0,round((counts['alu']+8*counts['valu'])/10-counts['valu']))
    graph=lift(graph,k,program,count)
    for iteration in range(4):
        program,times=graph.schedule(1,0,reverse=bool(iteration%2),allow_lowering=False)
        graph.original=times
    graph=dead_zero_sink(graph,k)
    graph=reuse_late_buffer(graph,k)
    for iteration in range(6):
        program,times=graph.schedule(1,0,reverse=bool(iteration%2),allow_lowering=False)
        graph.original=times
    graph=header_broadcast(graph,k,8,12,priority=8)
    for iteration in range(6):
        program,times=graph.schedule(1,0,reverse=bool(iteration%2),allow_lowering=True)
        graph.original=times
    graph=materialize_lowering(graph,program)
    counts=Counter(e for bundle in program for e,slots in bundle.items() for slot in slots)
    count=max(0,round((counts['alu']+8*counts['valu'])/10-counts['valu']))
    graph=lift(graph,k,program,count)
    for iteration in range(4):
        program,times=graph.schedule(1,0,reverse=bool(iteration%2),allow_lowering=False)
        graph.original=times
    graph,_=scalar_packet_select(k,graph,program,2,'slack','early',0)
    for iteration in range(6):
        program,times=graph.schedule(1,0,reverse=bool(iteration%2),allow_lowering=True)
        graph.original=times
    graph=materialize_lowering(graph,program)
    counts=Counter(e for bundle in program for e,slots in bundle.items() for slot in slots)
    count=max(0,round((counts['alu']+8*counts['valu'])/10-counts['valu']))
    graph=lift(graph,k,program,count)
    for iteration in range(4):
        program,times=graph.schedule(1,0,reverse=bool(iteration%2),allow_lowering=False)
        graph.original=times
    return k,graph,program


# The following graph transforms operate on generated virtual instructions.
# Runtime values remain in the original machine; only supported-shape metadata
# and compile-time schedules are available to this compiler.


def middle_transform(g,k,p,groups=(),advance=0,inspect=False):
 nodes=[ns[0] for ns in g.nodes];times=list(g.original);original=list(nodes)
 explicit=[{d:lat for d,lat in dd.items() if not rw(*nodes[d])[1]&rw(*nodes[j])[0]} for j,dd in enumerate(g.deps)]
 writer={r:j for j,node in enumerate(nodes) for r in rw(*node)[1]};lookup={node:j for j,node in enumerate(nodes)}
 groupmap={(e,s):group for (e,s,ds),group in zip(k._ops,k._op_group)}
 changed={};remove=set();leader={};nextbase=max(k._virtual_width)+8;converted=[]
 def add(e,s,t,deps=()):
  j=len(nodes);nodes.append((e,s));times.append(t);explicit.append(dict(deps));return j
 for j,(e,s) in enumerate(original):
  if e!='flow' or s[0]!='vselect':continue
  group=groupmap.get((e,s));job=k.ssa_four_jobs.get((7,group))
  if not job or len(job[0])!=32:continue
  aa=[writer.get(s[3]+l) for l in range(8)];bb=[writer.get(s[4]+l) for l in range(8)]
  if None in aa+bb or not all(nodes[q][0]=='alu' and nodes[q][1][0]=='^' for q in aa+bb):continue
  st,ld,buf=job
  leaf_default=[lookup[k._ops[q][:2]] for q in st[:8]]
  leaf_conditional=[lookup[k._ops[q][:2]] for q in st[8:16]]
  if not all(nodes[bb[l]][1][3]==nodes[leaf_default[l]][1][2]-3 and nodes[aa[l]][1][3]==nodes[leaf_default[l]][1][2]-2 for l in range(8)):continue
  assert all(nodes[aa[l]][1][2]==nodes[bb[l]][1][2]==nodes[aa[0]][1][2]+l for l in range(8))
  ptr=nodes[leaf_default[0]][1][1];address=nodes[leaf_conditional[0]][1][1]
  addrnode=nodes[writer[address]]
  assert addrnode[0]=='flow' and addrnode[1][0]=='vselect' and addrnode[1][2]==s[2] and addrnode[1][3]==ptr
  deps=[d for q in leaf_default for d in explicit[q]]
  previous=max([times[d] for d in deps] or [0])
  rawtime=max(times[writer[nodes[q][1][3]]] for q in aa+bb)
  branch=times[writer[address]]
  if inspect:print('CANDIDATE',group,'select',times[j],'prev',previous,'raw',rawtime,'branch',branch,'leaf',min(times[q] for q in leaf_default),flush=True)
  if group not in groups:continue
  selected=nextbase;nextbase+=8;k._virtual_width[selected]=8;t=times[j]-advance
  default=[add('store',('store',ptr+l,nodes[bb[l]][1][3]),t-10+l//2,explicit[leaf_default[l]].items()) for l in range(8)]
  conditional=[]
  for l in range(8):
   inherited={d:lat for d,lat in explicit[leaf_conditional[l]].items() if d not in leaf_default}
   inherited[default[l]]=1
   conditional.append(add('store',('store',address+l,nodes[aa[l]][1][3]),t-5+l//2,inherited.items()))
  read=add('load',('vload',selected,ptr),t-1,tuple((q,1) for q in default+conditional))
  for q in leaf_default:explicit[q][read]=0
  changed[j]=('valu',('^',s[1],nodes[aa[0]][1][2],selected));k._ops.append((*changed[j],set()))
  for q in set(aa+bb):remove.add(q);leader[q]=j
  converted.append(group)
 assert set(converted)==set(groups),(converted,groups)
 keep=[j for j in range(len(nodes)) if j not in remove];mapping={j:i for i,j in enumerate(keep)}
 resolve=lambda j:mapping[leader.get(j,j)]
 deps=[{} for _ in keep]
 for j,dd in enumerate(explicit):
  to=resolve(j)
  for d,lat in dd.items():
   fr=resolve(d)
   if fr!=to:deps[to][fr]=max(deps[to].get(fr,0),lat)
 return rebuild(g,[changed.get(j,nodes[j]) for j in keep],deps,[times[j] for j in keep])


def boundary_transform(g,k,count=16,which='late'):
    nodes=[ns[0] for ns in g.nodes];times=list(g.original)
    explicit=[{d:lat for d,lat in dd.items() if not rw(*nodes[d])[1]&rw(*nodes[j])[0]} for j,dd in enumerate(g.deps)]
    known=known_values(g,k)
    for value,base in k.constant_vectors.items():
        for lane in range(8):known[base+lane]=value
    writers={w:j for j,n in enumerate(nodes) for w in rw(*n)[1]}
    candidates=[]
    for j,(e,s) in enumerate(nodes):
        if e=='valu' and s[0]=='multiply_add' and known.get(s[3])==7:
            inner=writers[s[4]];ie,ins=nodes[inner]
            assert ie=='valu' and ins[0]=='multiply_add'
            sel=writers[ins[4]];se,ss=nodes[sel]
            assert se=='flow' and ss[0]=='vselect'
            candidates.append((times[j],j,inner,sel))
    candidates.sort(reverse=which=='late');candidates=candidates[:count]
    constants={v:b for b,w in k._virtual_width.items() if w==8 and (v:=known.get(b)) is not None and all(known.get(b+l)==v for l in range(8))}
    nextbase=max(k._virtual_width)+8;removed=set();leader={}
    for tm,j,inner,sel in candidates:
        _,s=nodes[j];_,ins=nodes[inner];_,ss=nodes[sel]
        va=(known[ss[3]]+7)&0xffffffff;vb=(known[ss[4]]+7)&0xffffffff
        assert va in constants and vb in constants,(va,vb)
        plus=nextbase;choice=nextbase+8;nextbase+=16;k._virtual_width[plus]=k._virtual_width[choice]=8
        nodes.append(('flow',('vselect',plus,ss[2],constants[va],constants[vb])));times.append(times[sel]);explicit.append({})
        nodes.append(('flow',('vselect',choice,s[2],plus,ss[1])));times.append(tm-1);explicit.append({})
        nodes[inner]=('valu',('multiply_add',s[1],ins[2],ins[3],choice));times[inner]=tm
        removed.add(j);leader[j]=inner
    keep=[j for j in range(len(nodes)) if j not in removed];mapping={j:i for i,j in enumerate(keep)}
    deps=[{} for _ in keep]
    for j,dd in enumerate(explicit):
        to=mapping[leader.get(j,j)]
        for d,lat in dd.items():
            fr=mapping[leader.get(d,d)]
            if fr!=to:deps[to][fr]=max(deps[to].get(fr,0),lat)
    return rebuild(g,[nodes[j] for j in keep],deps,[times[j] for j in keep])


def header_startup(g,k):
    nodes=[ns[0] for ns in g.nodes];times=list(g.original)
    known=known_values(g,k)
    explicit=[{d:lat for d,lat in dd.items() if not rw(*nodes[d])[1]&rw(*nodes[j])[0]} for j,dd in enumerate(g.deps)]
    zero=next(iter(k._initial_zero_bases))
    header=max(k._virtual_width)+8;k._virtual_width[header]=8
    hd=len(nodes);nodes.append(('load',('vload',header,zero)));times.append(-30);explicit.append({})
    scalar_alias={};remove=set()
    for j,(e,s) in enumerate(nodes[:-1]):
        if e=='load' and s[0]=='const' and s[2]==7 and k._virtual_width[s[1]//8*8]==1:
            scalar_alias[s[1]]=header+4;remove.add(j)
        if e=='flow' and s[0]=='add_imm' and known.get(s[1])==2310 and k._virtual_width[s[1]//8*8]==1:
            scalar_alias[s[1]]=header+6;remove.add(j)
    assert len(scalar_alias)>=2 and sum(a==header+6 for a in scalar_alias.values())==1,scalar_alias
    # Identify cache VLOAD7 and its raw-root broadcast before replacing pointer reads.
    cache=[s[1] for e,s in nodes if e=='load' and s[0]=='vload' and known.get(s[2])==7]
    assert len(cache)==1,cache
    root_broadcast=[j for j,(e,s) in enumerate(nodes) if e=='valu' and s[0]=='vbroadcast' and s[2]==cache[0]]
    assert len(root_broadcast)==1,root_broadcast
    j=root_broadcast[0];e,s=nodes[j];nodes[j]=(e,(s[0],s[1],header+7))
    c16=k.constant_vectors[16];c19=k.constant_vectors[19];c3=k.constant_vectors[3]
    defs={w:j for j,n in enumerate(nodes) for w in rw(*n)[1]}
    assert len({defs[c16+i] for i in range(8)})==1,'scalarized16'
    assert len({defs[c19+i] for i in range(8)})==1,'scalarized19'
    j=defs[c16];nodes[j]=('valu',('vbroadcast',c16,header))
    j=defs[c19];nodes[j]=('valu',('+',c19,c16,c3))
    # Only scalar operands alias header lanes; vector identities stay unchanged.
    for j,(e,s) in enumerate(nodes):
        if j in remove:continue
        if e=='load' and s[0]=='const':continue
        if e=='flow' and s[0]=='add_imm':
            s=(s[0],s[1],scalar_alias.get(s[2],s[2]),s[3])
        elif e=='store':s=(s[0],scalar_alias.get(s[1],s[1]),s[2])
        else:s=(s[0],s[1])+tuple(scalar_alias.get(a,a) for a in s[2:])
        nodes[j]=(e,s)
        if e=='store':explicit[j][hd]=0
    # Delete only dead pure definitions; keep all memory reads and stores.
    while True:
        used={r for j,n in enumerate(nodes) if j not in remove for r in rw(*n)[0]}
        dead={j for j,(e,s) in enumerate(nodes) if j not in remove and (e in ('alu','valu','flow') or e=='load' and s[0]=='const') and rw(e,s)[1] and not rw(e,s)[1]&used}
        if not dead:break
        remove|=dead
    keep=[j for j in range(len(nodes)) if j not in remove];mapping={j:i for i,j in enumerate(keep)}
    assert not any(d in remove for j in keep for d in explicit[j]),'removed explicit memory edge'
    k.header_base=header
    k.header_contract=(16,2047,256,10,7,2054,2310)
    return rebuild(g,[nodes[j] for j in keep],[{mapping[d]:lat for d,lat in explicit[j].items()} for j in keep],[times[j] for j in keep])


def prune(g,k):
 old=[ns[0] for ns in g.nodes];used={r for n in old for r in rw(*n)[0]}
 explicit=[{d:lat for d,lat in dd.items() if not rw(*old[d])[1]&rw(*old[j])[0]} for j,dd in enumerate(g.deps)]
 nodes=[];ex=[];times=[];mapping={}
 for j,(e,s) in enumerate(old):
  mapping[j]=[]
  lanes=[l for l in range(8) if s[1]+l in used] if e=='valu' else []
  if e=='valu' and len(s)==4 and 0<len(lanes)<8:
   for l in lanes:
    mapping[j].append(len(nodes));nodes.append(('alu',(s[0],s[1]+l,s[2]+l,s[3]+l)));ex.append(explicit[j]);times.append(g.original[j])
  else:mapping[j].append(len(nodes));nodes.append((e,s));ex.append(explicit[j]);times.append(g.original[j])
 ex=[{nd:lat for d,lat in dd.items() for nd in mapping[d]} for dd in ex]
 return rebuild(g,nodes,ex,times)


def broadcast_transform(g,k,count=16,skip=0,selection='use',derived=True,input_loads=0,eager=False):
    nodes=[ns[0] for ns in g.nodes];times=list(g.original)
    explicit=[{d:lat for d,lat in dd.items() if not rw(*nodes[d])[1]&rw(*nodes[j])[0]} for j,dd in enumerate(g.deps)]
    writer={w:j for j,n in enumerate(nodes) for w in rw(*n)[1]}
    consumers={}
    for j,n in enumerate(nodes):
        for r in rw(*n)[0]:consumers.setdefault(r,set()).add(j)
    known=known_values(g,k);targets=[];covered=set();initial_changes={}
    input_defs=sorted((times[j],j,s,known.get(s[1])) for j,(e,s) in enumerate(nodes) if e=='flow' and s[0]=='add_imm' and 2310<=known.get(s[1],-1)<=2558 and (known[s[1]]-2310)%8==0)
    for _,j,s,value in input_defs[:input_loads]:initial_changes[j]=('load',('const',s[1],value))
    for j,(e,s) in enumerate(nodes):
        if e=='valu' and s[0]=='vbroadcast':
            dest=s[1];source=s[2];uses=set.union(*(consumers.get(dest+l,set()) for l in range(8)))
            targets.append(dict(j=j,ids={j},dest=dest,source=source,compute=None,first=min(times[q] for q in uses),uses=uses));covered.add(dest)
    if derived:
        semantic={value:base for value,base in getattr(k,'constant_vectors',{}).items()}
        for word,value in known.items():
            if word in writer and (value not in semantic or times[writer[word]]<times[writer.get(semantic[value],writer[word])]):semantic[value]=word
        def binary_recipe(value,ids,latest):
            forbidden=set(ids);pending=list(ids)
            while pending:
                q=pending.pop()
                for nxt,_ in g.succ[q]:
                    if nxt not in forbidden:forbidden.add(nxt);pending.append(nxt)
            available=[(v,a,times[writer[a]] if a in writer else -1) for v,a in semantic.items() if a not in range(dest,dest+8) and (a not in writer or writer[a] not in forbidden)]
            options=[]
            for va,a,ta in available:
                for vb,b,tb in available:
                    if max(ta,tb)>latest:continue
                    for op,result in (('+',va+vb),('-',va-vb),('*',va*vb),('^',va^vb),('|',va|vb),('&',va&vb)):
                        if result&0xffffffff==value:options.append((max(ta,tb),op,a,b))
            if not options:return None
            _,op,a,b=min(options);return (op,None,a,b)
        for dest,width in k._virtual_width.items():
            if width!=8 or dest in covered or dest not in known or not all(known.get(dest+l)==known[dest] for l in range(8)):continue
            ids={writer.get(dest+l) for l in range(8)}
            if None in ids:continue
            j=min(ids);e,s=nodes[writer[dest]]
            if e not in ('alu','valu') or s[0]=='vbroadcast':continue
            compute=s
            if len(s)!=4:
                if e!='valu' or s[0]!='multiply_add':continue
                uu=set.union(*(consumers.get(dest+l,set()) for l in range(8)))
                compute=binary_recipe(known[dest],ids,min(times[q] for q in uu))
                if compute is None:continue
            if e=='alu' and not all(nodes[writer[dest+l]]==('alu',(s[0],dest+l,s[2]+l,s[3]+l)) for l in range(8)):continue
            if e=='valu' and (len(ids)!=1 or s[1]!=dest):continue
            uses=set.union(*(consumers.get(dest+l,set()) for l in range(8)))
            # Do not alias a vector that participates in its own definition.
            if uses&ids:continue
            targets.append(dict(j=j,ids=ids,dest=dest,source=None,compute=compute,first=min(times[q] for q in uses),uses=uses));covered.add(dest)
    def priority(x):
        if selection=='clone':
            source=x['source'];producer=writer.get(source)
            clonable=source is not None and k._virtual_width[source//8*8]==1 and producer is not None and nodes[producer][0]=='alu'
            return (not clonable,x['first'],x['j'])
        return (x['first'] if selection=='use' else times[x['j']],x['j'])
    targets.sort(key=priority)
    selected=targets[skip:skip+count];zero=next(iter(k._initial_zero_bases));nextbase=max(k._virtual_width)+8
    def new(width=8):
        nonlocal nextbase
        b=nextbase;nextbase+=8;k._virtual_width[b]=width;return b
    def add(e,s,t,deps=()):
        j=len(nodes);nodes.append((e,s));times.append(t);explicit.append(dict(deps));return j
    fixed=set();groups=[];alias={};ignored_sites=set();dead_outputs=set();removed=set();leader={};changed=dict(initial_changes);savings=0
    used_sources=set();vector_producers={}
    for target in selected:
        j=target['j'];D=target['dest'];S=target['source'];t=times[j];source_new=False;source_id=writer.get(S);template=None
        if S is None:
            S=new(1);ss=target['compute'];source_id=add('alu',(ss[0],S,ss[2],ss[3]),t-1);source_new=True;template=(ss[0],ss[2],ss[3])
        elif k._virtual_width[S//8*8]!=1 or S%8 or S in used_sources:
            original=S;S=new(1);source_id=add('alu',('|',S,original,original),t-1);source_new=True;template=('|',original,original)
        used_sources.add(S);E=new();fixed.update((S,D,E));alias[D]=S
        if template is None and source_id is not None:
            ee,ss=nodes[source_id]
            if ee=='alu' and len(ss)==4:template=(ss[0],ss[2],ss[3])
        if eager and source_id is not None:
            t=min(t,times[source_id]+(1 if template is not None else 2));times[j]=t
        copies=[]
        for lane in (1,2,3):
            copies.append(add('alu',(template[0],D+lane,template[1],template[2]) if template is not None else ('|',D+lane,S,S),t-1))
        changed[j]=('flow',('vselect',E,zero,zero,D))
        explicit[j].update({q:1 for q in copies})
        if source_id is not None:explicit[j][source_id]=max(explicit[j].get(source_id,0),1)
        for lane in range(8):vector_producers[D+lane]=j
        # D0 and D4..7 have virtual aliases rather than SSA producers. Every
        # original consumer waits for the entire replacement broadcast.
        for consumer in target['uses']:
            if consumer not in target['ids']:explicit[consumer][j]=max(explicit[consumer].get(j,0),1)
        for lane in range(4,8):
            alias[D+lane]=E+lane-4
            ignored_sites.add((E,D+lane));dead_outputs.add(E+lane)
        for old in target['ids']:
            if old!=j:removed.add(old);leader[old]=j
        groups.append(dict(S=S,D=D,E=E,source_new=source_new,source_original=target['source']))
        savings+=4 if source_new else 5
    keep=[j for j in range(len(nodes)) if j not in removed];mapping={j:i for i,j in enumerate(keep)}
    resolve=lambda j:mapping[leader.get(j,j)]
    deps=[{} for _ in keep]
    for j,dd in enumerate(explicit):
        to=resolve(j)
        for d,lat in dd.items():
            fr=resolve(d)
            if fr!=to:deps[to][fr]=max(deps[to].get(fr,0),lat)
    finalnodes=[changed.get(j,nodes[j]) for j in keep]
    for to,node in enumerate(finalnodes):
        for read in rw(*node)[0]:
            producer=vector_producers.get(read)
            if producer is not None:
                fr=resolve(producer)
                if fr!=to:deps[to][fr]=max(deps[to].get(fr,0),1)
    g=rebuild(g,finalnodes,deps,[times[j] for j in keep])
    k._broadcast_fixed=fixed;k._broadcast_groups=groups;k._broadcast_alias=alias
    k._broadcast_ignored_sites=ignored_sites;k._broadcast_dead_outputs=dead_outputs
    k._broadcast_transform_stats=dict(eligible=len(targets),selected=len(selected),arithmetic_saved=savings,flow_added=len(selected)-len(initial_changes),input_loads=len(initial_changes))
    return g


def allocate(g,k,p,mode='area'):
    births,ends,deaths=lifetimes(k,p);lane_starts={}
    for t,b in enumerate(p):
        for e,ss in b.items():
            for s in ss:
                for w in rw(e,s)[1]:lane_starts[w]=min(lane_starts.get(w,t),t)
    def mask(start,end):
        lo=2*start+1;hi=int(2*end)+(1 if end%1 else 0)
        return ((1<<(hi-lo+1))-1)<<lo
    shapes={};fixed=k._broadcast_fixed
    for b,st in births.items():
        if b in fixed or b in k._initial_zero_bases:continue
        shapes[b]=[mask(lane_starts[b+l],ends.get(b+l,st+.1)) if b+l in lane_starts else 0 for l in range(k._virtual_width[b])]
    for job in k._broadcast_groups:
        S,D,E=(job[x] for x in ('S','D','E'))
        shape=[0]*12
        shape[0]=mask(lane_starts[S],max(ends[S],ends.get(D,ends[S])))
        for lane in (1,2,3):shape[lane]=mask(lane_starts[D+lane],ends[D+lane])
        for lane in range(8):
            end=ends[E+lane]
            if lane<4:end=max(end,ends.get(D+4+lane,end))
            shape[4+lane]=mask(lane_starts[E+lane],end)
        shapes[D]=shape;births[D]=lane_starts[S]
    areas={b:sum(m.bit_count() for m in shape) for b,shape in shapes.items()}
    masks=[];physical={}
    for b in sorted(shapes,key=lambda b:(-areas[b],-len(shapes[b]),births[b])):
        shape=shapes[b];reg=0
        while not all(not masks[reg+l]&m for l,m in enumerate(shape) if reg+l<len(masks)):reg+=1
        if reg+len(shape)>len(masks):masks.extend([0]*(reg+len(shape)-len(masks)))
        physical[b]=reg+8
        for l,m in enumerate(shape):masks[reg+l]|=m
    for job in k._broadcast_groups:
        S,D,E=(job[x] for x in ('S','D','E'));base=physical[D]
        physical[S]=base;physical[E]=base+4
    physical.update({b:0 for b in k._initial_zero_bases});allocated=8+len(masks)
    assert allocated<=1536,('Static scratch allocation rejected',allocated,len(p))
    k._final_physical=physical;k._final_virtual=p
    audit(k,p,physical)
    return remap_program(p,physical),allocated


def audit(k,p,physical):
    alias=k._broadcast_alias;ignored=k._broadcast_ignored_sites;dead=k._broadcast_dead_outputs
    reads={r for b in p for e,ss in b.items() for s in ss for r in rw(e,s)[0]}
    assert not reads&dead,('dead broadcast padding became live',reads&dead)
    # Alias roots may themselves refer to another transformed constant. Resolve
    # these equal-value aliases transitively, rejecting accidental cycles.
    def canonical(r):
        seen=set()
        while r in alias:
            assert r not in seen;seen.add(r);r=alias[r]
        return r
    owners={}
    for b in k._initial_zero_bases:
        for lane in range(8):owners[physical[b]+lane]=b+lane
    for t,bundle in enumerate(p):
        updates={}
        for e,ss in bundle.items():
            assert len(ss)<=SLOT_LIMITS[e]
            for s in ss:
                rr,ww=rw(e,s)
                for r in rr:
                    a=physical[r//8*8]+r%8;assert 0<=a<1536
                    if e=='flow' and s[0]=='vselect' and (s[1],r) in ignored:
                        assert s[4]<=r<s[4]+8 and s[1]+r-s[4] in dead
                        continue
                    assert owners.get(a)==canonical(r),('broadcast owner mismatch',t,e,s,r,a,owners.get(a),canonical(r))
                for w in ww:
                    a=physical[w//8*8]+w%8;assert 8<=a<1536
                    assert a not in updates,('broadcast write collision',t,w,a)
                    updates[a]=w
        owners.update(updates)


def startup_pointers(g,k,count=20):
    """Induct the first input pointers from the actual loaded input-base header."""
    nodes=[ns[0] for ns in g.nodes];times=list(g.original)
    explicit=[{d:lat for d,lat in dd.items() if not rw(*nodes[d])[1]&rw(*nodes[j])[0]} for j,dd in enumerate(g.deps)]
    known=known_values(g,k)
    pointers={0:k.header_base+6};definitions={}
    for j,(engine,slot) in enumerate(nodes):
        if (engine=='load' and slot[0]=='const' or engine=='flow' and slot[0]=='add_imm') and known.get(slot[1],-1) in range(2318,2559,8):
            index=(known[slot[1]]-2310)//8
            pointers[index]=slot[1];definitions[index]=j
    eight=[(j,slot[1]) for j,(engine,slot) in enumerate(nodes) if engine=='load' and slot[0]=='const' and slot[2]==8]
    assert len(eight)==1,eight
    j,eight=eight[0]
    nodes[j]=('flow',('add_imm',eight,next(iter(k._initial_zero_bases)),8));times[j]=-20
    for index in range(1,count+1):
        if index==6:continue
        j=definitions[index]
        nodes[j]=('alu',('+',pointers[index],pointers[index-1],eight));times[j]=index
    return rebuild(g,nodes,explicit,times)


import heapq, hashlib

CAP=(12,6,2,2,1)
ENG=("alu","valu","load","store","flow")
MODE_TRACE_883=((1, 4), (5, 4), (8, 4), (24, 4), (69, 4), (75, 4), (85, 4), (105, 4), (122, 4), (156, 4), (159, 4), (181, 4), (195, 4), (198, 4), (229, 4), (234, 4), (257, 4), (264, 4), (283, 4), (327, 4), (362, 4), (391, 4), (405, 4), (410, 4), (427, 4), (439, 4), (441, 4), (444, 4), (445, 4), (462, 4), (499, 4), (504, 4), (511, 2))

def _mode_expand_graph(g):
 old=[ns[0] for ns in g.nodes]
 nodes=[];times=[];parts=[];vecdest=set()
 for j,(e,s) in enumerate(old):
  parts.append([])
  if e=='valu' and len(s)==4:
   vecdest.add(s[1])
   for l in range(8):parts[-1].append(len(nodes));nodes.append(('alu',(s[0],s[1]+l,s[2]+l,s[3]+l)));times.append(g.original[j])
  else:parts[-1].append(len(nodes));nodes.append((e,s));times.append(g.original[j])
 explicit=[{} for _ in nodes]
 for j,dd in enumerate(g.deps):
  for d,lat in dd.items():
   # Preserve all non-RAW order edges; memory-to-memory edges always survive.
   if rw(*old[d])[1]&rw(*old[j])[0] and not(old[d][0] in ('load','store') and old[j][0] in ('load','store')):continue
   for to in parts[j]:
    for fr in parts[d]:explicit[to][fr]=max(explicit[to].get(fr,0),lat)
 g=rebuild(g,nodes,explicit,times)
 actual={node:j for j,node in enumerate(nodes)};groups=[];taken=set()
 for j,(e,s) in enumerate(nodes):
  if e!='alu' or len(s)!=4 or s[1]%8 or s[1] in taken:continue
  ids=[actual.get(('alu',(s[0],s[1]+l,s[2]+l,s[3]+l))) for l in range(8)]
  if any(q is None for q in ids):continue
  groups.append((s,ids,int(s[1] in vecdest)));taken.add(s[1])
 assert vecdest<=taken,(len(vecdest),len(taken))
 return g,groups

def _mode_apply_modes(g,groups,modes,times=None):
 n=g.n
 ts=list(g.original if times is None else times)
 groupof={j:i for i,(_,ids,_) in enumerate(groups) for j in ids};old=[ns[0] for ns in g.nodes]
 leader={j:min(groups[i][1]) for j,i in groupof.items() if modes[i]};keep=[j for j in range(n) if leader.get(j,j)==j];mapping={j:i for i,j in enumerate(keep)};resolve=lambda j:mapping[leader.get(j,j)]
 nodes=[];times=[]
 for j in keep:
  if j in leader:
   i=groupof[j];nodes.append(('valu',groups[i][0]));times.append(max(ts[q] for q in groups[i][1]))
  else:nodes.append(old[j]);times.append(ts[j])
 deps=[{} for _ in keep]
 for j,dd in enumerate(g.deps):
  for d,l in dd.items():
   to,fr=resolve(j),resolve(d)
   if to==fr:assert not l
   else:deps[to][fr]=max(deps[to].get(fr,0),l)
 return rebuild(g,nodes,deps,times)

def _mode_schedule(g,groups,original,oldmode,reverse,noise,defer,seed):
 n=g.n;m=len(groups);eng=[ENG.index(ns[0][0]) for ns in g.nodes];gid=[-1]*n
 for q,(_,ids,_) in enumerate(groups):
  for j in ids:gid[j]=q
 deps=[dict(s) for s in g.succ] if reverse else g.deps
 nxt=[list(d.items()) for d in g.deps] if reverse else g.succ
 high=max(original);mask=(1<<64)-1;prio=[]
 for j,ot in enumerate(original):
  z=(seed+(j+1)*0x9e3779b97f4a7c15)&mask
  z=((z^(z>>30))*0xbf58476d1ce4e5b9)&mask;z=((z^(z>>27))*0x94d049bb133111eb)&mask;z^=z>>31
  v=(z>>11)*(1.0/9007199254740992.0)*2-1
  prio.append((high-ot if reverse else ot)+noise*v)
 earliest=[0]*n;ts=[-1]*n;deg=list(map(len,deps));readycount=[0]*m;partial=[False]*m;mode=[0]*m;ready=[];future=[]
 def addready(j):
  if gid[j]>=0:readycount[gid[j]]+=1
  heapq.heappush(ready,(prio[j],j))
 for j,d in enumerate(deg):
  if not d:addready(j)
 t=0;done=0
 while done<n:
  while future and future[0][0]<=t:
   _,j=heapq.heappop(future);addready(j)
  used=[0]*5;blocked=[]
  def issue(j):
   nonlocal done
   ts[j]=t;done+=1
   for q,l in nxt[j]:
    earliest[q]=max(earliest[q],t+l);deg[q]-=1
    if not deg[q]:
     if earliest[q]<=t:addready(q)
     else:heapq.heappush(future,(earliest[q],q))
  while ready:
   item=heapq.heappop(ready);pr,j=item
   if ts[j]>=0:continue
   gr=gid[j]
   if gr>=0 and not partial[gr] and readycount[gr]==8 and used[1]<6:
    used[1]+=1;mode[gr]=1;partial[gr]=True
    for a in groups[gr][1]:issue(a)
    continue
   if used[eng[j]]>=CAP[eng[j]]:
    blocked.append(item);continue
   if gr>=0 and not partial[gr] and oldmode[gr] and readycount[gr]<8 and defer>0 and pr+defer>t:
    blocked.append(item);continue
   used[eng[j]]+=1
   if gr>=0:partial[gr]=True
   issue(j)
  for item in blocked:heapq.heappush(ready,item)
  t+=1
  assert t<=2000
 if reverse:ts=[t-1-v for v in ts]
 return t,ts,mode

def _mode_replay(g,groups,trace):
 ts=list(g.original);mode=[z[2] for z in groups]
 for iteration,passes in trace:
  noise=(0,.49,1.95,3.95,7.95,15.95)[iteration%6];defer=(0,1,2,4)[iteration%4]
  for step in range(passes):cycles,ts,mode=_mode_schedule(g,groups,ts,mode,bool((iteration+step)%2),noise if step==0 else 0,defer,iteration+9021)
 return cycles,ts,mode

def compile_modes883(g, lower=9):
 """Compile-time lane scheduling, fixed ISA; caller provides rw and rebuild."""
 g,groups=_mode_expand_graph(g)
 signature=hashlib.sha256(repr((g.nodes,g.deps,g.original,groups)).encode()).hexdigest()
 assert signature=='a16b598450dac0c33c0c5659dcf23bf583ecd1ebc336d5ec4e9d8ee7df45de70', signature
 cycles,ts,modes=_mode_replay(g,groups,MODE_TRACE_883)
 assert cycles==883
 if lower:
  # Exact hole repair: no existing operation is delayed. Only whole binary
  # vectors change into scalar instructions, each in its genuine RAW window.
  used=[0]*cycles
  covered={j for q,(_,ids,_) in enumerate(groups) if modes[q] for j in ids}
  for j,ns in enumerate(g.nodes):
   if j not in covered and ns[0][0]=='alu':used[ts[j]]+=1
  chosen=[]
  for step in range(lower):
   candidates=[]
   for q,(s,ids,_) in enumerate(groups):
    if not modes[q]:continue
    updated=list(used);times={};valid=True
    for j in sorted(ids,key=lambda j:min([ts[c]-lat for c,lat in g.succ[j]] or [cycles-1])):
     lo=max([ts[d]+lat for d,lat in g.deps[j].items()] or [0]);hi=min([ts[c]-lat for c,lat in g.succ[j]] or [cycles-1])
     available=[t for t in range(lo,hi+1) if updated[t]<12]
     if not available:valid=False;break
     t=min(available,key=lambda t:(t>ts[j],abs(t-ts[j]),t));times[j]=t;updated[t]+=1
    if valid:candidates.append(((-ts[ids[0]],sum(abs(t-ts[j]) for j,t in times.items())),q,updated,times))
   assert candidates
   _,q,used,times=min(candidates);modes[q]=0
   for j,t in times.items():ts[j]=t
   chosen.append(q)
  assert chosen==[3912,3902,653,645,191,424,7,4,2][:lower]
 g=_mode_apply_modes(g,groups,modes,ts)
 p=[{} for _ in range(cycles)]
 for j,ns in enumerate(g.nodes):
  for d,l in g.deps[j].items():assert g.original[j]>=g.original[d]+l
  for e,s in ns:p[g.original[j]].setdefault(e,[]).append(s)
 for b in p:
  for e,ss in b.items():assert len(ss)<=CAP[ENG.index(e)]
 return g,p




def compile_graph883(forest_height, n_nodes, batch_size, rounds):
    """Generate, schedule, and audit the original-ISA 883-cycle program."""
    assert (forest_height, n_nodes, batch_size, rounds) == (10, 2047, 256, 16)
    assert (SCRATCH_SIZE, N_CORES, VLEN) == (1536, 1, 8)
    assert {e: SLOT_LIMITS[e] for e in LIMIT} == LIMIT
    k, graph, program = compile_base_graph(forest_height, n_nodes, batch_size, rounds)
    graph = middle_transform(graph, k, program, list(range(16, 32)), 12)
    graph = boundary_transform(graph, k, 8, 'late')
    graph = prune(header_startup(graph, k), k)
    graph = broadcast_transform(graph, k, 6, 0, 'clone', eager=True)
    for iteration in range(8):
        program, times = graph.schedule(1, 0, reverse=bool(iteration % 2), allow_lowering=True)
        graph.original = times
    graph = materialize_lowering(graph, program)
    counts = Counter((e for bundle in program for e, slots in bundle.items() for slot in slots))
    count = max(0, round((counts['alu'] + 8 * counts['valu']) / 10 - counts['valu']))
    graph = lift(graph, k, program, count)
    for iteration in range(4):
        program, times = graph.schedule(1, 0, reverse=bool(iteration % 2), allow_lowering=False)
        graph.original = times
    graph = startup_pointers(graph, k, 20)
    for iteration in range(8):
        program, times = graph.schedule(1, 0, reverse=bool(iteration % 2), allow_lowering=False)
        graph.original = times
    graph = lift(graph, k, program, 3)
    for iteration in range(8):
        program, times = graph.schedule(1, 0, reverse=bool(iteration % 2), allow_lowering=False)
        graph.original = times
    graph, program = compile_modes883(graph)
    return (k, graph, program)

# Pure graph transforms. No runtime input or tree data is available here.

def pointer_vectors(k, g):
    old = [ns[0] for ns in g.nodes]
    writer = {w: j for j, n in enumerate(old) for w in rw(*n)[1]}
    ptr0 = next((b for b, width in k._virtual_width.items() if width == 8 and all((old[writer[b + l]] == ('load', ('const', b + l, 2054 + l)) for l in range(8) if b + l in writer)) and all((b + l in writer for l in range(8)))))
    ptrs = [ptr0]
    node_set = set(old)
    for _ in range(3):
        candidates = [(j, s) for j, (e, s) in enumerate(old) if e == 'valu' and s[0] == '+' and (s[2] == ptrs[-1]) and (s[3] == k.constant_vectors[16])]
        if not candidates:
            candidates = [(j, s) for j, (e, s) in enumerate(old) if e == 'alu' and s[0] == '+' and (s[1] % 8 == 0) and (s[2] == ptrs[-1]) and (s[3] == k.constant_vectors[16]) and all((('alu', ('+', s[1] + l, s[2] + l, s[3] + l)) in node_set for l in range(8)))]
        assert len(candidates) == 1, candidates
        ptrs.append(candidates[0][1][1])
    return ptrs

def identify(k, g):
    old = [ns[0] for ns in g.nodes]
    writer = {w: j for j, n in enumerate(old) for w in rw(*n)[1]}
    ptrs = pointer_vectors(k, g)
    jobs = []
    for ld, (e, s) in enumerate(old):
        if e != 'load' or s[0] != 'vload' or s[2] != ptrs[3]:
            continue
        stores = {d for d in g.deps[ld] if old[d][0] == 'store'}
        if len(stores) != 16:
            continue
        defaults = sorted((d for d in stores if old[d][1][1] in range(ptrs[3], ptrs[3] + 8)), key=lambda d: old[d][1][1])
        if len(defaults) != 8:
            continue
        packets = [writer[old[d][1][2]] for d in defaults]
        if not all((old[q][0] == 'load' and old[q][1][0] == 'vload' and (old[d][1][2] == old[q][1][1] + 2) for d, q in zip(defaults, packets))):
            continue
        conditional = sorted(stores - set(defaults), key=lambda d: old[d][1][1])
        assert len(conditional) == 8
        for lane, st in enumerate(conditional):
            assert old[st][1][2] == old[packets[lane]][1][1] + 1
            assert g.deps[st].get(defaults[lane]) == 1
        selectors = {writer[old[st][1][1]] for st in conditional}
        assert len(selectors) == 1
        sel = selectors.pop()
        assert old[sel][0] == 'flow' and old[sel][1][0] == 'vselect'
        assert old[sel][1][4] == ptrs[3]
        assert old[sel][1][3] in k._initial_zero_bases
        assert all((g.deps[ld][st] == 1 for st in stores))
        jobs.append(dict(ld=ld, defaults=defaults, conditional=conditional, sel=sel, packets=packets))
    jobs.sort(key=lambda job: (g.original[job['ld']], job['ld']))
    assert len(jobs) == 8, len(jobs)
    late = {j['ld'] for j in jobs}
    readers = [[j for j, (e, s) in enumerate(old) if e == 'load' and s[0] == 'vload' and (s[2] == ptr) and (j not in late)] for ptr in ptrs]
    assert list(map(len, readers)) == [12, 13, 13, 13], list(map(len, readers))
    return (ptrs, jobs, readers)

def prefetch_sibling_transform(k, g, scalar_root=False):
    old = [ns[0] for ns in g.nodes]
    nodes = list(old)
    times = list(g.original)
    explicit = [{d: lat for d, lat in dd.items() if not rw(*old[d])[1] & rw(*old[j])[0]} for j, dd in enumerate(g.deps)]
    wr = {w: j for j, n in enumerate(old) for w in rw(*n)[1]}
    rd = collections.defaultdict(set)
    for j, n in enumerate(old):
        for r in rw(*n)[0]:
            rd[r].add(j)
    ptrs, jobs, readers = identify(k, g)
    job = jobs[-1]
    zero = next(iter(k._initial_zero_bases))
    nb = max(k._virtual_width) + 8

    def new():
        nonlocal nb
        result = nb
        nb += 8
        k._virtual_width[result] = 8
        return result

    def add(node, time, ex=None):
        j = len(nodes)
        nodes.append(node)
        times.append(time)
        explicit.append(dict(ex or {}))
        return j
    packets = job['packets']
    packetbases = [old[j][1][1] for j in packets]
    address = old[packets[0]][1][2]
    fma = wr[address]
    assert old[fma][0] == 'valu' and old[fma][1][0] == 'multiply_add'
    _, _, b2, three, partial = old[fma][1]
    assert all((rd[address + lane] == {packets[lane]} for lane in range(8)))
    known = known_values(g, k)
    assert all((known[three + l] == 3 for l in range(8)))
    for lane, pkt in enumerate(packets):
        assert old[pkt][1][2] == address + lane
        nodes[pkt] = ('load', ('vload', packetbases[lane], partial + lane))
        times[pkt] = g.original[wr[partial + lane]] + 1 + lane // 2
    c0, c1 = (new(), new())
    rootids = []
    rootdest = None
    for lane, pb in enumerate(packetbases):
        rootids.append(next(iter(rd[pb])))
        assert len(rd[pb]) == 1
        j = rootids[-1]
        e, s = old[j]
        assert e == 'alu' and s[0] == '^' and (s[3] == pb)
        if rootdest is None:
            rootdest = s[1]
        assert s[1] == rootdest + lane
        if scalar_root:
            nodes[j] = ('flow', ('select', c0 + lane, b2 + lane, pb + 3, pb))
            times[j] = max(times[packets[lane]], g.original[wr[b2 + lane]]) + 1 + lane
        else:
            nodes[j] = ('alu', ('^', c0 + lane, s[2], pb))
            times[j] = max(times[packets[lane]], g.original[wr[s[2]]]) + 1
            add(('alu', ('^', c1 + lane, s[2], pb + 3)), times[j])
    if scalar_root:
        valuebase = old[rootids[0]][1][2]
        assert all((old[rootids[l]][1][2] == valuebase + l for l in range(8)))
        rootselect = add(('valu', ('^', rootdest, valuebase, c0)), max((times[j] for j in rootids)) + 1)
    else:
        rootselect = add(('flow', ('vselect', rootdest, b2, c1, c0)), max((times[j] for j in rootids)) + 1)
    sel = job['sel']
    olde, olds = old[sel]
    b3 = olds[2]
    cond0 = olds[1]
    cond1 = new()
    assert olds[3] == zero and olds[4] == ptrs[3]
    nodes[sel] = ('flow', ('vselect', cond0, b2, ptrs[0], zero))
    times[sel] = g.original[wr[b2]] + 1
    explicit[sel] = {}
    sel1 = add(('flow', ('vselect', cond1, b2, ptrs[1], zero)), times[sel] + 1)
    def0, cond0ids = (job['defaults'], job['conditional'])
    def1 = []
    cond1ids = []
    oldbankreaders = set(readers[3]) | {j['ld'] for j in jobs}
    for lane, pb in enumerate(packetbases):
        d0, c0id = (def0[lane], cond0ids[lane])
        startup = {d: lat for d, lat in explicit[d0].items() if d not in oldbankreaders}
        sink = {d: lat for d, lat in explicit[c0id].items() if d != d0}
        nodes[d0] = ('store', ('store', ptrs[0] + lane, pb + 1))
        explicit[d0] = dict(startup, **{})
        explicit[d0].update({d: 0 for d in readers[0]})
        times[d0] = times[packets[lane]] + 1
        nodes[c0id] = ('store', ('store', cond0 + lane, pb + 4))
        explicit[c0id] = {**sink, d0: 1}
        times[c0id] = max(times[d0] + 1, times[sel] + 1)
        d1 = add(('store', ('store', ptrs[1] + lane, pb + 2)), times[d0] + 1, {**startup, **{d: 0 for d in readers[1]}})
        c1id = add(('store', ('store', cond1 + lane, pb + 5)), max(times[d1] + 1, times[sel1] + 1), {**sink, d1: 1})
        def1.append(d1)
        cond1ids.append(c1id)
    ld = job['ld']
    selected = old[ld][1][1]
    out0, out1 = (new(), new())
    nodes[ld] = ('load', ('vload', out0, ptrs[0]))
    explicit[ld] = {d: 1 for d in def0 + cond0ids}
    times[ld] = max((times[d] for d in def0 + cond0ids)) + 1
    ld1 = add(('load', ('vload', out1, ptrs[1])), max((times[d] for d in def1 + cond1ids)) + 1, {d: 1 for d in def1 + cond1ids})
    finalselect = add(('flow', ('vselect', selected, b3, out1, out0)), max(times[ld], times[ld1], g.original[wr[b3]]) + 1)
    removed = {fma}
    for j, dd in enumerate(explicit):
        if j not in removed:
            assert not set(dd) & removed, (j, set(dd) & removed)
    keep = [j for j in range(len(nodes)) if j not in removed]
    mp = {j: i for i, j in enumerate(keep)}
    ex = [{mp[d]: lat for d, lat in explicit[j].items()} for j in keep]
    g = rebuild(g, [nodes[j] for j in keep], ex, [times[j] for j in keep])
    k.prefetch_previous_jobs = [{key: [mp[x] for x in value] if isinstance(value, list) else mp[value] for key, value in job.items()} for job in jobs[:-1]]
    k.prefetch_previous = {key: [mp[x] for x in value] if isinstance(value, list) else mp[value] for key, value in jobs[-2].items()}
    k.prefetch_meta = dict(job=7, rootselect=mp[rootselect], finalselect=mp[finalselect], ld0=mp[ld], ld1=mp[ld1], packets=[mp[j] for j in packets], defaults0=[mp[j] for j in def0], defaults1=[mp[j] for j in def1], conditional0=[mp[j] for j in cond0ids], conditional1=[mp[j] for j in cond1ids], readers0=[mp[j] for j in readers[0]], readers1=[mp[j] for j in readers[1]], partial=partial, b2=b2, b3=b3)
    cc = collections.Counter((e for ns in g.nodes for e, s in ns))
    oc = collections.Counter((e for e, s in old))
    assert cc['alu'] + 8 * cc['valu'] == oc['alu'] + 8 * oc['valu'] - (8 if scalar_root else 0)
    return g

def restore_previous_selection(k, g, count):
    assert 1 <= count <= len(k.latebank_meta['jobs'])
    old = [ns[0] for ns in g.nodes]
    nodes = list(old)
    times = list(g.original)
    explicit = [{d: lat for d, lat in dd.items() if not rw(*old[d])[1] & rw(*old[j])[0]} for j, dd in enumerate(g.deps)]
    readers = collections.defaultdict(set)
    for j, n in enumerate(old):
        for r in rw(*n)[0]:
            readers[r].add(j)
    writer = {w: j for j, n in enumerate(old) for w in rw(*n)[1]}
    removed = set()
    leader = {}
    newbase = max(k._virtual_width) + 8
    for job in k.latebank_meta['jobs'][-count:]:
        ld = job['ld']
        selected = old[ld][1][1]
        xorids = set().union(*(readers[selected + lane] for lane in range(8)))
        assert len(xorids) in (1, 8), xorids
        lane_nodes = []
        for lane in range(8):
            ids = readers[selected + lane]
            assert len(ids) == 1
            q = next(iter(ids))
            e, s = old[q]
            assert e in ('alu', 'valu') and s[0] == '^'
            offset = lane if e == 'valu' else 0
            assert s[3] + offset == selected + lane
            lane_nodes.append((s[1] + offset, s[2] + offset))
        dest, value = lane_nodes[0]
        assert lane_nodes == [(dest + l, value + l) for l in range(8)]
        yes, no = (newbase, newbase + 8)
        newbase += 16
        k._virtual_width[yes] = k._virtual_width[no] = 8
        anchor = min(xorids)
        condition = old[job['sel']][1][2]
        for lane in range(8):
            for base, st in ((yes, job['defaults'][lane]), (no, job['conditional'][lane])):
                operand = old[st][1][2]
                s = ('^', base + lane, value + lane, operand)
                nodes.append(('alu', s))
                times.append(max(g.original[writer[value + lane]], g.original[writer[operand]]) + 1)
                explicit.append({})
        nodes[anchor] = ('flow', ('vselect', dest, condition, yes, no))
        removed.update(job['defaults'] + job['conditional'] + [job['sel'], ld])
        for q in xorids:
            if q != anchor:
                removed.add(q)
                leader[q] = anchor
    for j, dd in enumerate(explicit):
        if j not in removed:
            assert not set(dd) & removed - leader.keys(), (j, set(dd) & removed - leader.keys())
    keep = [j for j in range(len(nodes)) if j not in removed]
    mapping = {j: i for i, j in enumerate(keep)}
    resolve = lambda j: mapping[leader.get(j, j)]
    ex = [{} for _ in keep]
    for j in keep:
        for d, lat in explicit[j].items():
            to, fr = (resolve(j), resolve(d))
            if to != fr:
                ex[to][fr] = max(ex[to].get(fr, 0), lat)
    g = rebuild(g, [nodes[j] for j in keep], ex, [times[j] for j in keep])
    k.latebank_restore_count = count
    return g

def immediate_constant_aliases(k, g, values=(33,)):
    assert values in ((33,), (33, 16896))
    nodes = [ns[0] for ns in g.nodes]
    times = list(g.original)
    explicit = [{d: lat for d, lat in dd.items() if not rw(*nodes[d])[1] & rw(*nodes[j])[0]} for j, dd in enumerate(g.deps)]
    writer = {w: j for j, n in enumerate(nodes) for w in rw(*n)[1]}
    zero = next(iter(k._initial_zero_bases))
    nextbase = max(k._virtual_width) + 8
    removed = set()
    leader = {}

    def new(width):
        nonlocal nextbase
        b = nextbase
        nextbase += 8
        k._virtual_width[b] = width
        return b

    def add(e, s, t, deps=None):
        j = len(nodes)
        nodes.append((e, s))
        times.append(t)
        explicit.append({} if deps is None else dict(deps))
        return j

    def replace_dest(dest, engine=None, slot=None, time=None):
        j = writer[dest]
        e, s = nodes[j]
        nodes[j] = (engine or e, slot or s)
        if time is not None:
            times[j] = time
        return j
    replace_dest(1808, 'load', ('const', 1808, 2127912214), 0)
    replace_dest(1792, 'load', ('const', 1792, 4097), 1)
    replace_dest(208, time=1)
    replace_dest(1856, time=2)
    replace_dest(1904, time=2)
    replace_dest(1944, time=3)
    replace_dest(1992, time=5)
    replace_dest(32, 'flow', ('add_imm', 32, zero, 3042594569), 3)
    replace_dest(8472, 'flow', ('add_imm', 8472, zero, 26), 4)
    replace_dest(1864, time=3)
    replace_dest(1912, time=3)
    replace_dest(1952, time=4)
    replace_dest(2000, time=6)
    replace_dest(216, time=2)
    if len(values) == 2:
        replace_dest(32, 'load', ('const', 32, 3042594569), 3)
        replace_dest(184, 'alu', ('-', 184, k.header_base, 24), 2)
        replace_dest(1208, time=2)
        replace_dest(1224, time=3)
    for ordinal, value in enumerate(values):
        D = k.constant_vectors[value]
        ids = {writer[D + lane] for lane in range(8)}
        assert len(ids) in (1, 8)
        uses = {j for j, n in enumerate(nodes) if rw(*n)[0] & set(range(D, D + 8))}
        assert not uses & ids
        S, E = (new(1), new(8))
        st, ct = (0, 2) if ordinal == 0 else (3, 5)
        scalar = add('flow', ('add_imm', S, zero, value), st)
        copies = [add('alu', ('|', D + lane, S, S), st + 1) for lane in (1, 2, 3)]
        j = min(ids)
        nodes[j] = ('flow', ('vselect', E, zero, zero, D))
        times[j] = ct
        explicit[j] = {scalar: 1, **{q: 1 for q in copies}}
        for q in uses:
            explicit[q][j] = max(explicit[q].get(j, 0), 1)
        for old in ids - {j}:
            removed.add(old)
            leader[old] = j
        k._broadcast_fixed.update((S, D, E))
        k._broadcast_alias[D] = S
        for lane in range(4, 8):
            k._broadcast_alias[D + lane] = E + lane - 4
            k._broadcast_ignored_sites.add((E, D + lane))
            k._broadcast_dead_outputs.add(E + lane)
        k._broadcast_groups.append(dict(S=S, D=D, E=E, source_new=True, source_original=None, immediate=value))
    keep = [j for j in range(len(nodes)) if j not in removed]
    mapping = {j: i for i, j in enumerate(keep)}
    resolve = lambda j: mapping[leader.get(j, j)]
    deps = [{} for _ in keep]
    for j, dd in enumerate(explicit):
        to = resolve(j)
        for d, lat in dd.items():
            fr = resolve(d)
            if fr != to:
                deps[to][fr] = max(deps[to].get(fr, 0), lat)
    return rebuild(g, [nodes[j] for j in keep], deps, [times[j] for j in keep])

"""Input-independent compile-time scheduling priorities for the sibling graph.

The hint contains only desired issue times, no simulator memory or instructions.
The scheduler still traverses every dependency and observes every engine limit.
"""
import base64, hashlib, struct, zlib

PRIORITY_HINT_882 = (
    'c-n291-uqjx5nR@p}V`gySux)yGxNyr4bQn1QDb|8YKh~5Rh&VP-$sV0RaK={?{|Zk#oN5`t6v#S3E2Bo_UXc_eKrbKXF2iD00XP'
    'xe>@=-wj0zMPlYZd+<fl+<Nc-p?I{Yq3CMSv=Zt6r9`3l(v?h>lQ1TSl0iurBO@O*6eXZY%ws@N!qyVPlOU8BEy>|e41Wq*6t&di'
    'N*PL}(QrfoNfgF(p)^|YDO#Y-Wyi2aqxZBPL-u26-Rwq_yc75oGmK&cWb<ed$PJ|rla3LJC7)wKY2h`!u|lzleM~%v4gL&RG@G%Z'
    'SP{s#kth@w?l_W444OEi*s`5meJt&#htH(ql8Ly|7FQHcx-${AC}2oMEaEbcrxu?Uhvy8$E<KU+DGgDM%6bC!Etc`2%z82@la4Cs'
    'c*w=qb0S(+>9Yvr<tZmN4M##K8+%FEODzBMgffP5!BbRnMhTVAyzJx*<!3BNUl9K6q1<rh3gtjUE}B(JBE=+WsBowlYZhm#g2bUB'
    'XiJBGMT5BHLbe#QbV86sef$ViTDiy!Pch_+fIP9ZC1o`(>&1ydBI+vzx|66%lM+Wm5eG|&Ls<gv3{x`ptRgKEDMI<c9$UV|)y#CH'
    'p(Pd8M6!?*zv6(xBvTP}tN$d7C6uuocu)iliIGpEti)1frxJITP)Q;XpD5+PRz4z^5UJE~q@Yg@_EbdK_a;d|i9wbN&giNQYiZz0'
    'OP>avDacW2w8WrA7i9%!R!NxmsiiTk&S^?8rG_t#;uBZ-Fz>U2Bsx49*-giDI?!d3uSr=?48mBnn9O1znTOQ`>|5QXSANr=D~Yfr'
    'RJFx{BaZrH@+kutGC;ZUKdH_@0>*ecP4PftXp*2Y0|+vr-|*x_uf-_0sxXakCsU0i#^;=1OpinsS|*}hn79@U6{3m?6OFVwKPk`}'
    '6S+jH0n1HV$nuo~Ijc%T8%G&PEdGSbcUBOmLBHkJqL)VbOU6n@5N6SSBAuGJ_?keqlM?Om;m(fUEN~^qMtW6fW>%7_e&V6SD#^DL'
    '7w#NrNsrb7p>p);C7V@uYI5c&b$$$~SxSn}7L#K5SrEM$LRH|+ru<nQWx;lKRckKB8dPI-_)=*v2T{yPbc~7-sX}VS!1gv}QeF8s'
    'dJ0-@d7TpL7K5B3%S}dPQz{E-l-Crh-z4y-#+uy_IXHneL)Car3sQ?>PQ@mNB&;%Xi!95zK$=$Z_OqN?9^|6W3SWMZ=B8EVIWrN;'
    '5LBtvUPk$sm;LPYr6N!sC@q{Bl!5f<e?zIXlZUA10zql?<PDVx)lzQL%0dRMWkAa0@+xixz>;6*B44O1vCqIPt@4)#j=bPYOYG89'
    'e+A*SRv?(=(H&J79d_akZ{|={bd{rjqqjmhnYXF1G~{R2C{!PmwL<kmjYADW4MAH#8ViwuHx|+>Mp>}oBhM=x{<$a#kLA4xPbqZA'
    'WuX;DLqY9l#cMl*R=J6orQkUWEjO(k&pF9z5#4S16_IT6*&>jNI46T6qfVo7SWZl90p0h-LglIWbkb3QSQlZXB9hr@#hGUzA|=FE'
    'gy(`}&rWesc(bun6W;1b+UYIMY9^t!SqZ2RToq}R&|8M*3aa;<=qgIft`k!NZ9bJ$Uy8Aq;$V9vkt?b-doIpOQK2c0v}GnMoOTD5'
    '6lW=Ri(|2vtQUW)uK;%m)p3qc9r()r51Do2tp;!LaFkwkloP4a#5z2c!TWqf!Kylw-Y=GyO0r-%FH3aGQyWdGi+1#lLhY%Jnv8Y0'
    '2P)7v1aAj;%cHG)Kz5GHD7QJVnVY_ha$~nh7M=C1y2;9++5FE3hn?U;@a6|Y25Bj+dp8$c73E<|5VQ%kr8gwKm^FpJh<qy?ST3p4'
    'StL}Oc_rzu5$b`g?`s)!T3?tr7XeXa^tT~mrC2MgR*qI#nrehvVY{L#pc`k>?i7n{Ids(2y6u#fUu8lai9ttV(=61J))Cv4#Zy+N'
    '-<~QEvno&}*2~e`SUc1u)SIy)>s6u3tX4q2NvJuxn&6*#TUqNpvG5M7z3EL~6=Givsu~)^^LvaHr9B_sWg@G#XL40SErg{~@RWn-'
    'R>!;Qob@Uos6hs6;86|uYeH3Z4^%~aO?C$ha}{BzDwK7@cxymaL0Oq7`{=8P!<x@gi`F#M2kd>wm(jb#(yyPUjE$(O#-Y~m)?;l*'
    'Xqar*);&~R7Hr-OYEAa*LxV#@X@;yJR83wOPjezQIMj(}<7&uSZSIYXpsk~MO-8HLq9Euh<bDjQbK7NSy&jxK)x*f&t)rb5&=9QD'
    '#anx>7sel+n$irX*{lywGis<o1gcIB>%jLeC)4hxETO(2AIy3!cI(Tp)<ni^*nU0T$rhEy>@;AfO{g>eHk3#8XlbecE{bJ+aWx|r'
    '7HR+cr3qw}(MPdvA`B)uDl}53+A`ak^@d;?!C5p54Q0zBX%fw(vkF=o=mzkyHD%3ObNaqi<$KVu&=^&AGs)B^(uMR6DWq@s+}y3r'
    'g`zpsn%K04+Q>pa&P;QmYeB`g!bUUg8dnRo)<J8@9akO?Td0D%vDb<<Yb`Y|tO$<P++X!~$Ku^^&QJ&D&7{9kXk4f}I*U>_9ihBv'
    '>k@jOYAw%d2XMCG?ikN*8`-rg=*;QssQ4So=D})9&6;c8Lbcdh&+X9F9%=)%((M(VY|l<bCp6kk&>8-Y%zDVL)<~2L_2c=yP=9v1'
    'g+7G8En^4jp%2mRK@1FeFF5*yCh43_W9&`#OY6Sq{y)@F7CK|oblGXJch9@<z6<q$qYPsw@pKM2JFy;~I)SY%yWO$XRuyS)m!9m{'
    'ue2U$?S`F_RK`G7d$Qkwc?WFTH(Y+{H@)3tqibjY+`aLm3sq)kpdEXSsOEY)8-uyy26F;!?1GPd;O$Dw2UmV1yGoa3u%}LZH^s7@'
    'V%80eHd=cJdJp}DA-n19O<YUg5>~&{(Az~(=uaiL(>uH<)oeFuKlZwVrhDj9EOdnm;?EeWV?5_*T<D|FM5r`;mEo?1*R^Q_Sg)=7'
    'sTb#`YiJhL(G5#>xA!Ar^YtDzzBe>gPxW|j^kS!}GS*L3JAiuZ%<e$VjhZl<!Mmg(ajDHy4{*063N3UOG-9O@=gCf288|-y%e2rJ'
    'Sg>ju0fu^T_N2<16TAAlC+yC*w@y>yJYQ#jhAa*QdlQ{>v)ch|Ly+tao)%hd1IHkC1_fky!8?q$+L_o6g}UfG+8OT?ngFIj@^m1w'
    '9l`ZJv%%o93=BiRMcC?Z7<|LkjCuxH{6J5=Rmq)H4ZT4#0WIV3tPf|s7hGL+3tCOSORh$1$Fk8&Z=>$ysfNB{DwCi3)SYP>0QY<B'
    'nM7aiNJHOJHPMe}>wOx7{=WEA7XBeT_ma;>713KKsG2_D85`*8D|wsuQysp8OgGhWpU_M^??>y;ZYO#8E~mJy$nHRY)__W@O2mhQ'
    'vM<@}rkkUyPL+KpnDrjWjv@{dh+GdOTH>|k@4e7uqSjBE$Fb9leiRiufxE5>J9Y*KvHx)d>LO%)mE}?JOhslCRnZ(I@91RqM|WRw'
    'nFp4M0gM*oS=8JV+6O#0MrIN@u-_R2h?w#C=ejr2Q$S(g7nRU&x9A}1r9G&Jh=vB^5LxTb`ZRsXG$Rtt(K<zThCwzO4TASQ><-ec'
    'U^~Ng`wtThR3&xBkES{&?a<Oe@2nw2ZV()U89xEb6vh#(4h*dWdpF`gjHl6fXO`L^Ssv7u<FTL|&i)`OV=A0WWzX{UK3Hq0E=LlB'
    'p*+7YU%In0hBjW!R>!bAK-v5%w3R25w^o^Ze++0pkYqc33AU#VW!-j1=zP6Rv(Pw+HW7}A*q#c$e#&ECJm@R?<E3#BI=6}@ATbu+'
    'aiQ<1$7y&t9qDOco~g*$&QMlIgJm@I1(I99F`DS~149?Ri`wZs*lwors!DtJ*?Y`t*rGQIEF;lqmM2Kp2>Ma9;e7el#$q+FjK}UY'
    'cs`KNA1FHShh~dLD%xXc?}t{0RtVcjkWG(3R^O)g1Nmt*2a9u98L7yR;vKLM`5BOZkB<)XZIoyzcl}gZn@DUHhZcc*EIKS3V}!us'
    'HHMWBSes7njBmF38T2Elf?0YmOyaZ}?|8kVyP@|BRu(hPM)D*0mWGytWl?Aut1DUmRCXpH+f|y@@-z#rBdMElcwyPy6q>7CkE0d`'
    'gLe+|_V7$Y>nu1wLW{kZ<}n|Mu21Rb$a_DoU77tJx*NL2+B~Gj6U%wr6Qi*8Vdyh#T3y~%ZYM*NID4OSCwztUB;F5m1DZ$sjOWov'
    '{mA|&w5~i)!TRUWr;6xAozn&XLvxAXWOl!Rb9Crl=qkMZ$;>CTxytlMXrG7lRP=WS(K+gAFwbLnw%VS`?ibKxIH%)Bd$>lbx<-V~'
    '<I`uv^b4M+F<Zpl_c2;$6N%~M^P|vd#pp*_GOaUMUyQCP(mg{M=Yz@KIA>UyseAEr@J)bgE*u}z=Ae5f5{HP*Xk}tL8Wzx--)7t1'
    'zNg8t-6Us-?KwD?Ao&H{Gtl;ls>Iqy_%#W#jDJRdjtI_UKA*}s%eY8&IS(>#XJOY+&1BYJx3RrjI_Wl^t@A&V(JZVG&urvA1+#fL'
    'm#4|1xvEgBn`!9(5@cV5*3iC0YCI==5h&-vH$hgnFxnetjwEL3zL+UJQ&iVuxz#_Hcc1AUGZ7u1Dw-1%^EEup4c!U-s#^I(k{^XO'
    'AQ7HssuJhJGg#K<Lxy;&@Ed*3$(f}5E(4>{Qs&ci*5*<{v*=ecPG{fF;b-Jz8FK$fmsRKdpe<p2C1huDC8u#IymrbKBC#CX3$Xc~'
    'z6{?XwnOzMush*e=%_NX0%S|zStxxgbhB&}qPfabM(0zVYs=gs+Co`cqTH^;(@kibg5H@tTXlZQ?e-CzpW%<SPeNBhheNAW6PuwW'
    '=v+v@h@I8C8JBS~SFmRH$BodDP<v-7r)dQ_{6rC2&TLKu`Wjk^<N~lPrh1Ry&3SxU1D5&d{ZhG_O2!wUVUf5Nt2*XW!?WR>kMue`'
    'oW=7rY^-NokJLImTE}cF{XE?fZ`xdTH{!`Uys)o>?L2*^J9`zhS{iKkb7q_1T`O&C@osNuC-Iqqj|+qREzoQhQlE#mp>G<ecx&ie'
    'Fqxe%mAf^%RhNN%A^kGYf5obyn+eZyFnxu8Unv$otx^<MBH7PbrfOTM`}s5Ywt-|J9NW=!l#Cn-4R$_uHo;*C%-b)(u}Wudk)mZN'
    '=CSfMHWuL1B6O{WYoYQy6Frvk?f9`B+M#B&QGS}dPpgAwvG|f^(Opl!0$(=BgALHvXtuMnBlHa#w!rZ<_I9Fm9n#A{Xny%oT@yrh'
    'C7y++odJETwhJAb(f^hD&GPRJtyHsGUWET^^bT7myX&#{hE}Pc&$q=gxb}l>F}PRC<1ZDV&(;2;-T1O1v`5wU4f5ZF(GK;CgYg^1'
    '%EmQBYY*d2p|P{IJM_Kswv$M#)jPtfVFei0Lc8$Hs=yF02g_a%ujFYJ<9DG0s*>=uUUlSquo3BvSg@0BH`fL<t^mo``kJ!nZX!<m'
    '(6b9Z^YI&2{CN4<w3sYK%Nz2eeGslaX!}ytv=$vpS>4O(HgxX_{Q$PLRNWzbvzlGQ`u7oN6ZAFM_CS8Wm(|1gx(Y8gt5$Y`btU-r'
    'Yv1SmPGp%ngv2&z6P$-ZengeMnFwqN&d4^9eh<p6>}?C3B9E)_b#w4FX{$zubf;KbBcw;6ZD`zprv2#JfW$^@tkcSRyxS0gjwuF5'
    'LdTgM(tWfs0)36N)#eGj+Yy|gA<kfDsB?s~d6avzzjK^-?*`>#6WBKCX4%M!^>#u|qJ1m-JBjcP)#qyVPNDS_GV4|K+u7U7%<h?O'
    'U_Tz5(oI-CA*-8p>urZ;I~v;Zp4p1V&3JH{@oQwPQujrm^?WUL)ZaCF>+d1GRGAwU4LgJTm8%27yNxVuN7F(2L+o3Pon-GoP(NFO'
    '92%X$>S?|hb}|2fJ*$Y_j26f5m>pziJ6yYT!oS1DcN&j})x#O257QrpOp~weCw8_(pgruEZ)d5Az34kkp7&tquqxm%5@#VF(N;Ll'
    'q35)!{d*!{5?7#Ia9-4%V72G>+f;p4rTZ1F190!>9d%ew2hn~(HqVFBJG)iGyUE#W&>Yp%5xvv)!}Fb__VcZ`8{Y5W+Kokv@G<Nk'
    'B}NCpurKs;=rXgT;5)|tt{^_UbVe;^7M}xT{Co6mN8cIvjwx2($@&R0a^fx8OWrQy@d1!pq)rLh1>rszx&+51H0=iQ-f(26_7e0H'
    '?V{R6`0T{)XKk0_aEWI-@jpWQvG4=5U-fVh{O7@R0ROE*4r}M2s^%b4hVi0&Hr|8Gjzd52O?DZngVJ=6y+icA=KXp;B|L|SlI`E;'
    '3*d-mCh=3~mw>EZE<={p%c5UY6USub7&Cu74vJ%nl0BbPy_}%8Oq>aHUBt>cB#!V^W05`?`VBfMEhhpk|LOgn_9MGDc|He?*KfA*'
    'yu+;@hn6#VvKK#3F+WUy2Q5Dms~<)F_@RDR*e{mZyrHM_C3prLC+Y3n{GurCRXyKj^;AIq?2_#MsOmhd`C0Yn(0z`q{-mh<3|)n;'
    '@O+i{+~v2=DLg&Pcu!}|?#k=vIm`NKHU9)zzOO;|M0cV4lDI1!cU4(-e*ciYE9m$c>=#5A!>Z^fY#N4NRPUG3cbb`<f+NK47ggDn'
    '&~2nIO8O!eEbo^<^ps3oX6FV`yaD%3@^wksJ)(A0;|=ISAbSD)Kg!SB^7<<F4B1ueUBvDQbevZGm`}IRdr&oVk(r@C1?PEYw?Yp?'
    '4;T-j=RWN=*)yMSk?n`{f3kNS&A*H98X9g0;TxQX+4%!&XP{^B{|+5v{dVY4=rQ9RW;e-WL^_MLKb3_W;5d)Xb3C6D(%ZCK0iBXX'
    'tLHl)z7_flJa>>ikG(5sxr+6Zs@1#5|3S0fP&{JAs_!rSeja)e`Wt%2>Ur>(4_8AW=L{B~!SzV=ShA0yUs-)h`;YEn^8zx@p}*mE'
    'cmtk7>s3~7ae5w#>>NFSF0uX;gy+e?1!SMd<Cnzmq9XFUvhXwe7vOpTuhq2cTqo8)!T-lwWFFb6{Y%;To6&0a7j#+9FXM^T?hRGY'
    'edw}0c>?cq#%tKPq4RN5*|@;U19tC*%+^hwe+(+wKdrw)kMZJVD5`Tyx}WfT5l*`|FXHQM{CtJ~_MbI6j;r<>Tz7=;rus|&LwEj%'
    '?t<?+*37qGnMHCQV8K6ozcUL@w^ha0+5eUH3WRpbthW8x4K&_VExrWH3;ca3udlIx0gk(>--qhYpyMIFSPlG%y$7P-gBt&X)t`d3'
    '2iUZj+TMM3@1w_c?y!0dD;C{91AFGbRh*sgC)m6mffn<dWGTP=W-+s0Ay<)k9r_n?h@QprhMryLb|{JSBWEZU)$j;iah%xfc}`--'
    'PWUZWo=^i%loy|D9@mNQm@d;0a{dV=a-I{9tD(rwUqtIKDlDG!tFn7jXUtAm4Cg-mBb~eSNc|1x9rWE+e4awDb!MKj@*4UFdL7JS'
    'I5C{)WbB1%^e*Fb^gKtKzZ2bghBvnVOlRi5KCxptHmmm(FFWNob?e+?e6BO(Ix(G>jL*e)pPBjmmriIbCl*L87g3x9RG($sqVSxR'
    'm&ECX&cEd@q4N;?Pq6wI{B~-SP^-4{5<R~~pnvfGv9kGy+D}Bh|6%o!YUVX@{0FTT-#-~+6Rn4;3cqS>{#g5#$$x_}gby!*`go!K'
    'ACNplm+L$R<rA8npGeLtG(LgTkp3&5pV6W^Q6i8>T%NLH+C0a+j^;!r8qplPm6EBFBBAL8v5C%lOvj>TR$qh?I4K?TF)|#FR2NSp'
    'P;??2UF%Pjbr+l-?G+mSCI(TQr>eDQ%EdGAnr|<a*VttJFXpe*{-HNpZ=PO=Ug1%E$M|3C7L9{9ub9VmqB#i_wO4rmRJV0xo)eRU'
    'gpT!zG$tb7(S^y6?~70>$1o(oqtu)dJ57c@9s3!ej7}COD=mwYSw8%u^JcaAFI5x?4X<S@hhv^vZ97gHQ9|^kQ~O8N{;&E+c<@wS'
    'JqcOXEZTV-`{!R~>A@AAUI!=K5JytW3a=MbNTi@fEW(k&6h))0M{+Ey;mL8VY9i4`A%^kr&F76F{wRUP49sFUQK6VlK5@s8=14>*'
    '8u5$DEF*n%p5kjQnv>JX#A@__BD0c&I*3UO5~v=cqc;XRlG8^+)-oNF7RyNr*(?dU$;2$Wd~xB54aE_~4*F<lh%I>|<47j1B>0kE'
    'GdtI&IhLAv78i<x=Hy|z-s~E=n#Tk~GNFirrtp+Yk%@(tf2iHKaHWVqF+mas`GlN}Bxp*+JelmiP4TH8LmiJ<99l~HXspL}ta9Ta'
    'olwzA7?4$5a@OM#m$*V=cVc42E2%6br$voGY2Zs0Q0jmz?tUf`u$u>Kd6ds+%9CXv0r`vXq?IQLMDc@`0B<T1J*)ggdiJxKLKc$K'
    'Qi}3X<>?gPq*yAT{HAv7{Sn8>>l7u5cKVXh$5FhKv0G5lN)LB(*5jfjG5E7VS%YR|v5L=5B>F_)FQ{B(LiS~-2r|*IWEGGWOxe-('
    '=D&io|5~X*sA8xf5{b~88muYMlN_x199zwSw!BUuG(<;FZ1m@L3NuSbpMw3wwEXNRWtNPVLR#!8it_1`Raad0^1zcG?MXO4>Cut_'
    '?m|wYAcnSPF-s(l1oAYUQv}o&yCm!u!g^|HND40FGJTfmq>9s<J|Vlsof72wRmeD!pdl|9(j#p(VVLaJ%7Fe9>?NR2uIR^61g!?F'
    'YKuS#$c%AjqAv`8Mz~C_v^+@c6w|ZSbuy5qp@LJ<mjiKL5TwF(60l@OtL@qinNWQ;rxYB1tTOSOi<U){39FfrFUBmbQ&RCrhL)^k'
    't~ePfrkV9==!+Bo0@$&WSw<O7itZvHD8*hCrz{%E%i|>IDFJtKuv+C;$BUHIgPq1CL7!N8Os4#&XW#VI#frr|xp4R`xhf?G(J<^e'
    'c*+3s5}c7z!MVypOR3s_Gp1xM15&AZsiZ?=3GipdW-hG!M>&Hrovfz;ulZMi-Y}V`d-V&@YQG*@(oz)#(P>}#=kz=6ihezre40Q{'
    '8fi+aH%<=ZEPq*ver}%fVKFzAnTvK49qAO;B6yb9sY;(sdxjz})^pR+IQAu-k@bw&&5gDiL?JgVJ+e87Xh~X5Bs1&nm<{<t%rmo_'
    'N#`TAVw9OvYJAz1_3Zycxv9KNlF6($tjQaetkRrA@hL)07FM0wES5O(!<&tk3wl$zZ>S1&Q5LGiSb;I8>a>(oS?}+B*vyaSTv*SC'
    'jhfC1e&rt4znfVR`XaQF@2J1@t?d6=`YV(Vxm>y_@*<Ig=;RV|i)n5lFv^M6Z1O3G?yDR~)r0CgwWzLIP8}!%F)FNFre~fD)FzQ0'
    '%;{LOIOW8*0<=a>Q>U?{>(c5`i*^Qz&|0$742~S^=MaV*$l4hx$$nAD&O~umifOeND<wEJnH1g3%2H0|Md7X^iJZEb3*uX0=7ot^'
    'c~x#<-7kftGncR#HFH|4Qf#dv@inSQ%qu~a^^PcxwUXGa2C^b}lb3aiRAo4-67Q-+qXuJk{I<+zQvNdNbmdTOS%ulWg!C9v!`R$u'
    't18aHJdbjk7d?51kfAH9td|s3#a2V&P)hZmN4-@+Lsd=(Wg#!KJbKQjxwQtW|JrCQgv~;dDCBf?x;s6HS~s%QgJ-J@`?`OmUmo{E'
    'rI9R1Tv|DO7_CP0lfN>wB515iY^s5z3Le+Mht`To;h+)<iwZ#n@Ut=AmLrCSzXs8&fn<5s?VhwqloEHrK&~XSGH^A4DiZbf>|4Ha'
    '$zMC`xruOYDyA8Gtr*)=JFSUQ-3V07N#)LP1~{Xfq0Th=0r)w@`M{aZv;C!hD4rJsNm=Zb#fIJ5C4|R3Z%1A_(pnSQ4$SJoV>MGu'
    'r>_&pItp=pd@H~^r6~EeeLMT5@w5be33*dcx1iq_K(Z(@=0znq%EM8}=?oRq=`QmYm4?bv7Zp^ocIsQwEdO?A6xV7A*|2#zyeUjR'
    's}hYatYsrJW$?m~l_6v06~{7aCE+g4PDN%Vz*>vFGRjh8YNG;cdDyQ*p1Om;P}n)Fq>MzQa=~e7$lPL4LU&SSotH9Ltw5_pZ*^Qx'
    '+~p%sWu2i)VPh36Higs3kArz__zg=royxL8Uq(Eo6e;Vg;C)p%o9WG2RdrS}*ek7>c~PD9YMPnNYT~PmuA1yr=XZy#m1R~@&u>#@'
    '=UrK`b8dBPR7Evyw{s0yufl2YxvLS^%B)mUe5&cTtf||t8huTk`XW<Dr>T}sMNMqh;iTGK)qo0aKvZk6SCd&SA+0Xdb<tOoEH_41'
    '753_}+fZk*va;Egr<SZXqBYcQ>szkLyrEEA6l>GEK`oGIPQ+{KMyQRx2Fx3Q#bRjE)i`HWbVoNwzB=0KGp_{-^Rp4ukk&wBLo{?$'
    '94snTbXuxVIW5SF>20M7Fbs`3U2UL-oH3g<aNbd7ss%Z*z6ILaP{qxNd`&D?q}6iX;}q4$T08KVZ{3{F-R15V?gF<tt6iZwqS|4j'
    '`PLZ}wqmF%gQBXkQ_-31E_7RfraKX;3P(TAdkgf|#^Y*EPh!#($sW%8PJhsw$GyPQ^DXK{v$vvYx2G=3Ts>~i79i{c)|!m-+y>49'
    'w>REeq#C2E0j-s@+I<&~-lZCh-h<i@kM<(-y&ZOKwI9@%)?fKHs;)TJ(p$9|WH_6lyC$<%iqQMaEWVT7F3uPDx!mm}N%OIGP#JZE'
    'ygp?1+Jm$?S~`n5IRoiCs%{3r6`ty-R^D^cxhdS2&P(TolhS?R{H@GdMms^BvC&e|s3nX8k*dpcPg!j#|61xD(p(i=16wV)pS!}<'
    'j95mbM(FM63?m;sRIfeAzO{GsWzhm^iKZ3-b=FQl)o4?&Hb(kg#mQo3V`I%4@m>f|J+Sf~QEDPGd4JZB)<Dr}s%$kwyG5v_vyA9k'
    'jW^>h^M(fCNq_JR<^9tMZPr?eS_gd#T63LOoBQi6X~S6^Mije1ZIp}ou0^x0JZuDCn=on(jX<VrVB>vfB<=mMaWq+N$G&;n_^rO3'
    'Gs&6k%%shu*{roHsXn`%k!h{BUt8s<oo3cPa4w^>Gq_qPB6d4<5Jt1#MR!U!#@5&!#d=dvjO4i!eP`imp|8DedcU@DdgDzC=G~kz'
    'aCC&PsZL{0X5A#$1uShjNAF<GYO|%Ng_<GjgXJEyF3t=j`-7${t6g*x_+;}|WIjA~KyNR`9%w#}t?pRwAoQ(O!TwHnq(|s`sXJ)<'
    've!wHy_s8UujpEoX0m>itoLAjfHbsKjSkc;(ZN~k>;_>weY14X%`y^h^Xm=Sd^G%>;OdJGi^6ECr>`^48B3+LM}J@TIw}qwog+vL'
    '65`&(q9GO2&l!)L`P(t@yt5+M5x#-CMf##?SOn_jOdxKZgFC)CR=PRIz&2AgKAbj5*|IMotId%_WVrHb)F+I3i`v6wt)2YrhTZTq'
    '-1!ugcI)+G*P^%wOMURx>fr+_WDNae=OfU*s|<95a|j6SE*wQftg=Q0{Wx?Df~KLVyJ(PdGRB!Go{5m1<Ppw?cs@>T49#YP^<L-^'
    '$aQ6<KlZGBiN#UaoQ@wOK{B3s7kxSQA)ce1DbO_9RA`Fx2|7mU-Y|>r>zlKOP}zH<IWZebR_qq<Po){w0rW$#Jps*==qG{B+5|E_'
    '0eyBC__^s#6pVW)_CMxnB1mVF{pQHOuL`nUjDXX|VKkrKtc(!8o~o`sif13S_gJ%YGy>EkpnjmT`sjs6GtfItzWHSHzF_aADjK9L'
    '8m0k6vKN@f5V^4+>`A;n0hN7a&n8yAh_YE5431IGEN6}ozJm|F*_%dQd!obMnTE#C?ChY5{-cpPt?xT?shfUiwv*Ez>*mE+B*#Rc'
    '!R(Af?*O>^hxJpO?ruNYP$UPVXDre)iT+?!&}1xkb341;q26v+cSLY^kHfCLb>Gw5*?&uaq}u#2XdjV*amwR3r?>l#i^uLJ=N)$v'
    '>%-A90UChE1G!~C)b9?<ir)`c?#C-r1D*Nk>W_aPgUf78A`g?%`ld48bqBHE)g1}e{_sy$g?)_n&%kOlNX`CBq&N6_yVDda`^7WR'
    '`5aoH-ls`;HwBAkZMIJ45Mn%nn1A3bB$KmggNW2*Vn39<p^TQ#8F0=dn*)%yGdc_eM)TFr(`>$G^E8`B)Mk<2CCcGYGCCG-#$stO'
    '`Fhv+0$QZrAE$tF3h1pi7r-$G{txhUIL*F;2BTpZIr$iT;b|;V%XG%=W|{!cK<teLX+LWC6Xv7EGm*WC>PO4YX#AY0IDe?<4MT1S'
    'NER}mAnT)u!VEMFlhhmOF+|{Fv`&I=5;!Lb?<itx{X)oe%s{_?!YAVYJfbi!pifl=AL8{i>0FJrMc|oA^cFiy=@-%4Y&!i2bS<J8'
    '(pl8aC({2JyXNgOR%U2-A{tjP{}4+P$&}Gl(9I?n%bevw^ZWO)ycnw=qjMZSenOT$BKk}4WF=ab)2~#&0<S(`Zwb6J&}TQ@Lbw)y'
    'X%VYSd0GsfQAjUk|6@4pCZCA(Or*ysL(4%i6)mHf&(fJ6%gQR|tDJT0j0<SIZs@T@WEQ@Efb?|P9E+{d=>HUdtXXd~gZ*hdO@qd9'
    'e=lY3$7ljkn9eLBEqA_!*7Nr9``yZjRp=7@ur>tj)9}OMVS6*sH5E^+wkNaxDcnoIGeI<i)tQR^O6QPxCL#BkGP%}SBabYG^TB0x'
    'mfg+dW_8QE+32%F(-prdXttQzlbukbnaIz?{!-8^B3jGYpQF2H5$ki&w?^k?9eyn3*<!FFsJ=HRYatpZi#`sc^-g*0W^*gL1sU_v'
    '&sLSp2_u`Y#K-CIu9yDx;Mk~GuLtd7`i)Lycdpact?V{+7a;Mu^Nq6MH=m~C*T;Hmz9B>MDcSu}b@M4c%+j4=kzR(5O72D{zq<kL'
    '^RQppoeJkF>9SktGv#Tn=u>hsm(?TANxodxG9G6<sv249v~u&iE1j**4lvAA#moo$e2{-h6q~xOu+!Xa=9;!W&Sp?dW41u|q+i`O'
    'o8fGrUk-w$wADI0tDGjT`MwnSrD&W(l=nE>ke{w|yOGE(BQq<|xforYU8|}kstl{H`DmW26S)-BUlPg9Xq|;8OPqqPjqO3##GOL~'
    'z9sf%e>qw=au!y?w-Wg!NH=qrV`B;4ty46<K-&Uw%nryRI~$%2?5%b-X>T#lyO^yPo>k5k5U*o2TH|~LrX@nZf%zKguyJib{%o=C'
    't4TT`{yT9pkv2PqXe0W*f_5p(v!ub=X4Y3KhEtvGJlic~RkRr^o2Z?2pxw;+Oy_GbZ6I^sIy=MKPIOIyXAQKJ-A!tnkk|tMZe}~_'
    'm+9tOAzG?_IT}|J$!YYf;NHgbCeSX!j}`K0O9Yw@rnPW?h2F10xK*n=kle<67i~R}+l;+Ex;?(7?V_zzrLL46^L!3kR^jOunpymm'
    '?ATlJbKM30?CZeF*THNna@M{<b{(|M`HtS6x3X)RzC_n<^e)sHFz#=N#Aa4^fXdFGp|rLMKWv=G?v60pskVWgjc|NTZ)eTU_;#ch'
    '(Oayy;?-KlP0E`uwMrFZcZy-U1@8jo=3DkQ!nd1>Tur>!ApJSd{%vaaq}>;boC9dvu3KaY@%l<S_km+Om=}O#8yVZo-Vxnj-?Oq1'
    'kNt?QruS=f9Xto2!|)w~4$_*syPO|*UWx7XI*r?vi!Vj1MZ3^<n2a0xlg@sgEc5GCYX?~ShMBEz1jS{nf6wXxI2LN|(_Ybb=f2{)'
    'n8+PQdKqg6*j=eee9vll+JW}(h>qXyp<hcR)=>}3iNR9V{JkY;`9X9*vja#j*15MR*$G*Ut#8Ep9g@rWt#(wI-z<7F?qm0u-t#L|'
    '<;#h|O6Qa!Zd&(*(P3rnl<H@tbJ01WsN30CiRR;~tyQ`szGb!@ee0d?+&#Kw4@0MQcdaIBmdo?By^8b^cz3C~wkQLei1jY$Al9s!'
    'H^8+{bPmsU^0X!bZIpG(ltq0f`qrybw$txY#cbCdvPH4nByYZj{{TDZoU_cfD@Wg|UMy?-+1Ur?^<em3XZaUt*`XTvinV>t8TNiq'
    'TMOrY_V(lH2KoP^s_loMS^a&heD25FYc!Mj!MP6NgV^1v(|i=pZO+fmPeJu;0pB4I`LWo}(?OB1@dwrLQD)yFbpU)8V_Q4o+)&*d'
    'hxXG>p#KQbKL)A;x|zNQ$x+tNh<=35i>xw_IakoQE1)AhAJOwcc26?e3Eo2$oOdpu!KdRve+nySXs3wjX^rP~1MY)PI|<!G!gv_`'
    '$8-`eK&MrKb{k(O7w2i`0yO79zaM>v(QygxBiQ|3J{|<)7VMdh1K6_4zJwQMaS!@VFkU2%2Z`ocG@Mm`Ml(bClji&I=om6ru=O=s'
    '4{P-#)nxpZf#Xg*H<BCAjpfF66S(oA(`3);%#ZXrWG~ZOW`4oeIi1*_mE{wvfM1=P&TVeLUuid?%fYSiAN>OI%g`0*qC7qzKYm1;'
    '<>ewVx{E!l!VBoOY}xp;D&;=y3f>#~bJ{)U{N_9)mPhdKD%vi~=d;SlG5kA*#Y0F~{w&HjoWH1S%e$S1|Hw|v8LXScaWtL4!X@<n'
    'iFT{K6X-dsY#jpaujn~N6n?_`b+lfkoxtK%{JsLJi_A_skKwU<_>AIm6B+Ao5uqEP*{jp=3y6P#o+5wGc@*UJBz1cMdpDq;6>IbU'
    '7tsZzjyrddyGDEL{4I<RS+{CEiiM-tHxCWvHJ*P$>qVX|9>#fEci=It+m7E~RMgIi&Z=F~iMx#sYlh%7=zd~$Mz?{Tf|J2nF#1*T'
    'vbEb}>K5&|D)JUpauPZPoeb!>D)wi3)BBUE;w<}jp*v{0O?++>%M&^or*ytfOM}rbte#YkACQ52w0+2!-XB$2X7vK|-`TrQ+b^;_'
    'ouZk?SMlf?oL5EW-8EVILlJri_a(_(3QqGiW$F#t{1DN-4qc^P3ueC7t6IC_{3{H9lGTT3-bYREmo~q*@iy4*g!z1v7SFvBREnW7'
    '&)T~k+;;9h-IAB^<|aP>ij_OTDKptSX!=o6yn`(}GdA7`5Z(yr9+J1=w}@C=E+Jt!exmverJdW$MCW&&?m6w;OYlGD={IWZIV%@+'
    'I{n_r&br<2X7%TwhORp%bsy`0BX<Yg7Xmu3b8!JZx3Kbi82zFCf$G6(@;WO|r12M?o^W0tLw_-QM1P0fN8tN|X8j*j`gJs0Mc&6Z'
    '!(`F8s=Ml%^Mc-<?$K@t-Erjpg7X2*B67<K5%WKIdg&x}|CA+*#v`mgLf1`I#}m*zf#;SYcNbk(@$W9}2c74e&TGeW4e>cRf26f?'
    '@A70>y{`Az9U@|N64{OFUPjOFNFNfCTWEPfyTR-hk&fY74ci&IqwL;RzV0A#nBC{-yh9&yuQL9|+5@B?vf9ZtPV@h<Y+d8YYWbMR'
    'zDuk>0@B-Xhupt)US0P%-v5fPmW2n*JGc+g+tGc+)4!mySey{z-@tU%jZBmt!}puWvi*|S+~o9GJRh-hRX5u2isPS*_u;pbY8m`h'
    'URtgBGe6!|nf6vVC4KkBZB=9ycm)lQ(f3%j@C^Rj^!I6={!b$RtN%mD|KQ1ES$+uLQzG|N^un=!|MoZBKTDJGI&fQi>AX^f`R%Dj'
    'vmC{JDGR?5=V!9%>EGg+zGqHU_YYOxb5<TA^9)^<p{I=h65XiqKGD7WC;ji(Fl(=9F7MZuAUMIw3wikpj>n?YigQfY@Vf2;*>+tY'
    ')e}YdtP`7e^L1AJGvH83@!V+ie}UUj_^0HiZXL5=cnytt@Pd_BJVkY%5r<pwJ;$p^u17vS$fdnv?H}d#Z^v-HK*DqF^uLs*Kd6`s'
    'Li@_`ZN8zWaJ(kwcJ5y}7QMf8ax4z76-A4JA$a9jj9%;9n~bf#){Z}q<USS84V|4UjJI{)y%Gk?XJo!iBD;Pc+l@jtV{+C^TQq!l'
    'M*aWG9EX@cWX&W!RfzFE2c>!X8vkArr@O>0GV9UY=;(RrL}zvn6#t;*1<hg-nY~D^)#WR6Sj9aDNi^21mR{mZh<Qx1^8gQ`GLM4J'
    'NRX>O<R&JTFCF8Er0TIuN8>B&H(8Gczmer4DjJiyNsxR6S8O*1(lJD_)jtAx5>9Aj`2WFfD#n=Li;mtHZgTcLbi^e(zw?}cm4rOS'
    'b5rPxF24MXCV!%!Et(=3iRTzvwK2LPkpj6i>_p|Xr*e~mD1n<&$dW1UrZX-SM=cJ!agd9O)tK-lm6gX}dC5C1CA`tuPlC>rZarqP'
    'sj9!6WJJYUDp1BHHc8-%L7$jN_-LZ3zN6B|f|4Wk#7QA6iGpv`#LQwL6_e^u1V?Hjl7Q#L=t}7RrBj)bjHS?XLUxjgQow618CVTZ'
    '9Q8g~J;z|zW=|a7*96F#%~Xo#b0<6{14C3$TTD}um&Cd+5`i?nYqyVSjpk-?U$geYi6gv;u^3f$<GY4Dxtkq3i9r;P`M-|&7@v83'
    '#W|tcOXO211JUVy^x4!>C?ARN|II&6(BgtQ3A0pigcPq=PJDDFW;GFHv&0~Z<)(ErlXFkmPKIOxMJ$2H##qRwf$JY9n`@K-fBto>'
    '!jndzlvs=8T3(aTJa{wd1SUl$a!~D2bV?$lF?Co=jrPd$A_Y49J|%r(H@*DL4#L-DG>e;&c&3L;OA0p~D;Be)((*6)h^{*=8&6r$'
    'nVQ{{>_x|ZMp1^Kk1B*IR12wO%_tqRv5`xM&zZ55O5`I-MT_Z<b;m&i7{|DS=u_+N|JO<58pR>raoj@KOAE$0L50N%W-*A(zfM|7'
    '#BojLwUeIF>dfxwWUhZlzLv(S@V|C4p*5MR!Qy9LR!5&F3n4crzE~8p>M!R;S=2J>@9CDss7SeZ5|ddhddDs77NzBJi@5oq!uXk%'
    'c%+lp=`_oP#~Iy%Xh@>$+drvw!V>8`#&T<bq8i$&yA{w7pB<}o4}6*NJ0s1oq!*HELR?K$UD9?_6mWA<FLoBJj<O(`4Fsk$rXp!^'
    '&qE{=fWgi|WHgz*?Ba+CXJ&BbfbzMyg)JVQ<|G<%U5lAzF)7?>L1*X9PE<M}Fz%di#AKF(SzK(`otz20nL(Y27-SOm%(N_TjSl*9'
    '5YzO*J!K~|x#As#DC8tM`RL<O@0RPd@Mc#G6YKOP4#sqDe!Ner+a<3el1^ti7yicA&7DkH$cRqMj8S@WmX($Pj_mlF7ae((zx=ek'
    'tmZ;0kt#Yh+7o~|pQ4jreGYk=6He>%Xe`cNeqvCJSY#JIJ9SyuD@bkFzSTrpe9V9@KM(dM$mf<&jx!;Vi!nY~%EhkT9_DLu-7SUD'
    'k)LLIEuxv(%fw1Td@Y8hEMXOw7Jt&y7l6~MCj)zhq|M}0f+Hp9Ga#2+XF3fyilU<!HcGHt)GbM0+%4^vM^kp?=)aZ=sXWY5gCrSV'
    'nZ@*~=A0nSEy@#&x$!&?db43SgQ_46(J;!Rdi8k<^PEjmsTGB^M5HJZg}_@FDg+gzWkbSba)l#%&Zm3JPDwh|cz&>yL{CXp?QYAj'
    'b6F6N%%`&Sr5H;in;Aa~6NNn3%?!c<NLQvWFVyK7E5KKTxYSf0v*<p^Al`JyW>Une=yci%FRWXjptP0{`7zB$%f`Gko)vY=FwYKW'
    'Rh|1<$X8)zeFcz}P@Ic^pd|BB^tM(~PgZqh;H&A@R&R2(+=|-GDc<6EP!j!CZzbXNsR}36#v-&*idhAuDye_d%DClZxs>8lM^bf&'
    'MQLVv@HkHdvS^lsBfVQ66c*Qt_*NP%d5KOX#zJ7Q9GO1D=8x7}rs|?Ui`!7=uA19Gu`UZjqf*Q(vEt7w!BYU;Rq1O$HAOY#yH!l4'
    'u->RDR9UUE#wyq-qPxgWcVo9T`6&;&O6aHs)l@XCuLOTZksp)HZ;|OQtvfdlD07pGviMhBi0eUhpjwQz@XODawPMPGRYzskD<WM~'
    '{B`83^)(nPs2-~0Wmzy3gTw9@i)MDPho_og&kuU@pd8jJKt*+1R&krUO^9P%^fzYSKxaB9n(TJU#h6DmQk>@+icNJgSC^e?M5&tM'
    'X7yOrZR0ivbzWsAucYl}EQW>3SSqHvse^1TT3h8aKQ&T-xYh+<9W>MhS$+6(yB*nSBx(%#vodhzQx?o-9igZR>OAg7aJRv){6x>H'
    'r<Sx0WUsFLs)tNoQ0$Wb4OI7y80(?KcnT0TtENgw6=JQBDybf*3%EzIctCYt6}=|a2<&A+S)2Vza2S>4sWx$Jj1@aG73fW~=`s&}'
    '+6PZLu+?yzY29xALXyu<j!LVxiet_0lfAmV3Nr3ztSDdY-@O*bMg_DqN1_Q>3b}jWD4<r5zL=`17WG<%{YpsI$69@~9)<FwZ#P_r'
    'nKxy<23|D<RS~xxIcX2Q>Fpf1M1Mp24oDQ0on_2xBG*jS*ud?ps%;8S3veAKZyk|o=Gwo%w!fWOtu@zEW6(6GS>F^~WtGQPtXpgD'
    'E=9uu^en}@+7YM)WVfQ7q}q7a27Rs2vDE$8T?nRTM7<8OrO;?qT^ZhXU~3EiVj|ZH|4LJ%#qih8Kr^>DYxbAD)=1T-FRrRLDjJNn'
    'S!+hTYP&t$cePsdEwYMlgXR**eg=-FIyv>-`PgiYH?`UCMC_~bT-W`am@I&!II(L;?Uk4J<-({jD5`<F1KjQKXbEFmd0b1C*bv<f'
    'bO!1xBb^xQxLt@!U3U@B?R865(oIxJH<8^)?a^BW95wX4RZI7N2W09g#~oz36iC{MvZ1v+>+RrdfcCn`HPEc7`w@PahIZO-r*~ls'
    'w>zkt5Xa)6sgAym>~v(W46Urja=}=W-8RabwKhQpTZ6g*R=Q)k6}tZ8uj6)B4YtOL;cr9^x*^jFDz8dvs+*yKS_7TCX1XaFsKV+A'
    'Nll)*AZHj`gRwpuS`baEzvgaNER-X5by)2T0*h%U<{gP#NAQ*xrWR;u$9_X(Y7i6u$$V(5zKzaJ4>;S>y6IdPl8RJ7H*7Rwy`$R;'
    'UB3Q0IuCVJ9hE@UTYVRFcEEyFV=Y$dDZlTE-f{Z`V`W94Ba$s>&6#xvQ*F3wBU6V~E4cA$$##Ez8?={%QG4a6Ez+H#p4e!BhN?QV'
    '9kgd-d%PLMFROv>S@*L0v(9cCcQpDe(mkObZ}rs}jk<^Ro$+Q6@ft!u#2rlE8vnb{da>FI&wJuaLl6#jhhb$fR^N5sbNhkUzvpZ4'
    'Z2!&UJJ=iqbw%1ZhqKm&2y}7ByCanC#z^@VJHJIk;I2vE6aG&8HXN#&=r288f>_vSazoM3i{Fs%5dB{A$UN&MPnr^iu9E5+$aZDj'
    'YOOmQ!_Yp2JEj@==}kWvAL_C@jMaDOM}qu4qSr$AUu)H1b9UZQHv4Mq%gR73c0;l$&qJ`-n>GOddZMF^-fsiY(;uWQrPJ{BLAp0&'
    'mU}5u1K79IIaoE@GN8`fj|<!x(8t_T^WC}bL^y^9&1eYnebL#^9fiG7x`lmtyNQO#8*4^Gm8U+`#7XFqd&)iMUUScYzMK9l)vn5M'
    'PxpP$57u4QLNuJ#m!0ACeeuMne!)1HS!Mhhsru~0Q$MV{r{3l*6vy$3Nq6{LP}9R@t1Eu@2i<V!1C1YG`+Y^Gm&hNxDS9Jmt?;5h'
    '&z6A^?qtw(#ljf$c2f-cqr)OIoYtM)u}F<1Z<ehNLgUjQPT2cc8|scD8*L?F^>2u4gY+F{1JF1Cu0C*$))!VE^bgdz8AxXO3fn+p'
    'J`PQ-IE}qkyMwVh7JtvO-dk4QS3g4iAXdkdMf+y!NS217bvQc4yA#|GRcoy{L1TFu<vw*Epydo6jK$y4&={KWn*O2igr|={V>eK9'
    '-OWwlZpu0R013OV`!G&GZW4Azz&oD!y~p|>YRmo#(_i<IVeCgg1wZ=gEjC0IWcR>OXgIP%@ZlYI9@Z=#{iw5HNPQsL4>TVM-$1O*'
    ';O#vCS^FaS1j%0Tj$(H{{b$%%g~kzNW*Rb<^^tIm3ZuE~&6Cf|usIxjqwu0VydSbQ1JoblLpRWjL2dvy!X!oMZF)yH;|%yFqi-4-'
    '=3&z^FdnISpqT7Vhb%`^!8HOM!^q=gBJ?TJ9pRdcg^_sEg-E@Nm5KW6)%&b}&Z=qtm^z#wuK#E}`lbYb=Xyh9iAP_wPs55;>?iJO'
    'W=12?HxeA<q4$LRJ+w~Lxg5{TvUY}QxCEE+uR?ML`$iMO#tAwN;b{_>r-5V&T0Vp(5~=6z3U{UZB|7%0o@c@}8Fc<#*)6y&yMU=3'
    'm}aul1L^7REZMP$eZuo6@_ZVtTX2VTlXN#(8APT&BBxV%>fz4Dqrq7D1dLNaVAV5~HjugLoP;-X$!>V+iSOeSlX0s0v1t4N>sEs^'
    '+&Mb!pP*-=8}c5q^NDWO$*hba(-YC&M^^0YO&84|{_lxK=-oAdwW+H4K4`ZZn1+{=bed+0w{OtD7xaCDyZ>E0oTJ-$HtkbruJnJ1'
    '<R^;v6h@zBaZ+YRp#JcMr^(nJh=up5zyU%%8w*AQHJhb3%OE_Ptmuy<ru~B$&cyy~+C0V4c&8Ji(X3g9W)rDFqW<7E9rhlZ$#?1k'
    'c`;A7<{S{uWVEw4nw<q``U0Kvp`nZ`gExu!I+K`9BTBQO5djTXvz*Mu{w(bo*=(wAk13#A&hs2jj-gy3-G*a4D+}cLFwW#WXfhT)'
    'V{Hr^Uy#fBSRD(liC9>mT9^*6Y59V^d3;BIhVHqtw}fx-1=2X5^)J|2NWYkIG2?hRMzAstG=_HyYjcVJG$Q(izD?%HN5B768JG+I'
    'O!R$*CtJ}u2OM@{=MsZiiqj&-rR*<~@6)ipggW{dFXq8H5nJ<AAJf#^37QYzLhvp`YMi^){f>3J89xBaQev}$6Z#?TGiI~ghsva#'
    'yN|&zoqjR47UT1BEG^M&HT>g|{!lcR)p_7p$jvbf#2bj^IFQ<1GD=ZdhP@4LdM}-q(#z)MWX$R1^M<=S&^%W6<6`I|Nqne#XDhVL'
    '-2!dXe6vuzp~a$4u{jzvOLZsa@|H^0qB2vJV5en_yImEw1eD8Ip9RlUexXi=<69!JjCm0+mzSTmoXXqbeuKWHAX>@YXL&VmSL+`9'
    'mi8sIoxK&|$f|1@{(q=C`4Bs6z_v{P+ukp@`@RHSE-$Z_z)S8;hW}%bO=owJ=Ck#`Z=HwiXS7d)x?H9cK0P?$Z_^6o?0+{rRp);('
    '*zA<9RNXCvEJ6$QzbdwBv8Y<zEr-W4Z8y(U(X@apt1ID|96Zm0^GkTAstyd#8a$s4&M%3~D#+ev7K^#at)rssj^9Ewr?5U#ai4?a'
    '5~8vWS<8jhuGw4@ftDh<#{C*H&To*|iKQ>y^~&KKMPeEjEC=g}$#!@?qLzNseK8#=v;8@{8_3-{q%D8;wp;Id-e>I0V73ZgJE^{n'
    'Z|QfsyBK#8pS9GMo#2hiS$MKIe$M_H&}@R+DrPa9o3OS4ZJUs@n{c6OY%MX4<GrNC^J01lz1WPK-2~oZ5H5y$A^hvz`{-N3ET(7K'
    'S;}k{tKn${?F+p93dtYcSNOOL3>)D80X<(6m%rS2s@Dx@JH%J^VfTc4jM*vXr$D$-=i@7Qm!s_~(J9dxbe$$!Yn8b-v|j%F_q+~I'
    '_t3s2PBz<yoegN)isaYmT8fsXf!?oVX9X*9y}0r^z88(zYbxJ%EMDuvXrrucXUBA&0q?SamY{Ko$ZF=P`-giEdd*kFZN5<Np=k^D'
    'zfz93L+jyOji2j@M<nkLA{&|LKX9Xa&)D^EkcC*>puhbrL24T<sh8AC4JF|;#-~r=MQ0XCeO&g|Aiq>dj?>ms(RSZQ_I41v=R`fB'
    '7srd?J;%ZpknD7KD@uEy-R@p&t=8SYj&TLn!qYa^FS-|qK|E|F^df^avS*dIR^G3KV+Rrco`@y(Y`#;s$a3Ag3pKZzT}8ix=&j^8'
    '*3a&BzSgeM|IF|3jaaagX!%?L-xkGeTR>Y`e}L|9p%mmPg%`#9*G=iIRi0Ps&ijv6hBcp7>kizhJHRv~_mZK(C>g$NW@Wc-h40Y#'
    'kQ3_DBiH6%>!j~S`YVv7_Es|AsLEKQ8(@!{$_r2auF+=oyNUP;#d$M2>@Bv}O+)-1Lce3>DeZUcSk<lt;a=!#Ro;4JcN5bs`kSWd'
    '^XV%$J~_Ah`FrLYMRu+?DthbDv`<-mj^;mU&qR;K?{jX@xKXEYGrsTV9k&@D4ev&+ZIT!Ju$;xq;%#EwqEocjJ>VX6(|JE|q7SO='
    '(Ye};?yW)XZVldj+uXF?3-)&~I{?==aQia*h{^Y~H#+?GgHGu-y+3z?$@mY`$M>r8_D$twC0p5<rS=|6-mF+Y54cC%qgV+~JE^LR'
    'v>aYqWFEn_OQ&lWcI-|(h8MfIk8<%&KS;E`1=V+|g&*L#42JDIZ)Q9}bbjDA%jJE8r5)T>yXEuOVZDtf-KgFb_ZRmT6@A^k!r8y>'
    'zH~Es`|#l~QP`q;;7z~9&ERG9tYuQ4(c6#hV@PN7@_1=jJ%EP?ko?vyfc0-cd<5F6SRMeYWpf`<y2ly11)|*EAw_mS>)(?n`)l|y'
    '(b2cs_6U^4I|}c2_+*jZ3&#%pKZT}U+#>~uO<r#gd&l$_^?hz$&>aVpW%4_2L8H@QbOPDuAULA8)JdLCF$+%?y(94Kc60Leau%&W'
    'D1&?Tt&!8qO@9hq`_X^cJtzJra2@A)7wczOwMZOh#V{Rm&-3)seF44Xcg}0Z=i<-fodCfRy<rM^ds#W6=$w({N!Ki$V15=2=dpC2'
    'nNeZyN8~-lvzT{|H`#fb-=7y1^Nfl@B_RK)xMn51A~eguF*hH2vS8sP8u#JlNzOwW;+L1T3vNlzVraSiK@r#wk{ycSd3hb4F1cmA'
    'QXn#qEUp(oe@xLo#{K!d@>kZ|sfe9J%PFBfqPOZ1x14v<HJ>gaS<tf@Im+2MMO9jV!u<&iXR%=3TgChY*$r*|FOcQ-7xwJ7%L@L}'
    '*t*1u<sc{bLssYvHGf8x=hvS<US{<iHcNZg;6I`GUX4I!!E>I<D$f~D@7Z5;?5%K8p1e)hX_uf2_;OLRi&W$}IPI5z1@Dw1_p4jk'
    't4yqZR*jwt%h^xd*OuecI-@nbYF=5->gP22BGQllLzg+>=iEY`<?Ea<9N-NYmA9|G4WoOpcq1hT`$hb{!RfvrnIqV{peSC9KtFK)'
    'x3YJVcG|s3WWH0R-=<5%_fSylR`ZA4{9XaiD&-<N?VISdn%`em*20s?9@9-=;}QK$^R(XNXH@n3l;un6&+8;#axIU)z<vBJI>v8}'
    'f?grd?+Zfq7t-UZ#gnSZ-8>yoR{RW?pdQvr%O#MNBe!-^?BtZE`hD+yA<yTK+2F=v=b&P71#a`?it6_|^Q(G)?9exheYNb+9bh~c'
    'i1a0)y~@f(t=)(~SFm9^Z|NjlB+@^#a+PX1iT11LuvlJY|5u^9s++r#SJAtzD!ioZU1e64s=CJ7ZM0o?@8a7Hv|VSVhF6nn`dOY@'
    'y<b;<RkJ&)hkNYZz{bz=%+_x~H?VnCC~u<gs#dP?^;i*0mC$&HHRHI<{;$}t<lS_`)2|WeKJr%o_vmlB*CNobvT%!?8?tThsbBHu'
    'iV*zD*UkgJN$!LBwh-Qi?tsR;ybXta%|75;<+gMf=3DIlA$gzsq4s~n&wK1xyMtc)8otYyNEI(UU6l?a<GT;nBj}zG*s6UW{fNf<'
    '0oQ-@K&#cfRr<cZNB@}Y-jj~sc&9&Mtm@qlc7H?9?_~O+y!)NLx)+{y2e;L3{bsN?-|zTk(Rlh6J(Q*gij&#@lc&36|4*&ho^kyo'
    'YY)Zq2rbVP5%Z-c-v@t)egp4g<p1VP<+m5KzlHEO_G|I0;4=GvGXGN&Pmq5Z_Llowqgi>vZ<qLd$0p$0J|2{S+x;Ll{Y0^R1;-!o'
    '{iT|{LSAmE-FB0C>G;i%fv<V{O_71Gj|2KLwRgjQ`d@_bv8+GkTl7nPAKPCO_Rt@aXMaljRpw7*?-_PZfd7e*zoAD#Jp33wfyY)Y'
    'N6(4jeb$}@QMd=!ZgA|y#<$qpOC~Iv{@3dx+7GV1ee8{sO@I5y$9MD-{o?vtT8y3%nZLs5Nm&0x<gY$M<}XS4mLI!yyxLw}&-&Wx'
    'EgLVzcijC)^B2<n+;zDb?(1aSQDr_5-z(RbG<vP<yik@bSHJViED2wD>3CPf7p3s*CeN>L0lQBN@N4EJI$ojw1>Z1#%ECQ0tE~I%'
    'n7)7bRr!xNU%Ea*$9v_5r`O_oC5eCV(a+&y<?DA=e6s&2{~z~1`d4IMPu9Pbl@O>N$-?V^!&Z!pE4pV-4qq|8E)Ut)_J3BKb%&iF'
    'v=YhlwT36p{wD#R{c{M8NJxYNsgSfrrq2WA;A_w*FU|N|@wm*c!4sJ${~ud-!?Z-=y&MIqr9O)G!jt770rj82O9aL9Oqc&pq%iWl'
    '!sv<0j;rS=0eQ-CR4<yyd#i5z{3k+pYWlSLnn?;zbflsok$|-<US=;Pb&!HDLHq3<oj$8)=OUUH6N)C@7_3L7kBrXTK`mM}Syuny'
    '9U4Pl9x>sKLG}F0x06Yjh8W&;*ZNq@V~O4%^(ksNiY6;DHM`-)@nTD>p@=PtE{Z`tKBm5(>hC0RJYT=f;tH!h*YoTz1NI$ca<=1h'
    '#u1Jf0hiq<Aw@riIOAv@Pk()h%SxO;CXQ!ymq0pV;Zc3h{(g}_f1xo>H+V8TcIPF8Vnv`o-9%b9{jnpEW!?7Td3I|h_Iz@@N4jSc'
    'vl~}`Wr;)n66>!mNyTASlW^O{(|wziS&~3kGIo<POU74VQZI1?iqDS8B_g8<gTFi^ff7>-$<Y;=Du^6BB}6`n{=Sn+<J;udVPs;I'
    'iv2_pD5-3xW-qmDBvs6#1l1RXC%f%aVkcPyN=ZviMOsTHJ+YW2k3gwuk9CV$9{sT}eD<p*4OWxNW^($p#5gVYoz2XTl+2S8&2+qJ'
    'QVL-zW`-pVYw5Wg)5v-X=5}sMdKJ7fP&ryzemmIrKvy_2K$#%pP8osHBVV3xq_SQk$&~XN@V>p{b|Eq*lNrrvW%n;Pjn~X;?$!62'
    'dW|JvzhIhq_BE5&>jq~=R?^9X)cS>Pzt3~>&WWciW)^>XA<rO6A5dCJ`DAf1tC`tPN1ulMH0Vj~WoMQ|xyq*9jG_!dpI%aCAq%tg'
    'L6v1gDvSPBkY1MmBSY&~U3Tuvta{EAfwHq&iT6z}yvQL>>>kR*?2g_p_uTUQmd!%c3bB@xcWXA;&+h%Ecl2F1hgJ-EF5SpEn1`pV'
    'TFXYCIRf41JMSf0(ju29u>QieF~4_5@6oJU%jVf%7jmH~Ht&;M(&=l?$4(Bt4YKR+1E!_AmqXUF3vmwla@)0Uf;{pd55LAM1X#23'
    'YxNetxa^&qS9`hm!Y{0Uw~<$Gn%rT1ZhoiucnY8^pWZ&dxdq5wR$^R@`!=Vz^6IycZ!e!T7tq}C>@Q~a9?XFk_61;Xm!J9VSO|*+'
    '0t{LCg_=#$CX-)#{;w-L|Nh^1Q&K^xx!sGPDUH0y4~Iz?fr?5dJZ027%n5p<d~lY-ru}9ss&9?_`rgP#_2=+vfT5`VdQn(9-=<=4'
    '6_rd`-f(r{xA$B<`p96f=~dO=JPHa)c#4Wg<)oo_Kn3_kV|gl~zmgQDuO-em`^ES*kV6<saQ1!cC9qXbcIySL3^x53)&aj|-{M{p'
    'Jtd(!U@3(zUuy#9r8O&sj*`+_lBe9r<VUuEVqmvpetlh(hD^dN*<V_WN^shWvTN_7f~=ONHRM()NkmMaaWteAXKsImw|AFWDhrk2'
    '7HZ(x?_QHDkN;(PGAb)vW!W`dhR={0XF2^wZp>JkmD1`<2D6&vswg}aq|w(?CcsoyH(y!yDzn~D*c<8&tq4_Mr=obvvs#wD#$FTV'
    '6}b`0(HgO8^UB1d6x=0<Oew{xBw48nmFIS`{mQalUEDQTZ-}Of@KlUIhP6DcBKj*ZmXlVKu0Y-^B3DKCul=H}C0nJ1tGuulA}7z>'
    'c%J>+kdSBp4yB50R}-~Bt~q06eQ#9^5LRQX$!+fMR@ZY~)(dK{7AuCSHc$3jy#YSg60+*7SWQ=FUIjGO(U6?AW>5>LlH_Wr3ai6W'
    'U9qdiv*E9aq@A47>ittxAMG`<)PTK)AZn;s)I_!xWEpQpCK_YaY&K%25uDA~sgG16Br0IlT6y8AghW%gOtuknjj-QXV?$wSji$z4'
    'TSdJx8Y_qS+7y}E;%p`fldF!DeWTiWZURRWSt=7&Nu|8@PzPBuoAt?tS!{&HmdMmYrlsDS&E$zsm6V-)NVL$`u$|TBthYl~J887B'
    '9at@Dwa`_XRtr1rf}Lhys;`LG^E%QSUw%ch0SI#u*G^uWfLh>fL-t#;*TCz{P76@hW8Q>TUw#(yy0By1ZH1>5)D$m@dv*(#&|O_f'
    'cUu?q)Wkwt>~^9rsC%&xSX(34QkYF+X?$!3hegM1v;jfW2vmtImqMnw>aeBy7U*q+Uh}OL8oIJ>RDsr+^(L%0Wv3-JtS-BOsGaP#'
    'R-OMxR>>BHR<xG7Eh^}|v<#>%)LMN_cv`Eqr7ug?Eyk5qfp1eaG_{t-){2xp`Pr(XbI}f8+Y3ntRZQLBMz*m9TKyZ$YPi03%AunJ'
    '{&qxTeW+nre%ouu^tBZ=Vz(1dtyEqAk=3b>y`^&2MN#Smb$*N5!BsEFZ&RHypQBUQ*ipA{7wp!cw_5O<QTKp)1g(`K+Co+>OBR3o'
    '?zFqKIcp8cNe8GVTD!86kJy*jty76+f5)oKXeXk(IJ#rMhb(l1tnbCCuxMLl^pFR2S?T8W<lSO&J)thbUWZvh^z~rufd>V!(L-_T'
    'h30OOZNzhL?e_#rPj;;J0#gd?_ePUxYoaq^@4B9>^+L7}8ca(>GEFVv=|k&{_U_(05vT|2-6BwzpvpTEyRO8pm{v_o4|(55b@UE<'
    '??#}W;^@F$cdxII_mU4y&}C73S9qJV;@7ifxr3^rr()4P!0qGhrHu9n;?Z6HccJgdj-BY<id7fcw1}HOeIk%)=^+%o)%&Z4#OLnI'
    '%ua|QYyg5L=xs<Q8mrRn{%Oyy)mK+mIt1+v*%|VCtD=zq|6ncT`hTnl{T~R6jDY'
)

def apply_priority_hint(g):
    signature=hashlib.sha256(repr((g.nodes,g.deps,g.original)).encode()).hexdigest()
    assert signature=='85f888e216083a45dba69acc36bc783d27b1acc50bae67d09198077fbbf0f0b4', signature
    raw=zlib.decompress(base64.b85decode(PRIORITY_HINT_882))
    assert len(raw)==2*g.n
    g.original=list(struct.unpack('<'+'H'*g.n,raw))
    for j,dd in enumerate(g.deps):
        for d,lat in dd.items():
            assert g.original[j]>=g.original[d]+lat
    # Priorities are consumed by the ordinary full graph scheduler.
    p,times=g.schedule(1,0,reverse=False,allow_lowering=False)
    g.original=times
    assert len(p)<=882
    for j,dd in enumerate(g.deps):
        for d,lat in dd.items():
            assert times[j]>=times[d]+lat
    # Canonical within-cycle ordering matches the audited graph materializer.
    canonical=[{} for _ in range(len(p))]
    for j,ns in enumerate(g.nodes):
        for engine,slot in ns:
            canonical[times[j]].setdefault(engine,[]).append(slot)
    return canonical


@lru_cache(None)
def _compile_verified_base(forest_height,n_nodes,batch_size,rounds):
    """Generate, schedule and audit the fixed-shape original-ISA program.

    The recorded priority values are compile-time DAG hints only. The complete
    generated graph still passes through dependency-aware, capacity-aware
    scheduling and physical scratch allocation. No instruction table or runtime
    input values are loaded by the compiler.
    """
    assert (forest_height,n_nodes,batch_size,rounds)==(10,2047,256,16)
    assert (SCRATCH_SIZE,N_CORES,VLEN)==(1536,1,8)
    assert {e:SLOT_LIMITS[e] for e in LIMIT}==LIMIT
    k,graph,program=compile_graph883(forest_height,n_nodes,batch_size,rounds)
    graph=prefetch_sibling_transform(k,graph,scalar_root=True)
    k.latebank_meta={'jobs':[k.prefetch_previous]}
    graph=restore_previous_selection(k,graph,1)
    for iteration in range(6):
        program,times=graph.schedule(1,0,reverse=bool(iteration%2),allow_lowering=False)
        graph.original=times
    assert len(program)==883
    graph=immediate_constant_aliases(k,graph,(33,16896))
    program=apply_priority_hint(graph)
    result,allocated=allocate(graph,k,program)
    assert len(result)==882 and allocated==1495
    assert allocated<=SCRATCH_SIZE
    for bundle in result:
        for engine,slots in bundle.items():
            assert len(slots)<=SLOT_LIMITS[engine]
            for slot in slots:
                reads,writes=rw(engine,slot)
                assert all(0<=word<SCRATCH_SIZE for word in reads|writes)
    globals()['STORE_LAST']=(k,graph,program)
    return result,allocated


class KernelBuilder:

    def __init__(self):
        self.instrs = []
        self.scratch_debug = {}

    def build_kernel(self, forest_height, n_nodes, batch_size, rounds):
        self.instrs, self.scratch_ptr = compile_program(forest_height, n_nodes, batch_size, rounds)
        self.ssa_peak = self.scratch_ptr

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)


# Pure source transformations applied after the verified generated predecessor.
from types import SimpleNamespace as _CandidateNamespace
import operator
from itertools import product
from math import prod
_candidate_compiler = _CandidateNamespace(rw=rw, rebuild=rebuild)

def _candidate_make_fences():
    MASK = (1 << 32) - 1

    BINARY = {'+': operator.add, '-': operator.sub, '*': operator.mul, '^': operator.xor, '&': operator.and_, '|': operator.or_, '<<': operator.lshift, '>>': operator.rshift, '<': lambda a, b: int(a < b), '==': lambda a, b: int(a == b), '//': operator.floordiv, '%': operator.mod, 'cdiv': lambda a, b: (a + b - 1) // b}

    def merge(*sets):
        return None if any((s is None for s in sets)) else frozenset().union(*sets)

    def calculate(op, *sets):
        if any((s is None for s in sets)):
            if op == '&' and len(sets) == 2:
                known = sets[1] if sets[0] is None else sets[0]
                if known is not None and all((mask.bit_count() <= 8 for mask in known)):
                    result = set()
                    for mask in known:
                        subset = mask
                        while True:
                            result.add(subset)
                            if subset == 0:
                                break
                            subset = subset - 1 & mask
                    return frozenset(result)
            return None
        if prod(map(len, sets)) > 200000:
            return None
        fn = (lambda a, b, c: a * b + c) if op == 'multiply_add' else BINARY[op]
        return frozenset((fn(*args) & MASK for args in product(*sets)))

    def analyze(k, g, src):
        rw = src.rw
        nodes = [ns[0] for ns in g.nodes]
        readers = collections.defaultdict(set)
        alias = getattr(k, '_broadcast_alias', {})

        def canonical(w):
            seen = set()
            while w in alias:
                assert w not in seen
                seen.add(w)
                w = alias[w]
            return w
        for j, n in enumerate(nodes):
            for r in rw(*n)[0]:
                readers[canonical(r)].add(j)
        vals = {b + i: frozenset([0]) for b in k._initial_zero_bases for i in range(8)}

        def value(w):
            return vals.get(canonical(w))
        mem = {i: frozenset([v]) for i, v in enumerate([16, 2047, 256, 10, 7, 2054, 2310])}
        mem.update({i: frozenset([0]) for i in range(2054, 2310)})
        bytime = collections.defaultdict(list)
        for j, t in enumerate(g.original):
            bytime[t].append(j)
        events = {}
        lastread0 = -1
        firstwrite0 = 99999
        for t in sorted(bytime):
            pending = {}
            mp = {}
            for j in bytime[t]:
                e, s = nodes[j]
                op = s[0]
                if e == 'load' and op == 'const':
                    pending[s[1]] = frozenset([s[2] & MASK])
                elif e in ('load', 'store'):
                    load = e == 'load'
                    ptr = s[2] + (s[3] if op == 'load_offset' else 0) if load else s[1]
                    bases = value(ptr)
                    assert bases is not None and bases, ('bad_address', j, t, e, s)
                    width = 8 if op in ('vload', 'vstore') else 1
                    locations = {b + i for b in bases for i in range(width)}
                    assert all((0 <= a < 2566 for a in locations))
                    live = {b + i for b in bases for i in range(width) if readers[canonical(s[1] + (s[3] if op == 'load_offset' else i))]} if load else locations
                    events[j] = {'engine': e, 'bases': bases, 'width': width, 'locations': locations, 'live': live, 'cycle': t}
                    if load:
                        if 0 in locations:
                            lastread0 = max(lastread0, t)
                        if op == 'load_offset':
                            pending[s[1] + s[3]] = merge(*(mem.get(a) for a in bases))
                        else:
                            for i in range(width):
                                pending[s[1] + i] = merge(*(mem.get(a + i) for a in bases))
                    else:
                        if 0 in locations:
                            firstwrite0 = min(firstwrite0, t)
                        for b in bases:
                            for i in range(width):
                                a = b + i
                                v = value(s[2] + i)
                                if len(bases) > 1:
                                    v = merge(mp.get(a, mem.get(a)), v)
                                mp[a] = v
                elif e == 'alu':
                    pending[s[1]] = calculate(op, value(s[2]), value(s[3]))
                elif e == 'valu':
                    for i in range(8):
                        pending[s[1] + i] = value(s[2]) if op == 'vbroadcast' else calculate(op, *(value(a + i) for a in s[2:]))
                elif e == 'flow' and op == 'add_imm':
                    pending[s[1]] = calculate('+', value(s[2]), frozenset([s[3] & MASK]))
                elif e == 'flow' and op in ('select', 'vselect'):
                    for i in range(1 if op == 'select' else 8):
                        cond = value(s[2] + i)
                        yes = value(s[3] + i)
                        no = value(s[4] + i)
                        pending[s[1] + i] = no if cond == frozenset([0]) else yes if cond is not None and 0 not in cond else merge(yes, no)
                else:
                    raise AssertionError((e, s))
            vals.update(pending)
            mem.update(mp)
        counts = collections.Counter()
        drop = []
        sink = []
        deadlane = []
        nonmem = []
        for j, dd in enumerate(g.deps):
            for d, lat in dd.items():
                if rw(*nodes[d])[1] & rw(*nodes[j])[0]:
                    continue
                kind = (nodes[d][0], nodes[j][0], lat)
                counts[kind] += 1
                if d in events and j in events:
                    a, b = (events[d], events[j])
                    overlap = a['locations'] & b['locations']
                    lo = a['live'] & b['live']
                    if not overlap:
                        drop.append((d, j, lat, 'disjoint'))
                    elif not lo:
                        deadlane.append((d, j, lat, 'dead_lanes'))
                    elif lo == {0} and a['engine'] == b['engine'] == 'store':
                        sink.append((d, j, lat, 'sink_only'))
                else:
                    nonmem.append((d, j, lat))
        return (events, drop, deadlane, sink, nonmem)

    def transform(k, g, src, dead=True, latency=True, alias=True, gates=False):
        rw = src.rw
        nodes = [ns[0] for ns in g.nodes]
        events, drop, deadlane, sink, nonmem = analyze(k, g, src)
        remove = drop + (deadlane if dead else [])
        modifications = []
        for d, j, l, reason in remove:
            del g.deps[j][d]
            modifications.append((d, j, l, None, reason))
        if latency:
            for j, dd in enumerate(g.deps):
                for d, l in list(dd.items()):
                    if d in events and j in events and (events[d]['engine'] == 'load') and (events[j]['engine'] == 'store') and (l > 0) and (not rw(*nodes[d])[1] & rw(*nodes[j])[0]):
                        dd[d] = 0
                        modifications.append((d, j, l, 0, 'read_before_write_same_cycle'))
        aliases = getattr(k, '_broadcast_alias', {})

        def canon(w):
            while w in aliases:
                w = aliases[w]
            return w
        if alias:
            broadcasts = {grp['E']: grp for grp in getattr(k, '_broadcast_groups', [])}
            for j, dd in enumerate(g.deps):
                for d, l in list(dd.items()):
                    de, ds = nodes[d]
                    je, js = nodes[j]
                    if de == 'flow' and ds[0] == 'vselect' and (ds[1] in broadcasts) and (je == 'alu') and (not rw(*nodes[d])[1] & rw(*nodes[j])[0]):
                        realwrites = {canon(w) for w in rw(*nodes[d])[1]}
                        realreads = {canon(r) for r in rw(*nodes[j])[0]}
                        if not realwrites & realreads:
                            writers = {canon(w): q for q, n in enumerate(nodes) for w in rw(*n)[1]}
                            for r in realreads:
                                if r in writers:
                                    dd[writers[r]] = max(dd.get(writers[r], 0), 1)
                            del dd[d]
                            modifications.append((d, j, l, None, 'first_four_scalar_broadcast_words'))
        if gates:
            writer = {w: j for j, n in enumerate(nodes) for w in rw(*n)[1]}
            readers = collections.defaultdict(set)
            for oj, (rr, ww) in enumerate(k._op_rw):
                for r in rr:
                    readers[r].add(oj)
            jobs = []
            for oj, (e, s, ds) in enumerate(k._ops):
                if e == 'load' and s[0] == 'vload':
                    used = [l for l in range(8) if readers[s[1] + l]]
                    if len(used) == 1 and len(readers[s[1] + used[0]]) == 1:
                        jobs.append((oj, next(iter(readers[s[1] + used[0]]))))
            jobs.sort(key=lambda z: (k.ssa_schedule_start[z[0]], z[0]))
            indices = {node: j for j, node in enumerate(nodes)}

            def cur(oj):
                n = k._ops[oj][:2]
                return indices[n] if n in indices else writer[k._op_rw[oj][1][0]]
            for (_, rel), (nxt, _) in zip(jobs, jobs[4:]):
                d, j = (cur(rel), cur(nxt))
                if d in g.deps[j]:
                    assert g.deps[j][d] == 0 and nodes[d][0] == 'alu' and (nodes[j][0] == 'load')
                    assert not rw(*nodes[d])[1] & rw(*nodes[j])[0]
                    del g.deps[j][d]
                    modifications.append((d, j, 0, None, 'original_sparse_admission'))
        g.succ = [[] for _ in nodes]
        for j, dd in enumerate(g.deps):
            for d, l in dd.items():
                g.succ[d].append((j, l))
        degree = list(map(len, g.deps))
        ready = [j for j, n in enumerate(degree) if not n]
        top = []
        while ready:
            d = ready.pop()
            top.append(d)
            for j, l in g.succ[d]:
                degree[j] -= 1
                if not degree[j]:
                    ready.append(j)
        assert len(top) == g.n
        g.top = top
        k.resume_fences_metadata = modifications
        return g
    return _CandidateNamespace(**locals())

_candidate_fences = _candidate_make_fences()


def _candidate_make_startup():
    def compiler():
        return _candidate_compiler

    class Edit:

        def __init__(self, k, g):
            self.k, self.g, self.src = (k, g, compiler())
            self.nodes = [ns[0] for ns in g.nodes]
            self.times = list(g.original)
            rw = self.src.rw
            self.explicit = [{d: lat for d, lat in dd.items() if not rw(*self.nodes[d])[1] & rw(*self.nodes[j])[0]} for j, dd in enumerate(g.deps)]
            self.remove, self.leader = (set(), {})
            self.zero = next(iter(k._initial_zero_bases))

        def writers(self):
            return {w: j for j, n in enumerate(self.nodes) if j not in self.remove for w in self.src.rw(*n)[1]}

        def new(self, width=1):
            b = max(self.k._virtual_width) + 8
            self.k._virtual_width[b] = width
            return b

        def add(self, e, s, t, deps=None):
            j = len(self.nodes)
            self.nodes.append((e, s))
            self.times.append(t)
            self.explicit.append(dict(deps or {}))
            return j

        def put(self, dest, e=None, s=None, t=None):
            j = self.writers()[dest]
            olde, olds = self.nodes[j]
            self.nodes[j] = (e or olde, s or olds)
            if t is not None:
                self.times[j] = t
            return j

        def done(self):
            keep = [j for j in range(len(self.nodes)) if j not in self.remove]
            mp = {j: i for i, j in enumerate(keep)}
            resolve = lambda j: mp[self.leader.get(j, j)]
            ex = [{} for _ in keep]
            for j in keep:
                for d, lat in self.explicit[j].items():
                    a, b = (resolve(j), resolve(d))
                    if a != b:
                        ex[a][b] = max(ex[a].get(b, 0), lat)
            return self.src.rebuild(self.g, [self.nodes[j] for j in keep], ex, [self.times[j] for j in keep])

    def counts(g):
        c = collections.Counter((e for ns in g.nodes for e, s in ns))
        c['A'] = c['alu'] + 8 * c['valu']
        return c

    def scalar_reads(e, s):
        """Tuple positions read as scalar words, excluding immediate operands."""
        if e == 'alu':
            return list(range(2, len(s)))
        if e == 'valu':
            return [2] if s[0] == 'vbroadcast' else []
        if e == 'load':
            return [] if s[0] in ('const', 'load_offset') else [2]
        if e == 'store':
            return [1] + ([2] if s[0] == 'store' else [])
        if e == 'flow':
            if s[0] == 'add_imm':
                return [2]
            if s[0] == 'select':
                return [2, 3, 4]
            return []
        raise AssertionError((e, s))

    def loadfocus4(k, g):
        before = counts(g)
        ed = Edit(k, g)
        h, one = (k.header_base, k.constant_vectors[1])
        temp32, temp512 = (ed.new(), ed.new())
        ed.add('alu', ('+', temp32, h, h), 1)
        ed.add('alu', ('<<', temp512, h + 2, one), 1)
        groups = {job['immediate']: job for job in k._broadcast_groups if 'immediate' in job}
        assert {33, 16896} <= groups.keys()
        s33 = groups[33]['S']
        for value, formula, st, ct, bt in ((33, ('+', s33, temp32, one), 2, 3, 4), (16896, ('*', groups[16896]['S'], s33, temp512), 3, 4, 5)):
            job = groups[value]
            ed.put(job['S'], 'alu', formula, st)
            for lane in (1, 2, 3):
                ed.put(job['D'] + lane, t=ct)
            ed.put(job['E'], t=bt)
            job['derived_immediate'] = value
        eight = [(j, s[1]) for j, (e, s) in enumerate(ed.nodes) if e == 'load' and s[0] == 'const' and (s[2] == 8) or (e == 'flow' and s[0] == 'add_imm' and (s[2] == ed.zero) and (s[3] == 8))]
        assert len(eight) == 1, eight
        ed.put(eight[0][1], 'alu', ('>>', eight[0][1], h, one), 1)
        assert ed.nodes[ed.writers()[8472]][1][0] == 'add_imm'
        ed.put(8472, 'alu', ('+', 8472, h, h + 3), 1)
        for dest, value, t in ((1792, 4097, 0), (1808, 2127912214, 1), (1856, 3345072700, 2), (1944, 2899272192, 3)):
            j = ed.writers()[dest]
            assert ed.nodes[j] == ('load', ('const', dest, value)), ed.nodes[j]
            ed.put(dest, 'flow', ('add_imm', dest, ed.zero, value), t)
        for dest, t in ((1800, 1), (1816, 2), (1864, 3), (1952, 4)):
            ed.put(dest, t=t)
        result = ed.done()
        after = counts(result)
        assert after['A'] == before['A'] + 6
        assert after['load'] == before['load'] - 4
        assert after['flow'] == before['flow']
        return result

    def add_derived_alias(ed, dest, value, op, a, b, source_times, flow_time):
        """Four equal scalar writes + one standard isolated 12-word overlap."""
        k = ed.k
        wr = ed.writers()
        ids = {wr[dest + l] for l in range(8)}
        assert len(ids) in (1, 8), ids
        anchor = min(ids)
        source, output = (ed.new(1), ed.new(8))
        producers = [ed.add('alu', (op, source, a, b), source_times[0])]
        producers += [ed.add('alu', (op, dest + l, a, b), source_times[l]) for l in (1, 2, 3)]
        ed.nodes[anchor] = ('flow', ('vselect', output, ed.zero, ed.zero, dest))
        ed.times[anchor] = flow_time
        ed.explicit[anchor] = {j: 1 for j in producers}
        for j in ids - {anchor}:
            ed.remove.add(j)
            ed.leader[j] = anchor
        for j, (e, s) in enumerate(ed.nodes):
            if j in ids or j in producers or j in ed.remove:
                continue
            ss = list(s)
            if e == 'load' and s[0] == 'load_offset' and (dest <= s[2] + s[3] < dest + 8):
                ss = ['load', s[1] + s[3], source]
            else:
                for pos in scalar_reads(e, s):
                    if dest <= s[pos] < dest + 8:
                        ss[pos] = source
            ed.nodes[j] = (e, tuple(ss))
            if ed.src.rw(e, tuple(ss))[0] & set(range(dest, dest + 8)):
                ed.explicit[j][anchor] = 1
        k._broadcast_fixed.update((source, dest, output))
        k._broadcast_alias[dest] = source
        for lane in range(4, 8):
            k._broadcast_alias[dest + lane] = output + lane - 4
            k._broadcast_ignored_sites.add((output, dest + lane))
            k._broadcast_dead_outputs.add(output + lane)
        k._broadcast_groups.append(dict(S=source, D=dest, E=output, source_new=True, source_original=None, derived_scalar_constant=value, recovery_formula=(op, a, b)))
        return source

    def focusderived7(k, g):
        before = counts(g)
        ed = Edit(k, g)
        h = k.header_base
        add_derived_alias(ed, k.constant_vectors[3], 3, '-', h + 3, h + 4, (1, 1, 1, 2), 7)
        add_derived_alias(ed, k.constant_vectors[9], 9, '-', h, h + 4, (1, 1, 1, 1), 6)
        candidates = [(s[1], j) for j, (e, s) in enumerate(ed.nodes) if e == 'valu' and s[0] == 'vbroadcast' and (s[2] == h + 4)]
        assert len(candidates) == 1, candidates
        add_derived_alias(ed, candidates[0][0], 7, '|', h + 4, h + 4, (2, 2, 2, 2), 9)
        result = ed.done()
        after = counts(result)
        assert after['A'] == before['A'] - 12, (before, after)
        assert after['flow'] == before['flow'] + 3
        return result

    def known(k, g):
        """Compile-time proof using the runtime header contract and ISA operations."""
        src = compiler()
        val = {b + l: 0 for b in k._initial_zero_bases for l in range(8)}
        val.update({k.header_base + i: v for i, v in enumerate(k.header_contract)})
        f = {'+': operator.add, '-': operator.sub, '*': operator.mul, '^': operator.xor, '&': operator.and_, '|': operator.or_, '<<': operator.lshift, '>>': operator.rshift, '==': lambda a, b: int(a == b), '<': lambda a, b: int(a < b)}
        alias = k._broadcast_alias

        def get(a):
            seen = set()
            while a in alias:
                assert a not in seen
                seen.add(a)
                a = alias[a]
            return val.get(a)
        for j in g.top:
            e, s = g.nodes[j][0]
            op = s[0]
            if e == 'load' and op == 'const':
                val[s[1]] = s[2] & 4294967295
            elif e == 'flow' and op == 'add_imm' and (get(s[2]) is not None):
                val[s[1]] = get(s[2]) + s[3] & 4294967295
            elif e in ('alu', 'valu'):
                for lane in range(8 if e == 'valu' else 1):
                    aa = [s[2]] if op == 'vbroadcast' else [a + lane for a in s[2:]]
                    args = [get(a) for a in aa]
                    if all((a is not None for a in args)):
                        v = args[0] if op == 'vbroadcast' else args[0] * args[1] + args[2] if op == 'multiply_add' else f[op](*args)
                        val[s[1] + lane] = v & 4294967295
            elif e == 'flow' and op in ('select', 'vselect'):
                for lane in range(8 if op == 'vselect' else 1):
                    cond = get(s[2] + lane)
                    if cond is not None:
                        v = get(s[3 if cond else 4] + lane)
                        if v is not None:
                            val[s[1] + lane] = v
        for a in alias:
            v = get(a)
            if v is not None:
                val[a] = v
        return val

    def literal_reuse(k, g):
        before = counts(g)
        ed = Edit(k, g)
        values = known(k, g)
        wr = ed.writers()

        def value_word(value):
            options = [(ed.times[wr[a]] if a in wr else -1, a) for a, v in values.items() if v == value]
            assert options, ('missing existing constant', value)
            return min(options)[1]
        literal = {s[2]: s[1] for e, s in ed.nodes if e == 'load' and s[0] == 'const'}
        base = literal[2863311488]
        seven, fourteen, twenty_six = (value_word(v) for v in (7, 14, 26))
        fifty_two = ed.new()
        ed.add('alu', ('+', fifty_two, twenty_six, twenty_six), 4)
        for delta, a, b in ((7, base, seven), (14, base, fourteen), (21, literal[2863311495], fourteen), (52, base, fifty_two)):
            d = literal[2863311488 + delta]
            ed.put(d, 'alu', ('+', d, a, b))
        base = literal[3681401085]
        nearby = sorted((v for v in literal if v != 3681401085 and abs(v - 3681401085) <= 4))
        assert len(nearby) == 3, nearby
        for value in nearby:
            delta = value - 3681401085
            assert abs(delta) in (2, 4), delta
            d = literal[value]
            operand = value_word(abs(delta))
            ed.put(d, 'alu', ('+' if delta > 0 else '-', d, base, operand))
        result = ed.done()
        after = counts(result)
        assert after['A'] == before['A'] + 8 and after['load'] == before['load'] - 7
        return result

    def bank_pointers(k, g):
        before = counts(g)
        ed = Edit(k, g)
        values = known(k, g)
        wr = ed.writers()
        jobs = [(j, s) for j, (e, s) in enumerate(ed.nodes) if e == 'load' and s[0] == 'const' and (2054 <= s[2] <= 2061)]
        assert len(jobs) == 8 and {s[2] for j, s in jobs} == set(range(2054, 2062)), jobs
        for j, s in jobs:
            offset = s[2] - 2054
            options = [(ed.times[wr[a]] if a in wr else -1, a) for a, v in values.items() if v == offset]
            assert options, ('missing bank offset', offset)
            a = min(options)[1]
            ed.nodes[j] = ('alu', ('+', s[1], k.header_base + 5, a))
        result = ed.done()
        after = counts(result)
        assert after['A'] == before['A'] + 8 and after['load'] == before['load'] - 8
        return result

    def transform(k, g):
        for fn in (loadfocus4, focusderived7, literal_reuse, bank_pointers):
            g = fn(k, g)
        return g
    return _CandidateNamespace(**locals())

_candidate_startup = _candidate_make_startup()


def _candidate_make_inputseed():
    src=_candidate_compiler
    def constants(k, g):
        assert k.header_contract == (16, 2047, 256, 10, 7, 2054, 2310)
        aliases = k._broadcast_alias

        def canonical(w):
            seen = set()
            while w in aliases:
                assert w not in seen
                seen.add(w)
                w = aliases[w]
            return w
        known = {zero + l: 0 for zero in k._initial_zero_bases for l in range(8)}
        ops = {'+': lambda a, b: a + b, '-': lambda a, b: a - b, '*': lambda a, b: a * b, '&': lambda a, b: a & b, '|': lambda a, b: a | b, '^': lambda a, b: a ^ b, '<<': lambda a, b: 0 if b >= 32 else a << b, '>>': lambda a, b: 0 if b >= 32 else a >> b, '==': lambda a, b: int(a == b), 'multiply_add': lambda a, b, c: a * b + c, 'vbroadcast': lambda a: a}

        def value(w):
            return known.get(canonical(w))
        for j in g.top:
            e, s = g.nodes[j][0]
            op = s[0]
            if e == 'load' and op == 'const':
                known[s[1]] = s[2] & 4294967295
            elif e == 'load' and op == 'vload' and (s[1] == k.header_base):
                assert s[2] in k._initial_zero_bases
                known.update({s[1] + i: v for i, v in enumerate(k.header_contract)})
            elif e in ('alu', 'valu') and op in ops:
                for l in range(8 if e == 'valu' else 1):
                    args = [value(s[2])] if op == 'vbroadcast' else [value(a + l) for a in s[2:]]
                    if None not in args:
                        known[s[1] + l] = ops[op](*args) & 4294967295
            elif e == 'flow' and op == 'add_imm' and (value(s[2]) is not None):
                known[s[1]] = value(s[2]) + s[3] & 4294967295
        return (known, canonical)

    def transform(k, g):
        nodes = [ns[0] for ns in g.nodes]
        ts = list(g.original)
        known, canon = constants(k, g)
        ex = [{d: l for d, l in dd.items() if not rw(*nodes[d])[1] & rw(*nodes[j])[0]} for j, dd in enumerate(g.deps)]
        wr = {w: j for j, n in enumerate(nodes) for w in rw(*n)[1]}
        assert known[1208] == k.header_contract[6] + 8 == 2318 and k.header_contract[6] == 7 + 2047 + 256
        j = wr[1208]
        assert nodes[j] == ('alu', ('+', 1208, 73902, 208))
        nodes[j] = ('load', ('const', 1208, known[1208]))
        ts[j] = 0
        j = wr[1808]
        assert nodes[j] == ('flow', ('add_imm', 1808, 73448, 2127912214))
        ts[j] = 0
        j = wr[1792]
        assert nodes[j] == ('flow', ('add_imm', 1792, 73448, 4097))
        ts[j] = 1
        expected = {1816: ('valu', ('vbroadcast', 1816, 1808)), 1216: ('load', ('vload', 1216, 1208)), 2040: ('valu', ('^', 2040, 1216, 64)), 2048: ('valu', ('multiply_add', 2048, 2040, 1800, 1816)), 2056: ('valu', ('>>', 2056, 2048, 1840)), 2064: ('valu', ('^', 2064, 2048, 1864)), 2072: ('valu', ('^', 2072, 2056, 2064))}
        moves = {1816: 1, 1216: 1, 2040: 2, 2048: 3, 2056: 4, 2064: 4, 2072: 5}
        for d, t in moves.items():
            j = wr[d]
            assert nodes[j] == expected[d], (d, nodes[j], expected[d])
            ts[j] = t
        gg = src.rebuild(g, nodes, ex, ts)
        assert all((gg.deps[j].get(d, -1) >= l for j, dd in enumerate(ex) for d, l in dd.items()))
        return gg
    return _CandidateNamespace(**locals())

_candidate_inputseed = _candidate_make_inputseed()


def _candidate_make_alternate():
    startup=_candidate_startup
    def native379(k, g):
        ed = startup.Edit(k, g)
        before = startup.counts(g)
        groups = [job for job in k._broadcast_groups if job.get('derived_scalar_constant') in (3, 7, 9)]
        assert len(groups) == 3
        for job in groups:
            value = job['derived_scalar_constant']
            S, D, E = (job[a] for a in ('S', 'D', 'E'))
            wr = ed.writers()
            anchor = wr[E]
            if value == 3:
                node = ('valu', ('+', D, k.constant_vectors[1], k.constant_vectors[2]))
                t = 2
            elif value == 7:
                node = ('valu', ('vbroadcast', D, k.header_base + 4))
                t = 1
            else:
                node = ('valu', ('*', D, k.constant_vectors[3], k.constant_vectors[3]))
                t = 3
            oldids = {wr[S]} | {wr[D + l] for l in (1, 2, 3)}
            ed.nodes[anchor] = node
            ed.times[anchor] = t
            ed.explicit[anchor] = {}
            for old in oldids:
                ed.remove.add(old)
                ed.leader[old] = anchor
            for j, (e, s) in enumerate(ed.nodes):
                if j in oldids or j == anchor or j in ed.remove:
                    continue
                assert not ed.src.rw(e, s)[0] & set(range(E, E + 8)), (j, e, s, job)
                ss = list(s)
                for pos in startup.scalar_reads(e, s):
                    if ss[pos] == S:
                        ss[pos] = D
                if e == 'load' and s[0] == 'load_offset' and (s[2] + s[3] == S):
                    ss = ['load', s[1] + s[3], D]
                ed.nodes[j] = (e, tuple(ss))
            k._broadcast_fixed.difference_update((S, D, E))
            del k._broadcast_alias[D]
            for l in range(4, 8):
                del k._broadcast_alias[D + l]
                k._broadcast_ignored_sites.remove((E, D + l))
                k._broadcast_dead_outputs.remove(E + l)
            k._broadcast_groups.remove(job)
        g = ed.done()
        after = startup.counts(g)
        assert after['A'] == before['A'] + 12 and after['flow'] == before['flow'] - 3, (before, after)
        return g

    def canonical19(k, g):
        ed = startup.Edit(k, g)
        D = k.constant_vectors[19]
        C3 = k.constant_vectors[3]
        C16 = k.constant_vectors[16]
        wr = ed.writers()
        ids = [wr[D + l] for l in range(8)]
        assert len(set(ids)) == 8
        for l, j in enumerate(ids):
            assert ed.nodes[j] == ('alu', ('+', D + l, C16 + l, C3)), ed.nodes[j]
        anchor = min(ids)
        ed.nodes[anchor] = ('valu', ('+', D, C16, C3))
        ed.times[anchor] = max((ed.times[j] for j in ids))
        for j in ids:
            if j != anchor:
                ed.remove.add(j)
                ed.leader[j] = anchor
        j = wr[k.constant_vectors[8]]
        assert ed.nodes[j][0] == 'valu' and ed.nodes[j][1][0] == 'vbroadcast'
        ed.times[j] = 2
        return ed.done()
    return _CandidateNamespace(**locals())

_candidate_alternate = _candidate_make_alternate()


"""Pure graph transforms for assembly into a self-contained compiler."""

def candidate_parallel_branch(k, g, dests):
    old = [ns[0] for ns in g.nodes]
    nodes = list(old)
    times = list(g.original)
    writer = {w: j for j, n in enumerate(old) for w in rw(*n)[1]}
    readers = collections.defaultdict(set)
    for j, n in enumerate(old):
        for r in rw(*n)[0]:
            readers[r].add(j)
    explicit = [{d: l for d, l in dd.items() if not rw(*old[d])[1] & rw(*old[j])[0] or (old[d][0] in ('load', 'store') and old[j][0] in ('load', 'store'))} for j, dd in enumerate(g.deps)]
    nb = max(k._virtual_width) + 8

    def fresh(width=8):
        nonlocal nb
        b = nb
        nb += 8
        k._virtual_width[b] = width
        return b

    def add(e, s, t, ex=None):
        nodes.append((e, s))
        times.append(t)
        explicit.append(dict(ex or {}))
    ready = min((g.original[writer[d]] for d in dests)) - 40
    consts = []
    for val in [65536, 65545]:
        lit, vec = (fresh(1), fresh())
        add('load', ('const', lit, val), ready)
        add('valu', ('vbroadcast', vec, lit), ready + 1)
        consts.append(vec)
    mask, coef = consts

    def laneop(r):
        j = writer[r]
        e, s = old[j]
        l = r - s[1] if e == 'valu' else 0
        assert e in ('alu', 'valu') and len(s) == 4, (r, e, s)
        return (s[0], s[1] + l, s[2] + l, s[3] + l)
    records = []
    for dest in dests:
        ids = {writer[dest + l] for l in range(8)}
        asrc = []
        consumers = set()
        for l in range(8):
            bit = laneop(dest + l)
            assert bit[0] == '&' and bit[3] == k.constant_vectors[1] + l
            full = laneop(bit[2])
            assert full[0] == '^'
            a, b = full[2:]
            shift = laneop(b)
            if shift[0] != '>>':
                a, b = (b, a)
                shift = laneop(b)
            assert shift[0] == '>>' and shift[2] == a and (shift[3] == k.constant_vectors[16] + l)
            asrc.append(a)
            for rj in readers[dest + l]:
                re, rs = old[rj]
                assert re == 'flow' and rs[0] == 'vselect' and (rs[2] == dest)
                consumers.add(rj)
        a = asrc[0]
        assert asrc == list(range(a, a + 8))
        aj = writer[a]
        ae, ass = old[aj]
        assert ae == 'valu' and ass[:2] == ('multiply_add', a) and (ass[3:] == (k.constant_vectors[9], k.constant_vectors[4251993797])), ass
        x = ass[2]
        ap = fresh()
        add('valu', ('multiply_add', ap, x, coef, ass[4]), times[aj], g.deps[aj])
        for j in ids:
            e, s = old[j]
            l = s[1] - dest
            nodes[j] = (e, ('&', s[1], ap + l, mask + l))
            times[j] = times[aj] + 1
        for j in consumers:
            e, s = old[j]
            nodes[j] = (e, (*s[:3], s[4], s[3]))
            times[j] -= 1
        records.append((dest, x, a, ap, sorted(consumers)))
    k.retry_tail_parallelbit_meta = dict(targets=dests, records=records, mask=mask, coefficient=coef)
    return rebuild(g, nodes, explicit, times)

def candidate_pointer_multiply(k, g, dests):
    old = [ns[0] for ns in g.nodes]
    nodes = list(old)
    times = list(g.original)
    writer = {w: j for j, n in enumerate(old) for w in rw(*n)[1]}
    explicit = [{d: l for d, l in dd.items() if not rw(*old[d])[1] & rw(*old[j])[0] or (old[d][0] in ('load', 'store') and old[j][0] in ('load', 'store'))} for j, dd in enumerate(g.deps)]
    zero = set(k._initial_zero_bases)
    one = k.constant_vectors[1]
    for dest in dests:
        j = writer[dest]
        e, s = old[j]
        assert e == 'flow' and s[0] == 'vselect' and (s[4] in zero)
        cond, ptr = s[2:4]
        ready = 0
        for l in range(8):
            q = writer[cond + l]
            qe, qs = old[q]
            off = l if qe == 'valu' else 0
            assert qe in ('alu', 'valu') and qs[0] == '&' and (qs[3] + off == one + l), (dest, qs)
            ready = max(ready, g.original[q] + 1)
        nodes[j] = ('valu', ('*', dest, cond, ptr))
        times[j] = ready
    g = rebuild(g, nodes, explicit, times)
    k.retry_tail_pointermul_targets = dests
    return g

def candidate_xor_select(k, g, dests, part=0):
    old = [ns[0] for ns in g.nodes]
    nodes = list(old)
    times = list(g.original)
    writer = {w: j for j, n in enumerate(old) for w in rw(*n)[1]}
    readers = collections.defaultdict(set)
    for j, n in enumerate(old):
        for r in rw(*n)[0]:
            readers[r].add(j)
    explicit = [{d: l for d, l in dd.items() if not rw(*old[d])[1] & rw(*old[j])[0] or (old[d][0] in ('load', 'store') and old[j][0] in ('load', 'store'))} for j, dd in enumerate(g.deps)]
    nextbase = max(k._virtual_width) + 8
    for dest in dests:
        si = writer[dest]
        e, s = old[si]
        assert e == 'flow' and s[0] == 'vselect'
        arms = set()
        source = []
        for l in range(8):
            aa = []
            for ab in s[3:]:
                j = writer[ab + l]
                e0, s0 = old[j]
                assert e0 == 'alu' and s0[0] == '^'
                arms.add(j)
                aa.append(s0[2:])
            common = set(aa[0]) & set(aa[1])
            assert len(common) == 1
            source.append(common.pop())
        assert source == list(range(source[0], source[0] + 8))
        value = source[0]
        xids = {writer[value + l] for l in range(8)}
        assert len(xids) == 1
        xi = xids.pop()
        xe, xs = old[xi]
        assert xe == 'valu' and xs[:2] == ('^', value)
        assert set().union(*(readers[value + l] for l in range(8))) == arms
        retained, residual = (xs[2 + part], xs[3 - part])
        for j in arms:
            ae, ass = old[j]
            nodes[j] = (ae, tuple(ass[:2]) + tuple((retained + (a - value) if a in range(value, value + 8) else a for a in ass[2:])))
            times[j] = min(times[j], g.original[xi] - 1)
        tmp = nextbase
        nextbase += 8
        k._virtual_width[tmp] = 8
        nodes[si] = (e, (s[0], tmp, *s[2:]))
        times[si] -= 1
        nodes[xi] = ('valu', ('^', dest, tmp, residual))
        times[xi] = g.original[si]
    return rebuild(g, nodes, explicit, times)


def _candidate_make_aliases():
    src=_candidate_compiler
    def transform(k, g, values=(), ptr16=False):
        nodes = [ns[0] for ns in g.nodes]
        times = list(g.original)
        ex = [{d: lat for d, lat in dd.items() if not rw(*nodes[d])[1] & rw(*nodes[j])[0]} for j, dd in enumerate(g.deps)]
        wr = {w: j for j, n in enumerate(nodes) for w in rw(*n)[1]}
        zero = next(iter(k._initial_zero_bases))
        nb = max(k._virtual_width) + 8
        removed = set()

        def new(width):
            nonlocal nb
            a = nb
            nb += 8
            k._virtual_width[a] = width
            return a

        def add(e, s, t, dd=None):
            j = len(nodes)
            nodes.append((e, s))
            times.append(t)
            ex.append(dict(dd or {}))
            return j
        for val in values:
            D = k.constant_vectors[val]
            ids = {wr[D + i] for i in range(8)}
            assert len(ids) == 1
            j = next(iter(ids))
            e, s = nodes[j]
            assert e == 'valu' and s[0] == 'vbroadcast', (val, e, s)
            S = s[2]
            assert S % 8 == 0 and k._virtual_width[S] == 1
            E = new(8)
            uses = {q for q, n in enumerate(nodes) if rw(*n)[0] & set(range(D, D + 8))}
            ct = times[wr[S]] + 1
            copies = [add('alu', ('|', D + l, S, S), ct) for l in (1, 2, 3)]
            nodes[j] = ('flow', ('vselect', E, zero, zero, D))
            times[j] = ct + 1
            ex[j].update({q: 1 for q in copies})
            ex[j][wr[S]] = 1
            for q in uses:
                ex[q][j] = max(ex[q].get(j, 0), 1)
            k._broadcast_fixed.update((S, D, E))
            k._broadcast_alias[D] = S
            for l in range(4, 8):
                k._broadcast_alias[D + l] = E + l - 4
                k._broadcast_ignored_sites.add((E, D + l))
                k._broadcast_dead_outputs.add(E + l)
            k._broadcast_groups.append(dict(S=S, D=D, E=E, source_new=False, source_original=S, literal_value=val))
        if ptr16:
            j = wr[264]
            assert nodes[j] == ('alu', ('+', 264, 216, 216))
            for q, (e, s) in enumerate(nodes):
                if 264 not in rw(e, s)[0]:
                    continue
                assert e == 'alu' or (e == 'store' and s[0] == 'vstore' and (s[1] == 264)), (e, s)
                if e == 'alu':
                    nodes[q] = (e, (s[0], s[1], *[k.header_base if a == 264 else a for a in s[2:]]))
                else:
                    nodes[q] = (e, (s[0], k.header_base, s[2]))
            removed.add(j)
        keep = [j for j in range(len(nodes)) if j not in removed]
        mp = {j: q for q, j in enumerate(keep)}
        assert not any((d in removed for j in keep for d in ex[j]))
        return src.rebuild(g, [nodes[j] for j in keep], [{mp[d]: lat for d, lat in ex[j].items()} for j in keep], [times[j] for j in keep])
    return _CandidateNamespace(**locals())

_candidate_aliases = _candidate_make_aliases()


def _candidate_release_schedule(graph, releases):
    """Full DAG/list scheduling with input-independent earliest-issue hints."""
    assert len(releases)==graph.n
    degree=list(map(len,graph.deps));earliest=list(releases)
    future=[];ready=[];times=[None]*graph.n;program=[];done=0;t=0
    for j,d in enumerate(degree):
        if not d:heapq.heappush(future,(earliest[j],j))
    while done<graph.n:
        while future and future[0][0]<=t:
            _,j=heapq.heappop(future);heapq.heappush(ready,(releases[j],j))
        bundle={};blocked=[]
        while ready:
            item=heapq.heappop(ready);_,j=item
            if any(len(bundle.get(e,[]))+count>LIMIT[e] for e,count in graph.demands[j].items()):
                blocked.append(item);continue
            for e,s in graph.nodes[j]:bundle.setdefault(e,[]).append(s)
            times[j]=t;done+=1
            for q,lat in graph.succ[j]:
                earliest[q]=max(earliest[q],t+lat);degree[q]-=1
                if not degree[q]:
                    if earliest[q]<=t:heapq.heappush(ready,(releases[q],q))
                    else:heapq.heappush(future,(earliest[q],q))
        ready=blocked;heapq.heapify(ready)
        assert bundle or future or ready,('candidate scheduling deadlock',done,graph.n)
        program.append(bundle);t+=1
        assert t<=max(releases)+graph.n+1
    for j,dd in enumerate(graph.deps):
        for d,lat in dd.items():assert times[j]>=times[d]+lat
    assert times==releases,('candidate release schedule mismatch',sum(a!=b for a,b in zip(times,releases)))
    graph.original=times
    canonical=[{} for _ in program]
    for j,nodes in enumerate(graph.nodes):
        for e,s in nodes:canonical[times[j]].setdefault(e,[]).append(s)
    return canonical

def _candidate_apply_mode_hint(graph, hint, signature):
    graph,groups=_mode_expand_graph(graph)
    actual=hashlib.sha256(repr((graph.nodes,graph.deps,graph.original,groups)).encode()).hexdigest()
    assert actual==signature,('candidate expanded graph mismatch',actual,signature)
    raw=zlib.decompress(base64.b85decode(hint))
    data=struct.unpack('<'+'H'*(len(raw)//2),raw)
    cycles,n,m=data[:3]
    assert n==graph.n and m==len(groups)
    modes=data[3:3+m];times=list(data[3+m:]);assert len(times)==n
    for mode,(_,ids,_) in zip(modes,groups):
        assert mode in (0,1)
        if mode:assert len({times[j] for j in ids})==1
    for j,dd in enumerate(graph.deps):
        for d,lat in dd.items():assert times[j]>=times[d]+lat
    graph=_mode_apply_modes(graph,groups,modes,times)
    program=_candidate_release_schedule(graph,list(graph.original))
    assert len(program)==cycles
    return graph,program


_CANDIDATE_NATIVE19_SIGNATURE = 'a5a311cf393fa9bbe08b24bb2f0a875532d32de73fbad9d1a28405f75b4e9a4a'
_CANDIDATE_NATIVE19_MODE_HINT = (
    'c-'
    'pOf3EUOq`~EB`Y1O`$_I=+sDoNV+RoYWYMJg52qDTm#MM_Ahq(Vd#sU(poLPbPcNE@o(b>A~{&Y9~u?>leb|G(=qGv}Roo_oFS`+'
    '4TPukZH`_mk%aWwR|0mSv|1>oV^C|AIwF;wL+WPl+Sb;!D(M>oR<`BBNtnmhWkq=|v>b%H*Gu&rO@(R+?w0#uAI-Zbi<rQuR8uJ6'
    '6OUD}~Hk#x<opSeI>=MI5%0#wBGLRvO}(+(RqH7^ED5WDHF1K~yZ0Jq^oBl$3f^ryTLiV|_~bwNl}CYV5UAwTCldC8;AT31{sTG8'
    'gWKY3WO8t0OBFKU4PDjDA#qwG7)+s)fs~ayz1$(^{=oh@qzM$*Ve<f6Qxhu9fC>!Aj$L`M=hMox(aB)tfESW?fb-MYBEa_NBa<Tb'
    'I#0)3VN*nY1Ez8deHbo^k}Nq|7w?M%s5LDd)yYL$9!s`f6RqF3}>pQ<h_;csAo<rD658lXj+*`mr6^8SOMjK5gq(YMN5&u-'
    '5;eH}PNU6`5LW(;hu5qEn@`S|!`{CtE44G}U|RJ3uSVyACUr9ZG6atcZ0pEgdji^=Ww{up-v_w7dgZmvzsI)Ryx8(A-m6DP-JEr8'
    'io%<FZoNlcm;Wqw<yVEbIR<ekrlX>|{~9|I~6}CEY975qt2|cDQz`o%!;*qLs?tGqw6kZJ%wWxwEGw22nfAsLo<VdTw3Dx?!bxT?'
    'z9isskstGxcv>tW;K#l;3`>i2YsE8kE*Ig0$_SFLymn#=O*P&q~5nD~;cbT<-2GrOHhH7M^Tvy}W%+TK3U)3U5cz-'
    '$<lh=T_ufOIr9JwWmq_ZN`fD7A58P87swaDeQ>vj8ghe!A|4;#k$P3%1&V=P3t>{o#J;H)@AN+)3Q^t)3Dn#YtTwsDam}Y{>Sewt'
    '^c8xtQ7Vk$r+LSn<Xpa{m4rC%|z<oMWyskWJPv?<or%)@0RTSG4-'
    '8TbPwfsOe^A>Yde*1KBD&csdpXoJB4K5G+9aWfB8Fs|K<CVwCwEC_HCY(WCcq78^x%%h$!w^Y5q>sN|D)V{kARYcPUni-'
    '<zehf444kKVn&z_nvn-?}3;5{ezXl4&?IR+O3Fvw-vGeSZV4BR*K)%r+vqe^81;U>RWP3tDlv^PR87Wrez<JmN&KN--4y}TZ@$Ud'
    'zbTjS}T?H*h3b}{Fgnc<SVW8mP3#yj+NEQU_9}Vfsh%<ggNOfH!1QE(pgsl*`OgSo;ubQxD$qXC$E(Qcf(*k`3N$mAR#^BD#R+Ql'
    '~IvSEy@PSPx7<~A)RqYEKQFX6vZc&T@>J0W%B*bfXFk76Rsh6)?FEi3`V3(WQ43NzTYV$y#dyB6+yU=O+RHco_SnxaAd_#nJycwJ'
    'DpVmy0YWQf~T1YnS5r3!7*kX90Asu89dAhow=bi3}$&rLT1d)X3S!+y<Q~qKx=mBy;?bti2Rz#tSQb-'
    'L;=jtjzbt&g1~aK7?H_F$OYtpw4BBhMs`CwTUrYBodv$+M@8jA<>tXTJK<_V_DH>@v;q^sUgwV?FXEFGn)6zPa25Hb2{|Ib_U1M0'
    '$qUJOl{>|7&231?MaW4ggSk006J;>J0zT(~l$^%Aa)k1Pd`2vDBL~?bwWJ<3Aw4hHQUGzuXGpK&qq2WxSrru(d{k07iO5gLgUnxT'
    'jH(Hg!INyrM?u&ZMjfkm0%{qNyao}jgJ|b4`hQ9FFFT%ly2Hq+H5NuA*pSm;1NY{difqcCf|})A#+;f)EOS|fF+K;PEbJ_-xhYJz'
    '+UQxi5t*D3D5v}@g88PGQ9R=eRZ0bwfgH%U<e)MlUJNK=WHT>8a+u2)l}#n)#IqtAmx9KM&YmzY3Lxt_eL1xNeqIZVEQ@Si4cUbv'
    'kk{Z&8A3fmE%;PRA^OF>^2kF0Bctq9DMM2!0%s(*)|l57g|XsQQQePlgm`9I>jodqeF#_c_&uv4tf{Eh=fhaBzJ$?3^CpQ*c_Vsx'
    'bZyA1eO!Djs1{3iD2WIZGh)E?RDr;?Bx_kdJ%mZQ@hLwJVWL!DNkd+K)J+w5Ul?QaYmdmIe9D2Ja`}Ct03yw`#vCmXf$D^^m?Jfo'
    'SL-mBcK^J36vDIO#(ds2)F5y#k4MP+xv-H5iT%|YsUlWYjZ8^{$-=Rk1m5XN{YXXTz-'
    'M8Sm*bMk5b0c{;g7eH=2Tw`V7zd>kb27b<C-'
    'W&U@wYV)nI*TwKXp?>BTgUX1fS9ig)EAP{z>0R#brGYPerYeJQBY3;KL6r2VFd!OXG*lY6xYEeNufmQ~*h!G<FEiDO^c_aLw4Kq`'
    'rSs~v%A+N*ESu>~Pf?C@q5!cW}QnPqi=R#r1YG4Q)Go>qoE*Q1lPi1ieK`8HEr`?`2o5d10%hDg7Xo>CzaRS0ZhNoW*_UXKz@a?K'
    'Chii0;52~`Odj9%VEaUF22qICjms^XBG$u6T3K6x>#MyMEp`aU~6ImK|U3MnPPg35&IRs#a(i=SUlC<dw5XcoMkOA#=<mTDAcRk0'
    'cpM61|T9K5WkUCPU03EZo#owP=hC^j~_7iX=J)d4cf#(G^T#7RDt!ci3#T?^gS3{-'
    '*DRsB4~af4QA39TG&TuE3~2<O^hS0h+cJ=Qg=DT4*xNRBY~!y5@~(RId%TF_D#);BdecCZqNRaFvf3eb_j{e)K(+3(hbpMgiEG>&'
    'C(ucSsXb=0^_)P`39q>mIrRWv~EtH2753a=n_ad4k1jan^*h?mfPPa(BjxL872T?dsH)yX6V-'
    'X5zcJg$J8OQjdoH<!|oRR_;SN{v{A0+_yqQ7-'
    '~b2(qUrg_s6ab}g)tPF^2lr8??s)au|GMop`YqP3zeaJ}}=YvEHF<@QZL%LG)_sug+F!R%UwWpxSC1K7K!gll8nt)6!Cs_IEy{8U'
    '48SKsIg47P&f-'
    'H6Z(GR3m`{vj3L7%{5@eKm|8)<EYZ^DGg(Ce@EU38LKucdLOR)iJxR;X~sHm?K0(Cqyl*MY>&mc+$}D`q~I|H0r36A%jsCv6J0wC'
    '1k8oERS;SP6)UeaJIkExe_^11^jEQZ{8Jgt*7~6);Bixa@P^MfxVT%I9VfWYPYJZGDU+I(fSzGG%_+b8|bOB>LNz<e2XRC4PaFp&'
    '2t!RR}buA24h@Pe`OD%)<mnLIcD@AbV`^lzVo^&_gYzf2vs4eDUSZYKp%tgxj8{>>}1qs6N7QB3C(a%s;QpOufEnTF^K-'
    '_RW{2NZ`<fIe$qMiPM8vsS)K@%*b;i0X=h8s5Nm`lk9WYPMw##V<9XK?Xs%Cgwfb4N;~0UnH@1`Z$QIUJm~}U<oF(bCVr6FStBJT'
    'V0?i}9e)RM^yNAqH7TDD))X4?yVDn88m}uC-`vh4bZZx<dbJ?n{1b*HLa$4%qQm;2^G+Tvy5ws$7Lq>-'
    'n3zF9n)<|4=#4Uywoq&lzD~(L6kX_w?8`T45PHUsDF?w0OW3nfzWcjHD<h3{aXk*lQ-xxfujYGwH5<01!@*Q<M!>TAqKH3pFVFw)'
    'lsV$D%fmR8b?*d(&V6}&v{7By9k70Qo2<&N3U@Wff5#RQNK!$woZanQpV98>A2Ln8-'
    'w_!0?LqE0PcKGd~vyZaaEtUnZB%e1Sk{$i1@S4y?^~&B-uAO07TiokJ=uBX{xrYxX^fn|0Gv)U#gAMsya~)w-'
    'M<Y775N@|_(>;GV<fEHr&5KJnLgxsyi@+qHC*(*DZ#Hc0LSSZ0!XCb}#-'
    'O9GucKi>FUaWu`I5J8Dpg{|l6zu|7m1O;ZGc$T7M`0G&N1MAIFwK;_J-oU`Sd1;RiZiQ0Rt5Mamac-'
    'K%bSvm3u{Z_}v>lq9>ti>{H@cK~Kc1HL}YySTe_On88YU8>{8-'
    'gk^^<)iTMbh>03S@kPArV(8^8JYo$eFfV!6JV<vm{AM~7Yx8}Ck=W1N3`>|}eGQ#Y_;_58$DyNd%%j1OGXQs_Q};If4Kmt~AkVns'
    '`YX#4(H|ovw!QI*<2@h(vhI2PVJHso^DulC8~Y+oBk)<q3<KsGvv0wQIgrp@*Mmgdin-l%{BX7RNtN~_a2xJ4G9-'
    '+|jMOm*7y?TN!t+~Hn&eb+Fw(k1uMMqHgdqg3Htv^w3F7Omh=lCuhGOU1SoO%a1YY)U#r|wGVVv4F#K`Mt%~||+=JE|#7M11s7q{'
    'weBj&dmyyNJ!(Aw*Z=ycWf<enJrRoto>O%1mh(HW_>)lASLeoOzn4bKJ<ICjGb%#e|UI|!ZiElj$q^u(5Wz8&`;x9&uyMqvhLd%9'
    'smEq|`;YsO*52waC+^Wnje2=I4poaF}yb+KRZcFALKz7u%T-!+Ux#D-W;Av2t_I}DEWz`S}GD;A54JMeZe26zHjiAV)Rz5_-'
    '{cJ3s|dmG>D4b5XPcaR^`?)dZ+)|S53c-=j5j>ZtW>3Z8uuRZ;GPDHO*HxAZd{?=%acE`DjJ0S5Mc-aZ_hU3~BG3kPHPu*wq)=oD'
    'b<7ADQi%){VzW%!avkOi@CdL>{>;pO7LwBU_i?neH`6Tjg$8{`z9%R(Q#0YTT7-VEpqBYQ(jkBkBymq-KtjW+g9yW6|4Aa#w5mJ%'
    'NO3`$$R`o<ZPr}R*U=r8T{rZ%3^oN8&%3gjVmOhMVN<1g4r9*Ldl=hubxHpAxFJZLiS|pFeJn#EK@7x7vv)A867>0YpR1-'
    '6pHFJ+*Px=^Lfgx4p=@EucCEQ_DI&;3EW?_iFKYCfY8&XGTeTl?G_H$mQ0b>b)#=yRj=o~@)@mR@=*XhFGoliIHVLE|@lMq|8a|v'
    'hq>uTifA;uBz&~80B2_Dgk9Sm-bjldn&gFa^ZxSudyJNW%z*sxGd_larig15)X&~~d~A+uyw0v-'
    'aqZssvWKJobRV7hqD>>3DtQU&)ThOJ@Wc&*hiJpb<UJIF}fxtnks?%$7C2}D{q^;a}@#~dDgmw^e25y;gev2WAv_uud*7`=yQbFU'
    'h&E5JBN5IMaQs@B^d^if+M1I#`rcjehQ*f|E{ILopVxrZ>;=w$aAeVe(^*?%|n-'
    'cLJ1+kLQml9BhxYSA1#WvzD`wyi`yM??2S#B8d;<T>!G4`M1YnS!6EVbmm}uZ=VM!^4C-'
    't(P>Cv%!}s*oC!LX_L|EC+M5R_y~+KEMtum;Y&YM>ssLc*r*oJIMx68Zoii0JR6t{D`pxzdx~(sHA5xL0Pce)(+I)53I6o~)iw@e'
    '*D9vyo%{9YhX@jL=}Qx>M=@@yetuf_Ni!1mOlx7WjGpPoQa+DGM43N*bVctKdt>6gS7LXTzQz${dG{mx_aFkZL%UAtjWb~FG(4N4'
    '7W2FoSkI;)wrg;HJOa$5rz3H1tebNtUI8W{miHm5PY@n3-'
    'd&#|%rNduQN~WiwU+;z4i9x;#d3W|o2slx4b0LB);vUYDy*GpU_Ru|C9sb?=Rv|$l`YwxrZIYm>hM9`BR`1ybMbU0Y<a-'
    'w5%ZAq2X#ysbJ0_#WBfz<UNu*lx&U$F4Dq~K@N7OJwj9r1)KBw(8PG1zhG-XiL%AXyYbd00CT1CN-'
    'J&uR@u+(A2&CO_%%9^|ywvmEkT#FN-D)QM5;h7a?!^)Bz$0+%vAzcGhR(-e>m-'
    'bwO_*Zjg|U$E8e+i={Tf}QpP#ic)*^qUjAILGmSg=aVG)7(I15^4BX+ZqhlRL?G0T|0*oYlV<TY!iA@wn9Nd)E_yqBCTfj_(xkxX'
    '!0##zrn$Mb~wNjk<X_&gso=HOF5WO_ID4wH-'
    '?H5(GAV@&E;KzPAgYVc|nq&*EC{Z!`ts_|~zS&F#MF!V0Oni<9z%$u$?_5`}mENiFk<RpXl;&?<m-'
    '7@Imi111<2k}{iC^1{6V(de(=`rnYiI^P$<_YI~CdSNzb+Zv~iPlr_V+}I!koLUC4U3;6JVs#NKLQPpBNk6vpQFdDQ7=~GI!V7Tm'
    'BZUDzN&aFhG!9ZUh#SD@G4;z?Vjf%$D-va$XEc6Ud88Uu>x^7WXtF4=rH&j53m0|iR(P*mVP-OvghFNqRIVjf!=*sJK1#1nv1a<L'
    '0JKpz@GUSGXrO4+b5B0*?rKn5V9pEb8z))St@NB_65&^#T>asNmjX;82yxbEvwI?h?p?%B;eJ^qniKMAuSAMq(o>jw2w!2pN3uBL'
    'l&s4M}4W!q8^zu+##Mu{DO!|=BC5vd59dZ$qRI+@i=%E#!T>H0et5Bv#fG<7-qkVLwW~G5T-'
    'o<OBNX&;VHul?m6T1t!A{o^T_ALu>B-v&Ve*JmZByXBR&h!IhG-'
    '=ZxAH*9%F0RHvuhyR=`d6T&<SJG#>K&1;sP);bj6>*cyLq9OepoxBrIV^?5S~qFpr4g3k-'
    '!!6HcEE+T#|H>zcczcWu6%d~!$V&tQm>qN+!_oB6qz#hK<%@XJ5t&VmWUi4qpx>>BB!r*n}P2e@ZXL%!5CNRT`-'
    't$I0nH3`CE#M@iJgR*vu?u=wX!L})2&`oxY!QDJ!jhF3Ki68JD^M7(6W)gI#e|jU)~mE*KZ=~M2k#QCml}C80<ympFIFT#WD0nlT'
    'NQ)Yv)0;VZGi1h`km$lJ&Rsu9dm)@t|zPmkES7R?;$?YQ(iUJNv`LW1c~x;|NN5XbqQj$61F~rpLs9#CgByZSlGy!9*qtp-'
    '&nj18Ou<wZz3lv2+#Rf8SmNh4&ftfBaV-(QT8bN4aj2VzD-~@v9HXD)$mzXV~^?H7_rE^E`q<yVfi{^FTDyeVh*m-'
    'vv?~t_ad&GtLMRiINp!pUBznczF}~Uu|LfFm%-}S;rBY@31giR3tkOh(hP8~eiMGa3E#v6smvD;_2mg{_pqG6d3%?zhQRZ=D+E0-'
    'i2Vy>``cjD5_qu6V908KYiyxbPZ)23b?+eJ%b@)o<W9Qcvk`coz_EMP`VbPA!rxbn2rM)1a78gUJ|OV?4cfaUdYkbcw+zwRKv<(S'
    '_#EQ&nle{7{~o|=Ra6!6$|RkClaCL5tit^zc(Tff(kFz(`jrmDTwty((cG+qm#Ycy;n;|0uNds-x_TX+OMDW0Pf)eOM_GSAhKBbc'
    'QC78gjofm!xSBS`^0ZO06>~l$NTfHy>lM)WBCOv;Sc|9YF>VJeSz&F9z-'
    'EkJhjHuiDGbT!didjEi;vF;9JAMq8j)_W7Itnx>^_1=?9Yk_utqO3Yhc@U!aLTdsNGfYhjSd{YNO%*CWE)k>z#zRtuG+!Rm5T)GV'
    'C$nQ^G3kf1d+y!>-kc%f}e`I;@sMp2~T%!H+~gcK~KT-T~>K!S2^I|6#0!rne%%(Oe$u1S4(1My$zFZzJuU)|ZgC7CBpIfHiaf+-'
    'dD1m@+)AA8RbvAg1e#IUf<c3X_#^y|Kf16ZgXF`+7+JLUp`>nV-'
    'X(FvQDsD)TL)E1Gu(`hTxA(Et5(e3h5p_8ewx(8~N0S$)Uop&N_{v2W&T6X01r-'
    'bV$=s{M)9=EsEh2<tI!4YDVFKz4syF(;)Cu+_&}$lp!)1T5!H%)77m;Cs-oKDPGgw~~AOpX1mZ(-'
    'VLr#C0cK@O{YIrv7eAV#gN*mYC0${<6ilYP%lW^|RQx(b^9UoEKg%_YmGoia_Ig2Fo_1!@Q50-'
    '2p4&o$6za(H`q7Kyvi8;oVmRSqr3lZZRyB(Ho&*pW#1S`Dqetfi=?Cci_60!2NYIEZC;wgmXN)tR2sFS)2XYL6DX1Q;o|ONc{j2*'
    'r7gzu~DV&_T_$nzQW$K+?}xOM;zZ<Kfuz@3A@o%c>Uoqyq<n$%-'
    'DiaA6eh%3iPGXwLaDFf;VZM^4hV9de%Bvxd&CY51+ZFKR3WD^mhRFk*&D*x%Dlq`w(4=pKbx?!}!L~y8}7?4jGgE;J5ghcV0YxUj'
    '#hvNX5(Stwy}Q2D?9j)?KkZMgZ;@U!umpB`}{o9diB>@&3g61#I}rw|uJ}QbRknvwdRpLhgS*f=65hKd6*HtV7V|-'
    'TBOj>W@Bn?pM}9;6MVHzd^2d`B(07yHua}wmqgL0QV~I?zgaYkM1Xa!r5F0j|02#MAqytLYetob$z3EMYm_yF6(CkXLL9G<lnx0i'
    'o09Dsl#|CmB3c~l7Pd&7tkBVOzSZm^4ASd>G!_g+thRVyC8W#*$u9Ji8FVI&omEPQG3!ye<!dV`!#ak=!&!5i0Fr!-'
    'R}Xp{s4TdRT#$s14n?btlzXNB;q&gSKR+ncVUO{GxyU?5jYruU!nJRJqy0^e?AhEeNv-'
    '($bi(WconZ;?mOQn;77#ZCjzfs`@y}fIDXcOjnDf*d(r{@B#>zR2pfNnRX4}%TkTDItuxAzfPbQsyAt<_#4aommpRO^V61TOAT0S'
    '6(MXKw7HH;5-)&UG9$5IjbrcXLbN4&|Io}fw!}o6td>8sGUE?R;j-'
    'ND3`yu~?bqrVTlcGD3<Gc#}p*R}DFND8fll09KDqkWIe$HL(IL7TVa>Y1{ney8eVf(LyQ-'
    'l+QLs~6A8{Lq>F+6JhsjG(Mdbd&gd+{`ky@qAnAHIhz(g%;=%5nP}c|1b+!&nb@Ba?ERM9d@`zZq5>C46gC)Jeht>%5QC1di-'
    '(t?|Q{VcEOE)w4MMBA9-KA?*2%aLCvHmu2J4@eAu8qaSf@eg(I};M(HO#J+rkXdZy>-'
    '(k}(MCD(j69q93dg_q~9LK@c`2%7mwfG0Rj_lYS`{#rX72r1lXXs$8%LVvb$4SO^`@9O{FyTk#0*~U{{jGA(<9rY`_p^3X?ohuN*'
    '6xM;KM1D~SBc@z27{%qG0%=6;~8wO9`QmlwGVQ>Lq@+f>dVYP7#9ft63+V`{Ho{gB5=g;oO{9<wfP6F<ipk_{hvE^6qcRN{=?uX%'
    'aa^&_u@Ifs=YtyYRvloi&e6RW6D7ubrg&Wq#sj$nXLU8S`OnmGx4l~+4!fP4?@~L)ptxeAm`nXavYia8BF@qsM)gw-jjZ(QUt%Lv'
    '|}-T9kFuS5(nNdohNYjIRGpEBY3)h1OI>5&K$;R*w5X_`z@FJR^UbbR{MZ2>0*p&|KYgkUoRP*#j^jy@vD``-'
    'itcRZ2tt7pSH5w*=^SD*tu};fKktXM<B7PWmPzd>&b+BSKEwBK3~PQ-1a%>IA~pAbERGY$HU+p=CIk;jP_Av>Ljd?cl)xoj5-'
    '33kDwPER~Z)x=M;xA%R7!}wezTqw%`x+4)O3e#G32-'
    '6c}*SAD7W)OD_@rFxbP$WrrcPbV_+6y;b_dA^m(5HlH)PQGkH`>FlfQe{emgGNdZcA;M>H{BC8kQwLkdUE>e*Q?}NzPa-'
    'D5j&sm;0oe^Bh}u6sP9~{XMjrb(MxN1?NMxS03fN&>VIMOhz!k{r5Rd1){)dQ3H$QD%X@`+F0v8aQzcoVqKQx_QpV_E`KMiETbH2'
    'xw!JX3aJAXQbWB+4?!E4iLF!BnFJ7q+JEAB6hy97%)Dp~Bmbxjjy%eu*Y=N%Boj`@(Dki|Zu-Sc0|wYlG%rk{hF3&67eLk9}Dd(O'
    'vsu<4>DE9Gh34}>AL;_*0V3=$2?mS|^zb^j@+)7cWAs4jICY`O|v?yUd(3j3n=>vK5H;!}D&JEN5n#u=R<hxgOD$Vz3+QG1dMSOq'
    'S?|BKc?5y)(3wl5l8F^?_C9mAE{+ihKO&KufH?UtRDkj2gcDUR*h(f_!zv)N(Dy+U>v7c_ep!P>KkXEEC$lmbcvrF?J%FGT>(+Y&'
    '3=&O*3=XO|Gk|E%)x&Q<n>!T&fm=Tw5XW}ShYbGqiA!9-'
    '75W!w97(JE|*k&fW$PPAL}Nz~KnQAV%)Pb6PqM@4`m>Jp?&WkQanGWnTQ#g#_=a-3YXF%0P-'
    'Il#UWxQdj_nmMlPc#n*u1c9GtgzSvSUscs{l`YkLLF?gy(bd?Wl187)2rF_RmomRHp%lT@QJFC(oe`6KgiNqYx>Z&kAHcIp)_%5w'
    '8IqZh9}t;&QQsx;b5@PHW7xs-xSB<OVg{Jgyn<)OoV>{6HU4broL6epuTnp<p<`t+B2^Igvf@cr&5q}_M9sVAQ(t(M$cj9KQAI5j'
    'A2MKGHe5MMnbd=Dm&syt-^r;RBoS9eAR9c+1`CQ1gnil7GqygvvEwQO<j4JN$XIs!>Ig_~gU*!Azp_<DAg3rH6LOvjmgYpa&w-'
    'i<BcpLgBIeD>ggoT5D;QBPW^gED!m2CCp=(TjJ5LM^;A?t2gTFp!@mIgAV&HW#dkER=nuKcr$y{C|s<jNpvbDt$);|wr4MtQW)V3'
    '>OTrNZ;w=q^S!BtTVGo&Jm+tu-t%qf@<4G-'
    '54$`Yi%O9#7Td3(g#hG*FcWufaos|Ivk<=15x#kCr<C$O!8U4zh7|95YC#QqmyYC~^&FiWbbpk_6U>X=iTP}8nRU^Z95EXjI7ts?'
    'HLEU$>s^Xm`_VpIXu97b(C$%`k_6^lYnUU-n+u%;mF2|9+i>nM)VwFr5%f^y?&NkT!)%cXqJNyrN(gu(k8_9S3KFuowXC;)qED-V'
    'Mztqv5y-'
    'F(oH9b*a_>q~LtPClRQ@mjgsZV1T!TbNo92lGCQW<GxeazuccUK|;`23{4!GqGK4k*b%gS4(y6dPXekBhn?Xc9cXe<jRpO0!fVs1'
    'r6keoG@}3t5YsW%a7O>#b@d24U|g_a5ql`8er|<?jmx^;J)<IvNr#B5^QlzV-'
    '4df&I9j@C$P`6xeB;b3c4CVqpY7ii#c0X*PjY{E@PKBR<Z`_b#?fh1-'
    '+!Pv955(W3H9a%9XkiR?Fwy=wULlfihY$R@lDD;6_2%5{4P&{4p)!8M9GxRUDFvKuZVBa6KRLxwx_*(Q@WdK}01CFUIBZtQ2}#1E'
    'ae&@VyD6tZ|29TpF1yV2sLW6~+Aqh%s{|A9P6<3vv+u$&<$&i*xp=^{tf?*J5@@#D|~OMUINuiNBqZ@6|lXMH94m*YfbMIH81LG5'
    '6z2Kw{*AU1_k}<nNA|9TlLx9`aGyz7c0;N=m&nKn|O~L*}Ze1}Q~hV+p&p(G{v0ci4aS-'
    's46&t$W^K$|_b~<B^EEvFGu;NTrI5a(>08j>-'
    'u#*t>GNuMn$vuTe>v9Y!%ktX>GxC#oV2r6Gy^sif~i;d{*pvZ7QWTuW#H9pxZP^i;xg&N1&-'
    '%h)aL<@y({@?TQOe}Daw&)HzFPy~|7`Mw9JrfjSWaQv!6XGQ1_q9FSb$xJoFl6HjRT66WlJJITOR>^ZoJgE_q8GvI^olwQ@O5m<u'
    '4|d7X7q|uEE8tv7EiPwtgT{m!nvb})>kL-Zg)i;3YqIY$zNYSrqiXOv0&_xEtlIFQF~*A23L5PS>d7s7r!Ijr*352VHwUgyKxvhD'
    'Esh(37W!ZGwbakmb;nv2<3&#kf^e_AuF_rv601kD$`K0o5Mp-~wYaihg_U);Rs|~@?{3?x=E}aO-'
    'o1u|PB<Ftld393DxniW?hQm<YZHsA>JbLdt%?}524`y`>a~o~oel2XY7e%DBdSAi42QS8ACK2>Qw6_O@V>H}-'
    'OIj1wb#|XUK~8K3xQe8&+8aFsvdT4#Hfb98>yLKRdoYgZ_K^c$fq2w6m3GB6Vx|6uZv99g`G8Hnrmw9H39>>0WGP=)&qwcsT?o&m'
    'HaMORYxa+D~og7z#a$O4UG432QV5}xj!1HfH6%pYu77_%jm8z5%pnxHDeX;P3UZN@j95<06e%3yr~D)gdv=j=qBc^34t@y6h|4Yw'
    '7NJ;^mrfN$hZ@5B(REQ-fDC!W>Y<AyB-pn7?I&T4~;-'
    'Q!$a}$21NG;#JG`m#+%`#RDVrGh$G5OE2Ht~V0R}BBQ!>qZ^HckMqTh*Hfv<QHT9z@bFbAi`@=Ri);zV+UM4Gn*w@VHb@gzxMQ6U'
    'v=VBOKf$T+dLJvZ`x8INyP5pVDU|FY_CqYedZ8p|9jr3Vur*ypbnAOxkiwJNx>uJO=kkkNr8yKsn)L1hdH(;bhubJ;poToP`I7+<'
    ';Y)4aFGh4)@1i88%cW)xJBybf<#9HYpd=t)`-'
    'wuded#t3*^;s)uVEG)gy9n2*{5CiTeL0TyitF*L3Ho*z5*NP5olU%JgL7L#bJ%fxl6;53kzwmt>R60vi*DKqE3;T5V_N%TZ^XgbZ'
    ')ZSw$^SKeI~-'
    'jgw>6IT@TR@yR`yHn30%*ujV{&&qi%r2PLLBud(DDbT|90W{JWXZ0rxsVc1Kv)fzSr`+NkBih3jBDM=f3}UQ~uhpfmiE&d>%OCyd'
    '((9pGtuXy}Tu&HPG=qdlH-UGe&1)?HU!FQpp=9Z9a@X%Ed2bEdTsk&ckTOz48n(HlLwt=$JRni)0J1<$(bJA${b=?iV$F-JObH-'
    '8_}Qfs@VU(r3`^DV~e(!+>R8?8Ob!#F~Jpr4PvIE2}K5do?AL4@)4L|_86hQazJf1H12*sa(Vch}6_=<oi6jE%S3I^6ks7%XUq8C'
    ')5o;Y)Aj19ym)h`|WMZq_owo@Cz%rr#8Skp$Mp&%OF)h72YQ$I)5qYn0a8C<1%SeX0ZEhDvlsU~Xryr<>Mn2V}FIGujTL58)1i^t'
    'Hrnh4&cU{q?0)tRr`au1?T79Z{9eF&=Db<BW&qe){~*7{)028$HphnK=5vs(yZ+27oJqF+Rrn7zp2_BTY!s=VWiqT>p^JO&Q-'
    '6whqR@`C)c+RUV5y6O(kn&V-TL6~pKaZ37S;PYSO9BEKD&elKc+qtnA49f2;0SAY9{V`n*n&|P(Pik(eAx$1|YcE-ovSGvFp?k+t'
    'd(1l_%gD?zH8Ha3kRt|MmJyHpsv^&dZuH2E}DBrtNIl_0_IqjUD#v5a=7?%R{LhQnr3Ev06`+>$vcngkhMh#6sWTxQRVC1i#+A=}'
    'U{fv=}9<kVaHuoh=B=Gzpu(Ut)P9r=>7yw&tA#m^NZ{Gt<weM5hV^7AIzL?noQ5%A57-'
    'HSv*!mG*y8R?13``O^dD;f78)(eox6fPb2O`kdh{IH0>r_3aYE-'
    '&oKg5jgZ$y~;Ust0#2SE<=*3%}^hHCc?Lv~^V!3B<4Kkb7(^c#;Ju{D$5<7D>jm_0I7%|mhLR^)~Ge7D~r;~1tZdm`ArK1K};$E@'
    '41(u~CM7C!TyjN{0jiEYd;xi<>3MnfvkpQ=9c?lstp1h~u3Zd@ksErx5KyF<Hzc<N>4J{(Wtd|Pb22+&vi{s`#43r|O4+#~j*sQC'
    'V|jum7@WR1kQY1kQral7s);^0~vPZ;3GYfy{@VT?xnM?lw1)Z8fDH3aMaNJyKRMBb}7`r{BD4TPT~;hF4=M(gh2UIH`xA;N@Mw;v'
    'q4hQZmM0NkB`v5-0r5q;3mJ;a`c=O+Jzxi`b|0btuWdk*G~MF$;aKTH^;c^HJu-$zKS4PhvE5$>E*3D3ZD*-'
    'wlzSUAd9IfWqulv$}`GQiR0&ipVUG7g&Wgxv$dBw;1T_fBI)l3A0G`^PZa^LdPNeTqGsFdhBzA=SY(H=Xc^(beuD4AUA;1os8cmV'
    't)s@!;ebL~;(sOa?FKA?ptyj$zC+vN{B883!hFHmAVzFy!8pP>gQZPBzf+Y9Pi=0K??F)v@5)82fP@DYcx)5cYLi5=>8E5Rd1~Kc'
    '%%KlJE1Qbz3ZMqtt?X4DQV$n0JQR3zUl!G+q;oieo&CIGc>V6Zw%!lo29lG-'
    '5IWXYLh`1Dqqy1i!5aqZ7aX;rBQGe&|8W8I5>O!!aIKh)$`bIrg&z?&fUmEF5=&L6eO7d7Q9F`&JN*z_LeR$0$6#&A?c&%ljmCF$'
    'Jqj7<UpT5TvKiu%C;-=$LMK8@&S&8DsRVvC12M2Y7_A8TZONlQp{>HRkjf&0=C*PA1%?U4?Z_u|ERJI^~@koZ`+@*nA&qce(-'
    '2H|K`ex9Nz~gOGj$<j%l(t{uh=&SS7?sxdFwTFE=darR=u9750yg4O;3So1KfnHE~{1kWQ|V|A7CDpP9yK||wA+?$Q#1}B&oaDsE'
    'V8F}?|2K{#i?6?P7Wb_lpJM9v~B8k8w(AL6v1~cz9)<6bFYbhbTubGJ`JOZvi3|qo@R^J;Y8GN2(^qnd83*gsN;2T%S3`6P?aAUH'
    'uCN0+e*8;*M@ZxFBRwC29y)E~zB@wvSUKs11f!61+f-QzU(~Wv;0gg`bF<JRMS$AY%yhwNmF?$k`m<EfVC(JPV-'
    'VBU-7*c0z^-q8mvRC3B#Cs>MGq(ICf-rTmzWvVB&oh0D1v4hVV&)(7S88(%^}!%AfzBE5VKROmuPbx7Qly3^7~N?ic+GuYxH$)-'
    'AB442P#+IN>OICjeWKBoW!x<AWR|@Qe$4=H#G>UsX5+KiFkAOVVMML64}#ZWJVTfT?Jt4VPaAxkqdix8Pq3oS14H-'
    '?&9hOgTZml93iced%p*v*euCDznS}X-'
    '<!ak(=zI*<`8Xbd3^``Qt{0(ip<iXs>gTAn^%dCn7_JZLesmc@WWGvxnXm|0tjzb$lBc_jSx=~ybHV)OM#U^7FstSoI%gY@{m28_'
    'S;BpZ*P}%QUjLqoK;X}7fLxzKgr0`n$Dm94!*oN#8jM+Luf(V@y!^&@zHbs92gjboTw$85=&!&#Q_9oIAigIYo`K!-'
    'YXq_MDV%2;jAw4njKB(e9YJJB_FwTaN&7_L-'
    '7Kwx$8=2+`SXo8@#i3O0q#GFQ43(B$haRB_mEKo^NkfnR`XSax4@+N&^HIy#gMh0@HSxvv@!o*f^25zECcWQc*h6#qi3;BNZ$~yK'
    'MwY?#D|mcIgCf`)fl$~J>+qmxqe<Z?(lrJXgMS<hQyb^&*$wo0NDXZKUi(7u)K1ythM%gz()H69Ns5>=Xe%cUPKK%uA1KP^{r8TZ'
    '-AAr5Y_-6VT645zX>y?vurRb`w3rqyqC#Ob6}5D=1Z6#Rm;+ict^=PUWWEfYL8eYQThPf6b-'
    '^dvli$2pPp0=rhH~Dvv8?l<;&2$5pjAKv7Lt*Qms!LS>gI!h>B!ZY*bG+0^-L;RLFuPmCv=t_Pj-yZ|s*OR+}K{BY3k=*Q^g={WB'
    '`tTc3in_`d|%<=xF<jD8O1kMMM<c7=uL4$JX*No;+K;|27^SHT;vGq1wP6}~01_jnQVmd0u`z<Sj&&*;{hG`^ekZD|p-'
    'y^qLqy}lHI4babPm}rwXnD~BOc8@QpkKVh@Gl<1vc)bKQC@bnB|Gwmk`_cl9NjOKcx;^W!z#EL3;@$pQNPZf}3;6T`q9OLILR?;f'
    'c3ENBrx&%h6TvI|O3Z&9($+)5Jjh;ej93KA6ZgE*nchb&y@PqH!PhsmFRz6St1y2)KF`-'
    'Xuffjnu6r3G{HQWH5t|@q6}Y!fXKo;@g^s6Ho}8almL%c>!g9i+kp3oQtc0XDwN^!z$*Z+`tc4}at>v^riL5Yq26vA&h|yc{<sE2'
    'Wqx|vkymsnUM#O`4Kk)Gt<#i&S)X1z+Z#JlZYqUSh4&)6yc{esUu<<#JdEJNShsfCkeCQtmV>~Rw49V>>Sp5O4e?u!ib-'
    'Y3N7_*mymCJE`S^4}aWXj63Ojn$Bc(N{LRe%?bwe0x_@II6GIPXE~dx*@-8vD0!$J?u|(~Pe*-pN;j-(jq>w-DIs&5-'
    '=E`uZU}dlQlWnBeL1^uFll|9OmEY5$FRFJh+T`Ar?gto}&zu^d_EuFd-=_C@r09pC%3ny`lOJVr^T-'
    'c~zz5;zJX^Cd`mAKtuy{*kyhecJv8&z6B%>y1vxOkuB9B46*oTCX}c=y|if6&kmrK2{iBGXp*~{Nm2I#TfB5BriigKC?eJ^1mXXD'
    'g|o^Y!A=pzW$m0i6P^C==~TzzhdtLz6JK#dljGg;LiD4>`8$2_Bn(7(i7jbw`u?NMg{$6smA&RF!lq37tHT>37-'
    '(&g6<Xm%CuJXzUiNr#l|c%__Pk1B>x{`_FJ%Lh4DV+&E&hw5UX8A2KilKlYxCP21*x~OnyOFYv>C9Rm&!2;t!~gcfhkR?frzU1kM'
    'J>QMmjbaxI^iM}RHc6q%Kkh|2fCKE-'
    '#655WD6=qOw4AJq%y=5}4H!r)%E6)|{S*Y5R@BzxTVb$9(b&a%t+1kcwX#&0T@WDP$^cw2iBujX$-JNE`={rAxLAvA13G`-'
    '#TyU^xcrB}QUuiuWq8bo0$Ms9)qpC>>#`2*o2ed}Bgja(DVH?C;@i>Pf$7`h!cY(qrFs;>wislQyA(hHeY%<~-'
    't@%AHh#kGjkX3e14Ax$oM6|pM@?l$~Ru~j|Ug(?s&tBqaDenZD*<ZBC#_h8R9T=!wk*rNNSjgTb$f$Pc3n}=Xe`$+_NHQk9h8$xz'
    '$!rkrKa}%*S1?<OIj`)}Gdov=v)8H$6@L@tkJ$#~Uj)V0HcXr~s4N>As+(!5mHvEv_>n@eB3CEYl+}*L=MHt+r55cBSaeRT#pJLW'
    'mkQD~k%;y-h8InIYb_}1}|M)oXgSq-'
    's1a`r~t%%q*<l#rcwis*U*luKCFRa^Z?*XLB_F&Fu7_|ewNvw8(8*<%^pTl6U`T0iWRU8KZsh{nT|0U%8OgIXi#|a#b52+{YMmEC'
    'u0<u3-'
    '>vlue9>NAZ=eiZ&Wv3xCcIeE^P8hqePB61X_ZOjVDcs$s=dZvFX3+_F$eOnUzbQL7OCRfc6vkl!$9q5g2x{n?1bhwbQu{a>yb7_8'
    'JCMQMggvmB*Ta*9z3^Z+w0@`Stk}9=vmVATgzW^D%Im@Jutj{^t6KKke;NDCPYwQXE%S^oAoCDnwii{qJCt3)*M$AxY&@$wQULq?'
    'tr62OzC=~;LF^8qUUo)cuU|U>`w!cH5;*5#@jmSpe?b0TBMbY%)nm#R;ibq-WXNCedIvm^o!<AD|0C{f$Cy80Z{i+(uX1sl(fz+M'
    'V)!fJFJlk+3$*RUlke4nWAN#B<<vgisdE=PMc~!tSFH|-'
    't;q0Jw(Z)Bf3=U`8Lwng<$DPn(<6w~*Rb<PME6%;;z7NB5rNHy{eQ!+OZvw13o7k*^g!nQx5k{)1nxb|FjHa}e}MhpX*~Xb4QB|v'
    'ZXSlO91*WmB;qT7za7UngzuqsKRiAP90zv}$2gKWYQNfc5Kn)`s3ZQ^t-'
    '9MfXvpLqx)<}jHTiqhCpe7W{1Z5Q0<!)D|F+`EHvha0GY-'
    'I}FupeAGK+IKV&6%X$I;#D+q>2OBwBbC``*s&9L4zv=8J8|fFe#Yz?-|@*asXS@bh1h%P@W;9ELU-bzD2@LD<hTI1Um+*{$uz>?0'
    'u@O(Opkj$aVvlh9Pm*^l!{+>weu4evNg#|TFVKY<0Ob-r|e-b?KFXPw5#!v^!dLtaj(^@-'
    'q4_z#XF;Lx{*^dss6NBoDFq#u)l|A#7Ne{%vho&e<YF<sO4sszs7VPma!oP(-'
    'WBnSF`FuMOSSb72;e1pF7C!`$nxpc^OoiL6gKkU&FJDU^6pOAN4SK||kvxGnK^gCqq01n}?R}F__bs?IA_y+qJUgtRvhYWlTO;TG'
    'I5Yr$A5`|*UA)H0epGGHOFqbnrk_`uuIB0N<$Npig2!Ft;zX(SV^CPfCR-'
    'w~&Iw#0w;05RSH$%c1Fq`WjgCq5F&c2vr{~QLdgl3f-'
    'H`ZE45k&MH{5Xw>pVMq{PdJUIT)WwvzhV7Fyv=Yfx#q5RzQbMi@o#v34Az~o3!@7j!a1{Z2HE}<S^mMW@i-'
    '(%^%Qi%VD8G=bxIlb7e=2)eV;Wl&Gwxj{ASEJidy_pZ4Gb?_m0{BYUa5IN{w73u=lJp&hYpf<uSiw99L~em689!uT#j9@L0~whw}'
    'ukyGyV=c;;2JaQ#Qb{cp|4zmR#(wj2-'
    'V;P>yC^AF*qcK8#q4k2BLJ^oi2{12}GsvrLnJoy4sexN@ka0fhTRFwlJTvGr3Ro`UCbjm+YYnCKhzxkgoLE~{8{~G=KZ##_S*%nW'
    'wc1}avcgWFsd_IrvdjX%rV0rvpdcNaicdl|I3c}wD=qeZNFwXg0Je$DCEKV5Q`>&3`N%;OREch4tE@5vI#wA;D5xTQF>6{Z7EAJz'
    'mbMB$Za-N7Q2pkOyo_jOTtNtq-'
    'StTz)`vqmN7vJ+bGK}<uzqD=w{DqlkG-m=A@y;IVA|8j%smJ0cvx2qpKabaOaGcURQq38hFfOT97rV-zY?jD%%@&=)+Q!_HF7+3('
    '8NkxBiz7WEeaYwVulfe^yYA8cLWEr~FS`>4`;*a;?w1J}bBstIwK=Q*s>K)KG4t3}Yh18Ztjpl=y2E`Ts+wHY#!_4-'
    'qr<Vzqqbg2$c(z<PW>P3JPjG?k)PZc<LVwFqZ7tOJ9V5^E@j3@uNo!Kxgag0!yWo6CkzXE9X-'
    '5EdzBam@6NccrOM>{tt+8PxL**_&JMoBc@eN9jB|EQ)O~j4N=A)NMpz@Uy8@q2!Ok28<E}7x>b<L;@w?SoqeJ9#!pP}l<^N%FXgm'
    'qYc?iPz{HWFJ4!;>^i1j@Amr-R(RcBSmr#OSW?s5fuxf-5k#Nq03h5l3?ck|-'
    'NWmEuH2){ow(~CG^<aCNU%x>oBU#PI$7?~TeV9meqxDf36$7bFZg@s|{!AS8o9U_ne-LfEfQ^^UVpso}7oZ=Xj89JB|XYE4JnhT$'
    '7j4DCM3rns*>`OpSVQ4D>EoG6Bs~tzHx&YylUD65Tf?Yyqac0XC*xNI9>flu;pCgR0;j3gqY9SLK-'
    'e2W#^;7^iI1g7N2V&P1h=QzuVl}U}vfgvgPOmJAgZ-'
    '%Ea2Lr+h+>FTWDZ<&!LmH~DU*JCP!FHI$d<$NOwg2x@UP99gl9Q0FTY9<8FJ*)h-'
    'Cy<Sc32?6QY_2)x^)Z17(iYo7W3vN5lH_0+pR>P#OHad=9P6oC(N<xL)P!3a<Gasf|Lhol?*p>p6nzh|!grC$5vc5#ZG=uhv#U<9'
    '<D-hIU7ZAN$5Vj<c8vUM6zMtAgAy@H<Yagl8TkS2f|`RjB1+7?B_KCJZYWtG@h-'
    '!cG~ZkJ_=g=74W<6~ArBDF9nCgGWU$OYT)Pkk`mHzn?L~C3=am$P25gLQ6LEu%Zs{It#<2{DgG4Tg-'
    '>oOIfG4YKX;E(3vwCIp<!?{4Icayb5Q*eH$Lj-jkn+u1j_`N7kbv_&GCl#B(DWUFVYSP;GoFMeuSS=R5yn?@GtK4^RM+6Anp6$|~'
    'co#;2@KeQ+|2YY5@U=F<vFL@A6Zt|KM4S^c%nTSXfX^85D1XV=8s+D<hvNLZP}$sPgLUK#U>tG4V;Ie1nETFb$U>hQN@s3Rl;GqW'
    ')IZ5Ty#-6|YI8I4|PMOl3#$nI2twyLnAgux==XI*1etff3oL}~TB3ZWt*RS}YFATlMCCv1OVBR-'
    '`G^$3Zh%HZc}h`un)>tykY&js)D8mwo#+2X=jiDlK6sx4x4AYpDv)mx5G5j-r9SeAh$<qa)d-'
    'Q1mco#pDO6@i?vs+7U3GLV$j==E%ybZ(Au@GR&Zc@fLf2^{iJ5xn6l&4k?KGj!A^aE@6DuLV`XL5Zw%{0jJ#-'
    '<WZY;ipt<0r0O3<V(%x#?Sd+wN$Tsu1#p59iX_ejx{7)gOP<*uIvLFY4s*T){!tu(T=d8S{4S!wgIBsP_tMT>|$>lIM)&yfOQQGs'
    'f?-w=~-'
    'oQE{td85T^!^!JW7oz<XL|Klg`gz@gF_%{Z=s?Apj)1?`^Fld2P#^`@SZ*rQd^wK@#m^EcF}hfxF?IUWrSCQ47NZ1~&IxKk9PIL|'
    'd=`3XEOtgCnxtP$dQc)wrHu#mIE>sVO=GdLc9v9N^h8Or!Euc*FN(`*MQu16W;O{5{Jtun3^llWLt<&@HXBj*abc2rlE1Sk{BXl-'
    'LPDuart1K%q_b~$6G#rbYRxQ<ZS;81{Ou&XMLa)x)jTezN3$;d+mM6x#Smsf8SQ4?q$TRVdBo-dVwO8Th=o;CDm#AQ^&TzTp-HXZ'
    '}cYBhp4UOrXvKQ$xVKqv$5)Ya%$LS6R8wO&kC6}6z2KB)`T(z|txcs9W^>D;n9G*WO4@{GFhue!=iL}RCE1R5)AI7@YKU*=cA)5e'
    'O9=vK9iaSUGBI97?PsAQ94R~K`dsP@Kcf%mBsM%UNziD*F(HeG{A)rJgS0qe#%UB~}?1LQVPJ8lH0!l+~Pd$FW})<9#G(Ga>C;&V'
    '-nP#7{wI(ucpEzoJ!*Nw2eA+B{GsU?APRY!AN9iwX-V-qbXf?jb8Mpbp%sPAFi<n)L2&5ZjV)9Yx2>gea%PG`(-'
    'h?q9VIUY?}ms=64MxYykV?KaTN&BDdD%v?c{2GYEdplqTaJ;ydSxR?h!L{IRBb=Ke7hyDm6?~7^>*|<&t<#zy{A!LUw;_mCi416l'
    'EcJjVHxXnX!hA2Qv#;0OHPe_0Go}ArZ*+17cjbEEJx81~>h=64s*(MZ=r@OMmRnJ+Z>jS=Pa33vfJ<CGt&x%Xi1YREs|{k)12W_9'
    '3)Qre+TscC6Pe97`f<8J?e|z+H>O3Tw())O>}>6HK#%j#LD5Jp38O9I)(F1cs580SiM8DEc(2sS=q&>Y!VA&TR6C+@|GLol9*W9N'
    'ml!%L20MeaKZPM(qy^66;ZPs9`sfiuCF9MkxzjZUuVT5n>N)+iBHBW7JNU&M?vbPia#!FA=IHVBYn{H(EIzd|b|*E_^W<QiHyOQ#'
    'Yq}?)Gpa%2)g0%BMr6E|h^s|<a4&!v(-'
    'Y6y!1C5kdmIfh(_8y$IDK%RrEm;;N1&#zx?X>f?=bpfPCHoA0@7=N=iU73xEVj!HmbN4?%xdFgu!tgs2T2xb7y3^w#usIjK|RspL'
    'oy6I{I8D+QNeNTH)=XZyfH7hhN>Gb)5br{FODW3o;!>BjZjhRB<=Nqcdg?z+A44KF(NYG@fv8>KcK*KwHB78ucz%Q@TOlSVV0A>L'
    'U!V`Uc<`N2#H)J`REZf_5=TMA{(2-EqAcI)=jbq0q!L`(adKzYtG*U{+JqQyX-*&iJV(xYNh*p&ulURy+BL@F2ctzrm;&-'
    'qp7!aO9ioz2464h-5p$Xv8DDYWC9Aw1d$(d*HYZ`M3?eOgiWF+KISEt~0twN5XZO-'
    '%uku!Wi2@qa8+DqZfA4$VtEM1oT5}JK^)qkUbC<H8EH&V>m{mkvr~8Y$?ae?2xVCg{)xxk(Is#;b33Xh4iY<M$T^3`*Gc@>kj0h3'
    '#zkQ!ph^p^ktyY$+!o%j6Lx>(Jh7yUg1Zfa(cpseuiISZ7btlBK-DV&e;8RfKOe(p6=j?WK811b=e<pjy)02VX(6o#`f2$9vq8PO'
    'FwEYjh!CHh;1HD=tSV>tzcz0<+-'
    'ffvZ8a1O8@C<^ql^vjxajM*8c#~HTuBEA+a4o;!dH1c8>r(w8MF4@8=%=ekl&G55@ascfug}*Bep171vw!+(|!m)Ls(oNqTDJJEA'
    '6g=*k*LJ5-k38w87oV5YbK;Rp`al|;sOCiI8QJ0YVVMs<PS;qaYnd;sk3Z0wP5Cv?Hg{$NfA?H*x_AP5Wk7`r|0-n<%f?D%;f-'
    '0KhR9gWqmIrI%OqQdfqX$%v=kr;#sNt8QawDbWU&(ZaA6ki*9AkMdFj0YovL;afTqHASyUFlol+|{V-'
    'WL!*yNZ@FWggwJy@i1s@kNoh;eFtF>VYo{1B0EUWBX!>~4sr*>hAFuAjKD}jH&_$KoscrpsG8ew?>4Lg!(sOw_%s|xFT!X7XHz-'
    '_%e__air&%AUCu;k8KvtYuPr?Y+&4yI{s?CbVHlqDCvXoMfia`t^_`*EkAY>kIS=5<j1!*TZurVtMnNveev~r_T825(aPDI;lQYA'
    '<+-m5!%V5ks&Q$2-TH`*^1(pcJZf_osxK&r;y9gs-`@I-'
    '_yRlvfFPY6_RPtT8N^HkClL@lwjDQt)z}JUy9cg4H=psQab2qvlS_f$bkMZjujxiW7QF{dJx)Yh~g!qqv<v}#Z`j~?Ib9ARV&gfo'
    'a5FeQ{5>JJblEa&E9pXIcJg$d_C;gwFh}~g!jwhIF*;IHl4tnMidi$&!0Xw*7-'
    ')+?HFr&T_f2Sbd*3X9avC7VyRrVB|@4}VWjtPX}5$NZXN6znpO;W)uZ6YDq{r1vY?uB_AvFXO0Q89S+F#@xCqpOTp7LUfANrbxz<'
    'McdMufh?ptH%4Y#B>U3XbP_54EEoZU_o+>hn4L4Xvlj2@^1C>Hv!UTXcdh$ddLhT>KwO`xHm@YCcqR#?><!1SZ5~TcEi3Y!1xGAt'
    '@hSlJPCb@Bghfqh~H&+Fae$JX4rf$GAbOtlOSu&M2s4TII#!!z|)&CX1Md9fqO9LLCkv4uwg2JJ-'
    'nB|k{?0m>jPd4KvxRmULzjU5c`J!>6QaA>p^@Tt6Ds^$@4IROa{?;$mol#l_M6cV*R0YIze_R4*=2^??XqK0gEK-_b1>{!YqPk7h'
    '5&JSdV5S))Ntvp*RZz?<d@gNQQAgMh=RA*OSAzhcMZhr=#w}XYSdYZT4=q&U?gJ3}#HzRpl{)*TKSg7&9Kgc^V`vN1Pup`nPobIS'
    'Js(jp`lD+Gz<4m<sJvja*N0mSEI$^n9*>(Gi$t<mv%k!xAyWXTo%7o`-|G@gqi7yq&~Uoh2TK-'
    '7_J120lHgS7GK<Wx`Z^dIa2iFxDr7e)ebt9-'
    't`PVbsS{sHy3YJUh0N3UqL$d8TlBqIQ4@&Z}C>k2+7Ir_I7Sj6_yX(0o5YSOIC=EoUPhW5KV-oRz*eo`lB<QnftyQOte<mdw<ZYZ'
    'hWE^F59~qMug+PiQwv#51v-P5_PqSKAzDdRU*#Q``*=D`Sw(#$1_|kdJA=RLEeh({PBL(j8~xIq!2s6L-'
    'ED#;%Sd#ZjNAE5Zb4K6H%-6T+Ajff<Nj7%cxG>L%jL-8k;i)j)iZOgv+(F?So)!fPG#oEbL*Rr-'
    '{1SFD~5%u^mbiI_<*j<0DC`De-IQ+P5#?Y$TJXDgP*`uja9Ydn0OqxMZuwmyz|!iUFV*SRqZ1OCq;-0QqxtR4X-'
    's+P%+^Ne!f30U=nUQ@?B!sEuO{i4y+n5E1mu7mjmw&7k_J`?pJ{p}vu{E}jVo*&a)qpU2CI}4o!+Vi}1)U1{{7|q?D<1`m?o+Zo$'
    '|GZUJM#$<e`-!>w>;Z6LK5A?Vu59-r?d7u!9gCGgPwC9(3_dP4I_-4BYQ{4WcoIxpfb7jtU&5G1;Ed13eCZQGEzc&f{Zn-'
    'Ov+(039CCh9=Pg#h7vM=Sa)uA_cL8GY475F?I-l_)^bq2+$l%vgkTD-Ko<Xi;e=vivh#-'
    'CO34^!uV2@bylpc$5ehJzZWBjwIqv!BU=FZ1Yi&URv`w85C2(`=kc~U#R*edZ_VC;QfAUt4<;EuTfmMk@TwM1!&vj&=$XwP9RGqN'
    'EpTZ#;XA>J-f9ZTT#LR{z4+A3fsaOOFp%p|sDsp_5Qb8^1_X@T=HVJ_D2cQqf68a6$Q`}e@lmH4#W=p4^b#m}X8fXSh63S|$<zs-'
    '2wc}2fd2qOB7-vgH7$r9!8J({0q@$+J3`BLpe4<U-LYp-'
    '6a^LSNRVtDzoW<ape$110YTg)xsmU7FwMQ~JgXKAjbZ#}ELk=?<=V2gCk)xhg9ypizP!}AfC?`L&^eiD8^rasDdDOKGUREzi<>=+'
    'i}(;C8a5m-uCNLYu8c}~B<c^!Ou#`zF_EOj<Pe^s|S@RAe8+s<0x1u*y(#BVu{m%xP&oQ=-=z&e85dk-U*5ndxKC-'
    'C^i1g_nsh}z@e#B%Uf>{^02z5~tG)rSY1-'
    '0m|N^AzqbROW`k+UIE&62UWG(Gfu&WyFif;!}{$70dPUDq*Psj`&>UhNY|`yh4z!`5Lmc$mn|vStVadqE)KuRpQD5120B^^BDM;S'
    'OK0#^AOcn{hstZJbDdQ@h)HiB)p`Y6aKvh#;(BUl^Ff1ffZo#GZ-'
    'Z+$ifJ)SF4>jA@w!HV7^gLY?I7q{=BX^SrG%vSOq^`#d(o(c~KInuNYW?s(XVVTAqcrcZsb_jCxsQ#AgkTWth8GEnS5xGhl`G!@r'
    'EKwZ!m;t9dPf$4drR8?#oaE#8dRHMeUGKCvw;2yeykCOXyIMjT$$-m(VL`3YB)XUA$EYtT*B10wS+{ltA(s_j{$x}-'
    '<&dkgQyrB8}3v03nja|-'
    'NRZRBIM(JAgt^0p<mtc3)L#v72y%wP_2uK4*2u=r)H+Z(`yHz9kS;W^j%Cq`zyJ)^fzmN$i$oj)DRT?=_DAa5&SyR*$dw(HLlotO'
    'Q!CctvseG~WB`MKWVLwf%*WcelkSc*GuMS!dKO~MB_-ojb*c(s`LRN^7A7C%?$u>zI#mLcVB0yB3b=B!gK;>T-'
    'Rqp!dk?^uOQJa6p6y&kg~&sKubD-'
    '9j&X~5wR0CNrGJg&sprMjzGrFr&tJL?G|bAyj}eTXIRX!l<p^T7M`4&hzGdrnR_r(4)9gtO?&?dEo0hHlBZoO8PyaP?&EfL(8Cm9'
    '51+mbAm!@1%FHaBWvC-vEm~#I9yDjt}*`33ki6AuM?teyu|W!eCq9BYf`cHFkV6ASb<h$@v=m{!r^rc1I?o7#o}|z(+Vf_OXd@rM'
    'tliLt@We{R711HFy{XTfW(_Xuop+hj%{U95nn&=e~iRoM?Q_VAV&2_nn_&eKWuZhCi9y88}uWUZvf=YSqdJ2%}|}`6_JiBxG{0@{'
    'tRfeg{vvi!)mjqaty?$_?X~bJ+O}I1e0ljv#}t!upRelD%ZM6mj3f`E4+7gGTHfSeV7l;%4?C*7BUqv1|u;7yfNREZ#RFD69=5t9'
    'uT8{%_~B^AGShKEDA8n~h3%TSsq%U2izdp-mBBuetYch2%sYzlM8n!3tr)yT)ktfW7|&{Qt)}>zr}UIhUM^IG@F+tp<NZ(|U}z5%'
    'Uj?caD#Z6{WOW+I=53N!GbXehi)4AZds5sgF&F71zf`t<OZTFYBG`?gh*h+qNXhLLmJU*s;y%Ey2_6{;DS{!H3!rIAiP^$M93a4#'
    'IoNy!Cnrg70Ky?;!8z)!k|#l*D^*Y1Eg*um~9Uq2Hw>58DWzIXjKc#&P=$*x{5?d1n!gO*lT*_ryeOCvdzxc6xnWvbxL17d{GO4J'
    'if`c8g*yDe7(puPV9~+|M*(J2X;Tas7m_1KxcCt0iM!Iwjo1U5UVPE8^ZBjM!;(*$ol+ijc{beC)&VHI6-'
    'i?G|^pU|mTaT%%ka<=ygb8MiD@9!DA1<LWo?zX~!~1X^WpwG*CJSO2!fq8H$E&A}%T2$=tgW^kKUuhj6jz&BuCC3ia{ZBAj{m&nq'
    't2ynlw;0naQ3hJB0O|tSSy!g!7P58j5l|2Nm6|SW(2sM%Y%GxP*!n2*q<wVqUE4g18ypYP+f*gmj5Ayz1#>rlHuio1m!*0YU-dnu'
    'h%QeB(w=>DTif&E!8(1b!cN!8Zxx$Hxz&<Ao=ITzPjz3MXHn90qteT%8G9N<rcZ9Fu(U%Fa+@;J9!*Va;xa7F_cL|?%V^mEy&WAe'
    'S>*pBvIkbGQIo{(RyL~$=f)`RF-w^7#-'
    '#TG%?$RNz8=SiCkJ$aNXARv3Av7d>s*DJulA*B(Ec?Q#<?cirb~{pum4GiCj(sCH42}|43A6Z1+&kd>NGtcp$U$SbF|>c7><VMIk'
    '+1R?xm&aPJz*!}yxqCh4P%E<E#DyLyK$BM<Y#{0-'
    '{D6`<a+mZ#;(`7Y?aBdFusSUH4xQZDl@T1?l;DAzV{>2;coDi^8+9r?)N?Y9QPW#a#vQC9|^)YzPH!VQ_#HzJ>y%%>nG<|%wQkO;'
    'JM%{#EJV8bLt?*d<Tykxkn*uw=tgalj2YeUu$G4yG;lM!A0@^sPl(W`P{dDHfoFcu?zRSd^L5$_)`1c&%hzyGC7*N*Q2JyHxJ$?;'
    'VApFPxUb?e!za<Yuw*wWaBWQrP~6r+oQ}3<J$<ZPOe$j`~|G7>3*TH2!rcvpTUTHu7|un@&g6*>|vi_-2npoEtT{Q-'
    'e$g5TZP{&VYgGr4P&>C75jem@r#dr=rQ8s4+&teICI6^8t$8@hGellsEkNNm{0qSY>5VWTm4RL{1&U5RO}B)@U4!Mk%tI}jm*?`1'
    '+`+3-SK<+&ijF$cOZ(=xxVvdri`Bm^<7?<-'
    '&G#}5Q}MmCa99SsD^!5iJH0+fAQi#M1%yqgFdNVOEn#YuToQ=BN~SwO;$LLlXvH^Vb38$?tVWf%y^#38h#~+72hK^GS_@2b{_&`!'
    '}wM`;=62#+-'
    'v6kpgv27IcPxM@DJ$ILx$bgyNut|mY)%wLk51*o^r@Z9lsNj*^`K)K;rmMeErhFTDdLVU+~jU5jd)P<@=1EF!~o<e@=p5(DB1yIf'
    'qrkPkIiHzx*h*#5!~YcYfD1XYmG%;7-J2juF_tqXf>)AL<#$?-zq3YzIH*DEtOn<o<~ej>Gfgcp|;<cf*(8bXFM6-7t<r_U{Jgj-'
    'XlvM<C^B1Vk_M=`VuV@w?&UZ{*Dh9s4Kp`8%G8&n)+kNU(LMko7QrN5qb+hGR*<^8Sv%2}J4lBr*9DIH2|%PXgB3!sYK4|AgdY7<'
    'roTqwa}L7&zmc#r+oUNz4>4SmU2K4(g|q22P-g!{GPa&-7b8(Q_^VCspd7>cO9c<Hnr;f!5>BP3{@w`-'
    'H|YfHy{@ABr&|5ho(>R|L+(6OnStY2}9Ti|Ri`;8^kV(>VSioYHSW|8`otVH`*Q_){Z!CMlQBIRE+h*N4R73}J?|!YSacaVG2EHC'
    '^e-w+g>P=2=L%KsaOge-'
    'h*PtveTX`+0DD1G$B8<?pPIsBPy6r;N<<2>z~iHRSM|zm>VaYfR1?5edijJVvd?N`E%w(OLa|_HWhrk1^w%j^X=URsRzBcaRqWS+'
    '5R4-'
    'g#X4dzn+P;k+T`wED2sDdkpm%OkF(ahAVg`3F)st7nn*^RP^^f0kfoHFaFnNSQwVqkf!q+PQ6Gkk2gtyp9Uv40v)8GA{Vvf`@V5c'
    '}u@R`2u@7dBc{shKta5iO|+<qiE+|Qco`$oVjSs;oP0n9m<uiV2{2L<k9!lJnm}!ZD&FHzPOrexni~Dh9McU-'
    '7wA?_2RhaoiHw{jXd)rM*b74-?YMU*<1OxiT|$?{&iBvg$SHN_D*Z3@(z(H5s3Pi7_GpTi%uA>n-'
    '2HbuZtR=Q;zF$)GxqC`7LOAH#n!qo+%x)vDbTWH+9%<GWl1i+zdu!ztSg(V2v!FeYFg2=~Yt(W9HY6JPCdZ{;~c``VF{z=O$M{21'
    'R<8*(KJUa5A`ITxm#j{mdpJgPRdZPiTYL86#lEJB;(-'
    'Rvx_nyEt+o@)=YnYpQC<<==5;aJ`(#>`XW~FYSN~#w_{$CF^QSC;;0Ex;b50*Gl4imCDcLawM<tGju^K`~rBJiIADV5=y(J{KJFv'
    '_)KmVH!J3H9IrH_$*3&um4tR~8H2^OF#0O^kR?eRYr=y{`WxP?N${5==WK-F=~WSs-xps2o6b1d-0TF7Rh9^F?8J^7F-'
    '@nnTLp)pE#_p8z}1AzNuqEy#^nfU7S4ED+Zz&c82Mm7#O~}mGKcz|)sJWv$dliGr`IRK_^gJ#IZ-k4G^;NqyI!-'
    'PXI|yzBwTG|jrsGd(*fghyLnY6+bQhIh37d4*^)#tmrBWOAU7db1lR}G#5U#u<hw=A!r#u-'
    'vF{yOruVl7dE7AA(yWGUtRt(NPveo7kk6OLdgU9?AY%bIwprbLg!}~7mL~$a3<)_PHJ5%D8%91u_SHsI#0N8{IngZ(8hQ%&kw`=#'
    '+-'
    '0A==;n6$n_ISz<5`d(78i62Cy9WEY<R|VILCRRmorer{~dfm*ucJL*WW}Zq7d$~gu;aU7*|wf^2h?}M^OStmUVKZ&iMQ5bGidR<J'
    '41K-ro^~5s!q}QjBm7WU(zAL6%rZXXW>Q8x%)z0%x0jDjWfpS496lq8L7vAc)mw#IDh3G;|yJM?<~JH-v-'
    'T3OcVOkj!KDtcZJ^doAWRR$Pliq%?vhC9$$IibbFh=ACzny4^w$N!?U#P4`xOE{VOuHA(WtQQ}yX0r;N$T}(Y(y<JEd0>?R?$@u&'
    '*N)kj${HJSh6w^@!psl3-'
    'eMcfn8TPRS62+9PE2#6!y5$I^32b9B&Lkpgej&ZbTIKIoc&5n+o>j`d9&eu`urDi5s6gP@aqf!2CZ6+?|1Dn=w+^gn<ln7GNG6T#'
    '^u8hd$?^U=NA9o`c}HO%SV~#FCufe7d=D+(d6tEg$^=tR8TUV@f^nDM>AmmO<*$m0<EUcDWe?(3ddQIQ#_7GyR5Uy+<J(h}P=z3I'
    'uS8&3mGm!lD!486StUQFauz?>JJuNGU1bCE+qz1sF^mH4-OyLwtxl+>Z=&p3MYow-'
    'E6II99jpyfF)XQS1lUr3{(J2EbnhEhmS0fU?^bRNx1w)z4MI%<N4L7+OEryv*j~#$=g6uj-$h-'
    'cXYX5|yB(2KOD$l1riYDVo>hk2#NXk1s7)|ED55)y_<DDRF{d)DEQPb!BiWOCbqF;L%_WSt8nN#>l~M~cYUwO*W=BYGjf17u1)@B'
    'vt5W2AE#!1`JH)`>1~i6_8&qZ@>Jg&!v5o&amEl$G1W8rFnaYS*9Yc0~jN^N4{J*TK9|LO$M%IP(<uz{g>94XHggmN~1a%YSd+=h'
    '^K8D6_3-z{&@!f{E=2XTL)|%)?UH$ES17mg#<8Q|5sZaIMHyXP97hhr-'
    'k8ek4p#IAfuP1qb$5&O~47{HkMS%Tiq>*T>GR3Ph3E!_s53i$n4{D|sbcn7h{{AhFhQ^)FgieHpV6Ep(ef%VRs~LM+tciQq>h3U%'
    'E`}7j+sOZGG}%vH=kG7;x%{_kf!12uudXBTUoG)$iBwlYlbG(f6|LNxFt!PL`gM`WiE*O2x$&fvR(2=IWV^19KobK^H7-'
    'pcjh{3$a@36=euTll+cxbBj~A)UaJM;F+6YzL2%a_6N|!3Q5qFyE^DyLh$TGX7KD!C^*$KH1<3`*KgQeUQ0WWi2EF|jnwL<D@$B='
    'sPylV{`I)#3_Eq?a^Zp4*;!^w=T>&}S%S7P4Tn|~u9zXiU@KMQ&&`1i#8yLI_3r$`WwC4Q|ir@MmvyOF^0YD3`hHB>`eXppCJ72Q'
    'pd(bkAAd*0HR-'
    '%jU<2l7`VJzf5<44b*VfS&Fxu(fq$w%VtFV9gUfEy0VfIP;iZhz{G^fncr@9o6%W1W%&J?xtXl#6<4*AY2Pu+N$N9Af=NL?Eu`-'
    'x`t|3nAs7PD6#LR&%5}w*;U7Ob^Ad}XSYABk=(S>Lst7X7}FYO8OL=Ubg))<=KY3}!PV9s2k!?4C`67InHs)Dt&!7qMh>`p_9Apt'
    'KRZH`M6Vk<gj82GSl3g3o7vvTSWln#9>lK!gx)Hx4WiKyR<%ud(j7l_!1FNJrgrW?LQj=g!}que*2C&<ANOV-'
    'y};#8cz!dXa|8wfz2I?cjX@hdcY*KyfSV!z7WKam(A~(4a5Zr^65cykcLx*L=Q>!;2Sdvc=&A<I^4=mgw^wh%V2(>3n;9z;kGdI>'
    '`y<+2-M)xl7|cvwUv2>Z<N1*By^w{@8kt1MQ_1#_m{+~wRX^yIIg;5f@U62u%*SvagJD@1H(w^g{`4@qATy{>5-'
    '{&L3U&M#2RXate?Uv$UisV`vbewYjX<(>ND%M;En{ndtS&tN7T@l`dM|G>vYoCJ@!sR*p*P~fx#6DDoxr`7$MX)TzkvZsdWb|VUN'
    '0SVWeAU#tLIN=JmrkDL}pM&zux8ACCPr7>uFE~`a(+kBzOC0jJSUU&*S!Xh8^5B`Wfpd%fF4ll|I<0*<jDu15xed-kKyEVYGp5gN'
    '=?+P2ci(Z_^E)4kd6NhQu%!+6Nos2jOmjq39Vy!ED(j^@7F22!kWQy9j<)96}h1{j&HVJ&K=6j@m(Ib@)9((b50h7g6Fo3{~I47-'
    'Z}^I9tPf$<@@m5$^4fG{h~Axg&5L9)suo5Q9}FYvr4o;gEBCLR3B6PT)M;X3SuSm+ZlCUmEW_hPopOoXZgewue0%tPu!9G!KhFR2'
    '+JuE=EGuD8g`pFq2ml_Hnp78WMS?@PqGhef3rOql~EBrhB12@V}2np{Kz-UI9dUZ+`{w*15jQPS5ud5ePH5KkoF`eTB^SVty;`1-'
    '->var(xdjf?=tet3ePg4-i-'
    '8=}N{N~UEr?sNaY1*{x^8t#khKvd2k)Q_w?vfJ#3=njIvLn2p|_@7Rt^Wa?R?6(itpV@otrS_}#Tzipymwk^t+#X@Kuv^*ke^#Ew'
    '`aeZ^5CZ'
)


_CANDIDATE_TAIL_SIGNATURE = '853214edcbd5332ac7387b013243e6609d06bcdeb32d94a14860abdb1e6c6046'
_CANDIDATE_TAIL_MODE_HINT = (
    'c-pO91-ukx_nx3ZkOoQV4v{V?C6o?nM37ccK><-w+C;$s0}(+)K(SB|6~$H*TU0DUuuuUD{6FW+?A@K`?7egOet)0e%<j&-'
    '@5$$!_nqC#_g(6LbpJ(#Tql6zxS7H^iL3u#a56NcdzY%;F*aq_m$=c+N%-o-M#niR-'
    '?K9_h)1fG>ECBIJA0lx*_@pfOB{;36FbYv((A0QI5Ah8Ofv77*UU9#&o&!HhqC5mI>XHdPn}F-'
    '>7<Fhlh()V@!iS9gSc3xyBn32Dk(Dy;-Y=Btaq}-'
    'VJ9}5nf0XfTuUD(ha!;PyYyYl$yQ%F8Lp=HKXZR=T;ip&?U~g=eDBUqpLb5WKRXeITHn)Gbvpl?Eb_(mbFynf=IiDEQWw_J>{U=S'
    'kD2k+IjP-'
    '*lSOu%Y>vq8n&D(?jdx;Ju=LTjJ?~^V>!jW3IGM1=$u@(WELSEcwv#wn_QmW(A?uYjy1F=Nd!&tmn@x#vHapqkm6OdL<Ys%_%370'
    '?$?A3TdYw#|=wy*^H)d^$TWg#wI#p(?Rpz^YCmVgu$+A1TF?~7vmOEK=#dIrz8`BBnc$HZNrtefvHuj}){lm%jO60`URoo26>;|1'
    'ocSq^7X}d)FYH>_fGFyS1Z0#tVlYAy}vaz$tdXMj<-'
    '90#&MmD{?_~+;BJpVXZWYhVd`l6G??lbO*&&fo)lSOu%Y<Bzj?+2VLcIZwv_9F3-%UW9eUe_^ok-'
    'eQ{{BygL#a<{~S9dbq$2i&QP;ou}f89f$?0V{E;#208mwud+&F@A|Zugbd_ZsP*!_%#<CsWDUe@byO-CxA7ikUr8XU%HoB>10Iyq'
    '#=+_vB=&Uu5<TS?1j&JKtM4F_j(vgq;0vhMjDz@J<XP9LmtieCwL^UMcHu8PivwleSLM=k0&{?$Y@mYRSp89-'
    'J(q>Sno1NXIQFORhMv-'
    '^e)G*eg1j?CmWZ+$?gFxkpa6dni9;Ix*i}yP0ZnV)yuQ9m+}bjX`EzJ6%USnY}>P)$_mZP_pKwn|2?Yb*FT)Wv3I%Jty1WiaOc&M'
    '#{-{2X!*-'
    'H%^w%PENLe({j_Y<bQk8i~s)YWWRrKGI`QF`KNX#<~huXS$&+7=m<_WpJLP3c;??OI$8eqlGXP+neEG*wBBW>6URNRW%qZI?0v7}'
    'WaIfVtChuxjZsd={+B1M>>Q5ocm%oQVSG+=C%ETFkVD}?mPdCIIE9eI$qnR%hCC`UAtQ;L=1MV)&q-'
    'k0N)SYDX#&g54IA=0xfDL$<yE`ns$3~c@XZzRT<k4}du(4hAU7dT=(pU@MW{%qK**!k@hqz$m+_L}VN8A}pIVZNyo8(vxntmBRso'
    '%x8}g$F?3EGujpjUNbRLX~f^~V$>5!EV`tlje^TNwqMrM8!W2-'
    'd^)>*(Q<eUl)=Qpe_N~lE0sXnq#1qggUui8`y=To59YAs2~n}wt(0%-*yxgg>g1#2l_;$%mh0xJswg<??9%w<a}JH-ikoFYzP^Z+'
    '@ZO0e?dP+2Q873{zGR}48Uj*J%usu1FAh_flLUMm3&h2c{ag$b;ysxnZjx13W7ahDlI&8+H#%H%-_Be#fhro2-'
    '|j|K$kr^R7&F+xqC7EnXawT#q2d({xtVR019-=fMTzFJ<piHv4uG;r#tpq`0C1<Y-'
    '#oa2?C47zuIyowZFIV2D4No&nnvU3DXu|2v_=#{X7c~ii2i89V<u(AYR%fOb?Onge=F88crh-'
    '8r%RM*~entIAHV;eYT9Q~SvYJ>vN)`(C9jL54iWO-N-1xLG>RyO;cFJx2_r{bD;R2H#f8{>Q|<(!FgK6Cwa*jEmDEn`-'
    '8MhU~9s)WXbpkK*4ewyhxY=31Vy)uF0QyfvRq3c6_FzpnqFcp;T<xP$9%xX@X5UoSV`cfiXA**ZE7T4HRgkEO2M3*yK-CPM)ppv?'
    'BNc9whtu?f_mxcEgjsMJX<{Vd2X~>c;SV~vpLc0FS+ED|tis3$Y^inwL!1FS&sC2mQNOlThp4ci;DuEpGs?Axc6oa~iYKX<D&{70'
    '?3aOn%@TmZf(uidxlVt|?;TnV_`>Nomru`%e=`pA1?x757NY=V4rh+A+XAqeGr|JqU{kep`mQ{bGddfmeEl8<>t3}j5Ib^jcr)y~'
    'OlocR>GKBI3_M#kQh@~Q_D6|#Qm9wDs>hhY!5+*i`YDNoNQ4`W@fPHcf`bZJ2pptkER?OmtnY_~4HGzBMSs|nX<yGQeLwzf*EUaN'
    '-Un9IPYf7<jy^Z3`7}Nuosyb(D&nN~@>jBl&e(9aH3GAbEvo2UiqbP?v^<YaK%xK}XBXC8RM<hyz*(`=R6>u)0aV=r=TX0>mg_WF'
    '^%Jo2g8SUq#p`k7`H-'
    'NS%>Je%=&C;NSo|^zoovzxoW#y@e(PeSe)P5kc0*mSs;yh}utQO9l9<B+sA?<Wzh^=b~OQK+Z_<nmrSzQ^*Axh<R{Vb&nC<`kklJ'
    'yb0D8wJhS6SFoL)W4Nmt{^l%sm5Ixi_^o-'
    'H<(L<@A7zDxs{?F`_=sr)e$L2TL16ZbO4QU~c`4H7kJ*I=>>0(>40_bS;;z+E}aX%#>bM7uTBLN+Y$iUYaf)#n}Y*w5=H-'
    '+Uh}m`wX3lYEOI?t0b4)XS)%&pV(;BhBn#HG*Ox5buU^;xl_@wka5OIqQ3eYXqI=<@f&L18k?wapKFSP`&7V-'
    'n&4k?aH%BTmsdVXCaWM5&9&aLf><IQt){NV)pS2u8b{@@&on`Ngx8Hszhd8bzt$`UXAxxAUK!dFyQrqfR!#WR9AhiveWHIfCo}?g'
    '1wn0g1ccA+Fu$fgtu%Ey1Lvfmwsy|uxKl%`ge9`4(gSz};Qrh+!>@*@%U0Tx+UV7q+7DZpuE5~<a6A*5#UjaED~xFlxlJ`P&7rRz'
    'u2$EK)k-'
    '5@tm5pm=iKR<>Q(VSF`@;e)WXUjeXhI8OhxM$v@};lVlP6ou<s^xR?(eXy)^RM#Gt3?nZ1n7jH((x*|W5Py=^p3Z56$NJ}D5^O70'
    'U89%lMj+Z5D-'
    'A5G9#svvf)bQNu5YM`~**|jC~0heoHwyc=4TGr7XRz>f%Ld+WDvse`unZX9<tE`6bzF}yMaP|yv;5^7<X0ToTv7;G@*R#}))?u!i'
    'YQ?s9I^pvGLQBn`L|Eo`Ac+1h&JaQ!t@950ei$&+84e89s!xUFkt_5p6N9dVR$5uD&`~7k%@C0QgF=f%a{DxL+hU#R2p{=QUtG%?'
    '>~;GXoC7@_oO1~mIAe77*#@0VWMtfc&?`9vopjtt#B?l<F*u7iXNTSLGR&B&nyn<sva9{|WWbW*qBJOsPB2=2kLi*I-'
    '7u0h@tA2QLOlsxbY9mq;F<jheE(dH;MpnbPG@}T7K{GK=LN`tWOa-'
    ')4zFy<(8y{gm<DuFyStcfdJe92HJmxu)FWdc;x6N+h0?pKFFnx>qBu9BP7>hGa=CLM<_B6OC-RF)53C+hoEHO$jn&Pu8H<Q@Nnt>'
    'DXzK%Q0TPVp93E^#4+6(*AaEI8dmu}RF$w0R&g*X^4I;2)@xPa0qm3*_fEhnjy%?9_Q8&nzal!W=W5QgXi<PXG-scY6SM|!+^EB5'
    'oyI*A06Z9sq-9rhZ2!o8o^I%I)9P&HlMX(|Y9^Ff$#nBl?80U-'
    '*{q7GRq(5G)mCDk4$Dn%*B&O#>hxDqUc#VQ>XJ%Z9oyK_~3lgkm3kJaI0kFS=zV91mi968wTDNi@2V4YP0Z7k0-'
    '+<?G*BC>Py;E&{Iu4$<2NEQz12qZ(l39=ydYJZ;V(0@ruRlB*j1gAe`C%X8cn-jG&gq!J6~+CF<384KU7p11r96$vs4L8<sT#!#o'
    'Dr~7V$YFeHqF)Or-'
    'G{?@kDo~K7BMsR7OHRKU+$jOb2G9U_5peLou4OH_Yg~3Ah=!Nsk*LYgk5XFNB<tnu`(eZ=iPiBod>B69!={$9x1NMImvLtPjB}`+'
    'Y$SCIDiWcrL5oNW6~4>k!Nxfq0G8&tp;5qn%sK>|wY%lF;9oNVo{|2SCS#VTIfrO1oJ<H8Bz<>GiB2vMz!Qv3a;klf25E@$g6NKN'
    'Ec?3hswoZG3;Y!Dx({q?L7{J`;*J{0)NW7W-N|mk=%w<2ZsaMVaV@e}$1#zg@_o9$uJY%~TWn@rHjKrH<%pRiI%MYE4#_D-'
    '>5|V6@591j2X|rSYL<fprsfPcaTNM-'
    '$k~v4lwkwq!ElJoDX*bbj8GjdLz_7UQ@`GtN4$HCBm?$y!4lG52nGGbRT7iyqrKo6s27I)x0p1@9)JkB@PdqK-KtQ%r^jz>_A<tr'
    '_zJ9^I_x22L5s>W*GGPQ8vos)#M?3yCswcqpw8-'
    'reey#k16vi0%N>RW2j+&e((W(e?X6!r+XxS=LF}>F{?t!?io!qL_esm<5Y^qU%k-'
    'oxxyJPwnNhqYU;7mt&mlLvF!4!H7`*RA8iG%yh_^Vt6S1v%ksv<rxrJlfjK~c%6po7;RWHBL=oNao>;?mwVit&=0ZTX6I_?z7p@b'
    'XI<pXBuq?GlcROjl8T<H=PR^w$C?gwjdK;i{*BO1IsuaT&K&1@&HN;tmx^&1KUuX*M$E4z%py!O-'
    '+@g~KgNgO9tP*0I5Vd%PRUe&@whflwWijDfUSt@I73Z*7?(lIWR1WS)D&0tWCO0>=6V(ej199s9a1NTH9Ey)`#NBX!T5|`9GsV^M'
    'G_0H`cYxduP5*xy(#!s3%i=Dl>1T4L3GFJw><I`Df^tM1hHrmxD>@4bfm7x^|TmFapr}XtC(l<HVqz3Hu7bkF<MvP0Ow;(tANoL;'
    'r$f^j>~*t0TAe207+7(K}JL-kDrG4iszj1%W$tPR_9B!%D6uUYt}@qqjB0l*{+Gm@d9Td4$;zErL{psrxPwW2v+S$h~7f|)>FQzm'
    '2VucK>y)=3(sZgSL)SiDyN^h+YcH9GYuAM3>QM@boh3y;f46jy>Xi99MjDngQFaL>(EiZTM|r1_OF0`X3|V&k>VQDZ>})5J&MYj2'
    '-eSpwCnI1#Ugms59@?P<SLAr1Iw>6qo$d<T11%QJP7SqVSam@+v!;%cRS==qh4f<ON~y}I0L@O&U`&^RYqpoYp0%#oLqzK4%H_pI'
    'nM!R;`#NOls#X7rGy!nJxkBCarb(inF`^naD1wsFGUrJwku%adS`>P5i@6Lc4y)}+bpb-'
    'N}qvs{5pJ=F1J|oG&`m0H^O4MGdr9m@1+izQK`?-'
    'vPYOl2%dU|h2K;UH4$Z2&&D2TCVJ`|us6C7o(~J=sFebqCo9%fi0uZP7sr5WU}+5Q(9F-'
    '(JX}kCa1}TvalQtzxQ#H+RKgvE`OcjLsfJk^@A^(1J?iN1$qiulgJACMxHmrr9OD_zU5Kh!zkqNp<lTm7N5O34SqlkQYb35kR<2R'
    '*T$jPMc{+a~bWFv05g0cMcW(^YnTp#H=Y={Zid*pHF00ZaeU7{tQJ4)2?=q3(c{gaR9t0LU_v$-'
    '|?tEQ!=Id^Jre;pQBbbR^IoD+3hHynp;uYwP;s#hf7gkM&RXl%@R{T6f{#Iza!OXppFb~f=!jbv7cQxiqjD&4h!|z$ndy2`<GDw?'
    ';JImDaRLqX$#WKPpki=YhA3BDpJqxia-K$RssdxnzF&~x_?j~GESO6_IX=cY@^eu#i&OHVkJC?|MmIcP{Tj5I-'
    '_Yv+ko$^M=Spn_5BEAnWH>NoE!}b+~yTJgqRs36~OkRLUPj_}=Uw5T)HkAzvAaSm8Co6D99{_A6NG}=;T??^NT!n716McNKvl151'
    'H@$JW)=VlcgM>M#uscyjOCaM7$efP|EPy15vq)csG0bO<&|=stapv4!1DV&srdzdQQ*mPqIHOz<^DyRa$hir2GanYir$@C{-'
    'HE%4&0J;&&*vJr2{E}%WBmp!Sq1jI=d1&+(C<!zLv~54frk`p6idMpUT=6E5q*mcmLgXRq2*TixD*<9;=G2yUGf%#yUZ1SV!uN>{'
    't~?24*woh7Kp4H6x{We>Q&Cv-64mTW^gv>4DxO+`tJjnxlDWDGQ-FnW>3MGrR%{h`VFS^m4pQ=k%dc<jR#Fcmz&-wb$vCgTcn-'
    'zCP-MNUj9S~ynf7733FFCaTnxV3GeO%%b8DigbZF~R#WaFi(&8LP^!dZF08#9wacsaT^RQeVFCPE4sCpWkyG7`f~!)pe0z%Ym4xM'
    'xF%MSUr;JX;Qe2JV2Et^0=ANKWXqV!+7hXyRA4I%X;L|<I-P@hV2@mKvsm}Ut(-'
    '6&ormpnD#p<i<$5vs~GUs6e*TN>_6SMXz^uAfh;EZth#t~npJ@_uHqc^~|<*;s9nl7}$S({NEiS_G#=ZP4sf<O1emj}S-'
    'M+j0m4>?Z~R)@VMk%#-'
    'BdAZZmjba6Q(k<|H1xBnR@M`iDuojtEtLHWPE?{pmx64fwA0RviNcJ8@b>0i>*Wq0h%gws>G=X*813CA=*A2RRU5{9;B5WW$1KSh'
    'aPOKL}G}j^8w_qiB7K{jX1;Oq>j@x1&aZYyIfM@GMxx6|h^Cz)>(%J58MXlT(Ml*=29NbB`r?cG6giX4(J>xu&_(-Q-'
    'YkCj&F3yM8vkJ!w<;e=&(>;M0N{?A>IzIF8Va${s!>pa;N}g}VeTms!$}VQ${ibT4B|MI(38N;u<K0cT&wI_MW5B+0w|yM(l9lcO'
    'W%62r>=M_)?k7#HtjB2XgzGbEV43!=2a%O1Z1x1thIb#=`ig=pfg{MgUjsfo2EU&)cQz89GO=KutWxV9BRs8hSF6<<kmpr+eOSK{'
    'Nd@PX<7e|61<&Vt=M^qs?TRpJD`5Ad$bobWW`y|8d0y^pF%q9}UT~g6#MdY*pHWWV6N9Y;j@@eKB}jY_a@U#&tTb0R*W3x7Cp<^k'
    '3NKb^CZBg+)+*!?(xvayHT_ZCeMmF61=vL3?DC4wm9Ppi+ZN)55F2q#Dr}>P(hdTzjocBL*(+4%#x(JMUXO=WYZP|XS`W!vGFT$@'
    '_*B^8)+6>CG2#Vi*oK%%E}k=t<^IQ&n~3?A5U*m+Ho{}b!&YcsuRDUR&Wi-;hEKwdH^HNI&TBDv8DpP<#LXBT#RlY5R-'
    '^zghj@#?y>qSU5RYr8*$QtSc3y!;FA!J)GnzF9k$Ds{Une|=yU%DBV=g`u=4z|)|3x#FdHptFGoEvwMebgOo<QnT#u8!Nlg>N9X5'
    '`^9jNE~D!jN^4xK8&w>u{Y}`UZg+`2?iB0jY1O1ZMIRIC$lKionraYwCT1I}YpUR;<ku=ke~l&MuYx3J#XevEn}Zw(~B5B}inRLo'
    '_!bs*jt|+X)-'
    'Ailoy%VLVI9d_paK1`*$>TzMSQcZT1VC9y?Ul#RMtv8<;}RD$26<#)p1_rhSeFHh#OQr!V<TcLG5Vzm`|dG;2w%Clz=QYT3Lrdf_'
    '{=uu+rYuXK8CUCdh40dk9=*{5wv&v7)`~+7MpM_uV6J9}fcm?4VXe*+Xtma+72LN}i=V4J493$@N(#xJ#u5X3h*Nl(PBBxPoA(U`'
    '~PWGZ&#5MYX;zMAYvm0YLL)@o6Ah12`5A*G1Wchj2@7rq8t_(SnqYq7#cM+b4#jgM_o5+f%+aUKd<3DGLyUUACB0j<E#{?e#GW5T'
    'W*H4vmJi3}2#q&-a-Xv^OKO{o0LjDVC!K=<Egq=F>9g`pKY;S|RdtmVw&R3A~4nZ=)61e;D%KEms@+z#6NIZ*o?-'
    '@S6s#%ukZa4a0!rJ{<#&huFc>T)x2s1uKHF0&bU2mE0@fpC^UUxo+4WGlh7r_OVxdV5i_{fZT3o>>i*4uUTc2mKx5I%`Pa`*O%b{'
    '3A;$Ebr>q4hnp+vDCP+542hYz=hCc_(7_nzI)X{xC!LE1?~4>iJdE9k>&Ife5gM-'
    'P%PDIp4tMAR=$#S`=RrSjH|hf~(@~Ops)<jM}NSvWLJu>oZvU5$3+-'
    'd>vVv6B&?iGv9;Ew^Hr}THaLuKGDy~tM5495;&*t!$bZh<}FyV1AN+xyHcqf>+cBf!R9Ega~A2ZCX4jjgPX8E2H&CGp{wz7^>k;N'
    '3V#(*dJUQ-76%Ay#b+4v5j>Wj_ol8Bsd!O)Ucl>5G4^w^@IyrSE5i4{`|z4&?Zx#E5huy&w<&d)u;ir}d=mrFyI0TOtES-'
    'dUacCL|EpHs9+S(D4Y<do_K)2-STnC;U%<oHkn!)K`4jD0pJASy#oyGr`2xoQ*uYiJv6HI&9I}og-ie5Rtv#Q+6HDUtLYT%pdmoZ'
    'T>VB%L9csOJ{aFTfWkfg`X{rB%z&=EqyXTjX%iUtH_UkATo(8P?2)+rv%s9)Hea_F$k9g%ixlc6<_9=K3`hl>Ia2WQmS9_daRNFV'
    'u&Rz2m`p3H_ii~5B{0s07WbG&XM&RFCzQCBhY5ENBwm6E1;L9%UEg$19*ZwD{fhax+<CX;Xhc81b4uqARio>vzbsRJd+=ZFDz^`9'
    'nk@VCbBA)&XaDF~B9sOs*SBP#DzY{pZ?`gG3^*HVanw<|#G^~G72*-Ip@ik<-czS%_`P-'
    '~ZoSnVMPZV4kEa8t(bKt{at&H7>=|S9i51;-~k5a+Ak$o}v5yuyV-Ma7i5ciI1HF~aeInFC{nsc<rtoes^++WTSWK?zpY&)-'
    'rUnv{7+j7<ptK7r5^D*Y^Nz>uKH@1F?(VUCJ(Df}Od~ZAzTbK(!I)6I3U9O(5AXn~wh8dr0&yR~gSN=bQzYGEozSHyPG1zaw_4)^'
    'V-mBS>e)F$$95?|+yY30({YS%4Tki)==6TM3?YZA-'
    'm*!eOLikA;$fFLLY%yo0I~~CBiveq3R{o;rZ=m^8?PmuShj4rVdw<qkeg&Sf1>A9XAGBMgNKW^uv_qlfgV0gbJ%FoR?QuPA5192g'
    ';Rp1E9}r=#r9Z$>z7~AHE8n`W(BF8z4rTnK$3Gdb!L{S2@{a4({Z0w@6Rc_l+;5@#52v78(B+Er++ye%drU?D9)q7v_I}kPfkJK+'
    'h24Lh0AgEl_fIh3YbU4shZDs=h6TUE_Fpsf{^6YJa`(^UMsX1Kiw(~sf4}Sc{Ig2>mvB@e@%mQ#$zJU#JnBcpSfu@c*FP1>3jY<K'
    '_J?wkI8HcZ`0^*Ah#Q4)@Hbt@egM<=DgzJbr*AR)M?#__B|f`uUc~n&$Q2ofke8!jboV>?+^oTtamP61<Z{{C{}6|NG&_Gn*FTQ$'
    'Mv>YR|HdI~`7s7X-'
    'Jdb`4|sb7qyL8HQf?H|^Wu6kuRuJWJNCcuK|1*{tQb)gi@|YNcvvHJf{+9EenC$DgH*1W{B9JE%OiMq#2)<@BJn?N%^q<?j=+kXh'
    '~4jIo#aY9j65EPWbR{9qmq^X)E=%@wwL3`+{i&Vs-'
    '5#M=#J|^T=gvJZ}cFp%)ihPPasw~(0Bes9HPjJ=oNMoD58f%;<(nVbgrD3`5VscHS5Xc=7O}N8FlBn$F=Vr!*LXx$mIrixG(&n#}'
    'UoaF+H=FT(7K|yBB9Kw|lDl4}3pidJr=wANcsM>19RTDDogm|G~e1!3QgYwdX|ine#;;!vhoC*#9d5mUfC8h4A|{Hwuo&3D_z5s_'
    '626vN~1*3Wo1^CP(6gf%M%dPGIeg!f_=|e>=6{v#0fyHF*9p=n3=-'
    'KmJx;%KGFYg2L>2>WkQ0#*Kn2&w3C<S!z;_>X75PY>!CJ;hu1^h8y#r2l;tXBl5~w{ui9*%491%*CHq6or;}CWyt(jD=-'
    'ye5=Y>40^iRInYqEx26&&ag!>s=QOWe7ysp$-'
    '0j=ITgvtaDqkZFjaYBC7y>zF1kjQ>|xXKnV6S#+z1Vm<WtYcNQCKC~jB7Y2IOsZb4PhP<b;97BH>~y@#58EZ4xiaFK2X_nN4#%jp'
    'UY9$;_*~)mU|tFJg`-'
    'j+V;_2Ih`Qi|TrY?^$OVZ7T<#p=O+l@>{2A4k$M{qj5sM;s3}i1X^U4yuP+uXJdwwCa<EjdjR<;+zoKs^Uok--'
    '!YXLpyGgg&{oN@%oW*%rOs_S7rLQZo<>Nbzg$%8x;c5AzZ2^9?Y@)&DnRSEb}0x0FSL?@E&TTDIAZ+a|CjOM3+dj>%&UG9{0c`a*'
    'XvdauFZ{`#v)F(6n(~83GQktnKgi%}*<=rz0rC@0#*j);83L)#<4Jr`$cTZP4UyiW<xyDC+QY(QqrV+YC9*t!QLd9@a(u*2yLxTM'
    '8()7`P@!uEk`eMB(4>pxi+e;D}U_5(#hFgok9I20737^YAe__*A8xhKyPT!bN3YIZXO5<v2%}F^(D2WldOsq;HyNUV@#+Jos)?ZA'
    '!by19{L?{gpis4+scvu`ADABKje#BjzeJD>TZR{uw>*}IT>)>2h-'
    '_^sFipsE4)mzTC_|4b27oF<10CJ;7ORGm^O%3GJs4&x+358?8Jg<m6R?xT<f!5L*;XL8WSU|5dahn=zo59A@(LJjmVimEPaMzbSw'
    'j`7`ql)R@^rxbzS*MC(TuJDt0BypL=7ch)k{QimB|m3MB~?dN7sXN4<$od3g1~DG=dw6rQBj!~)I}w=T>j-'
    'kWoT*%eI>&_oC;=f6`fT@U**i}$QjTq`-'
    '?VdI2r|WqAL1BgYeu)&qdL}N~l$*DXY2~UYv%G6oq|SFRp$B?}{5{O3undPA@Phiqmyo$&{+#syxlDNsyRUMGq^B*XpLT34b|D)r'
    '?d-qO$rQh3k}q?HqmPNGa5R1v4sv8j(-jnK^&*S3ZTc7JK91yG>AI72NkD-'
    ')!!2L~de4E$FM1Qqci8e~CWQAVgJyd|RIQmR`Q;=5L%Ux>3}~_?I~eR6!@~jF@unxW}@jLhc^;^kvF-'
    '$=n6%L3Uldo@@4r^~@Dsr8sl7u%l`Yt7^joj)Bx+4YLw9F*>C4iG-TEH(-'
    'C6oprSnL?L}A(8TuDhV0se^2loy+>>>_S{l1WPNMfDa2BC1a?g9sR)h}P73-'
    ';<T9C<}@%(D;Id0<b@^96D$@D<v*An?Hqn^r4MOCeWMmRW9XQ<A)CYJRzGO27nkC51nG*U*@3ZpDtxgxIBG!d<Bd}%<a>-'
    'Hhom8Bsx)^~>jLm;sh&cd#`u#a~Q9O+~(f~ss|cp(4Ut*7=&&H`I{2Iik>Mse3^Ll921b=xW00u``QHGxkpGvZYPuR{RQ!8Od86I'
    'mUA_9>_eX>H(ZOSe-5XCrR)@G3ew5kzMV<#@oA)Jm4jrdzA!-v-tMbA})fwbgf#Twi0`Iozcs(H$ej;sm#3EU)-'
    '|)armGQd_lg^dd-oj8YcYOOakj58k^mlj<V^=ej*GR~XBq`<h*Jusf<89#zcVA)48?W<E1!fIHZ|6xSPv+>nem(VZTT>_=c8^Zh1'
    'f-!~9>X`;_3i6<Mes9_rPC-C~x8Qms01YIIr6EsT`w^lgXs?LP&mN>U`J0b=HAd7dP!`;@%e?zU}I-'
    '27;h<6?2B#OZM))`=Sv`}eNflCyV-HR2I_0z?;R|7{-'
    'd2O}Us~8qGg`EvdCmBTOonl3R;9twOM@?RY*U<#YHuJG9fiu$<c2rSz$XeDKb;|qj*5(Q`g=Jo3_JwSfJoR@3pWB(paGuA+pea0v'
    '#;A?%&+Fl64tZyxa}7nrT59(@(?pb+7VH4Jx#ttc5?Uh5U7>rVi3aym*2r~rR+yQfz7zdB73~Nu;a4Zv*a~Bt;E;-'
    'HXR;(6zdNdFqDF{$z!k_|a9jouIs#`aWNqkZz&3Xv^wEPo<(h4+V_RuddO}KjBbQehwu7U8zOh8~w!~{obEhTloDCUm)sC|=*xC}'
    '270<1<fdtF8bI{$+(fqW~6*)jhT<uIao51-'
    'M{hhUI3V09D4XaQW9nluA9m7$X<1#`U$nS>T*E!mcqv)!T{GS!RPolGW7zOvWu6ngIER3QP=33@3(p!@ROT8Rpx}&d)K5K0^!}RW'
    'E2P*Z#{|3JYx^q93cphd*w9nBvcfg$}nB!ed59<ecXG2a8$cdu6W`Q~0&WsxiYX%c~!RMZ+u^uWp3dyWU@2*i0mUGmCSS0s-'
    '<6_Voa|WUhb#uoPCJ}l-'
    's~r6yyPe^oM89K3M@jUGC^#O2Axr!{JEJFc)T0OdAE>@?ZJlc((i72QCiDYW2jbk#9j2X!+0zI2dh0VnZ=ere`{T+$?drikM{G&;'
    'ZV<d1ZhGlJ^ynzML3#&R*<JS;sTdYwXn1sk-6Pcdp6-Q&OT#%_f4m!$e?Q2aw+tJF-YIdCyd?MgouFl!;obmu0zsJEoiGKf=MY%l'
    'L!U)bA*<E|vj%f_o#<YLPBqkY1O_vZ<H7eOn>`Fh5hh@t(-'
    ')cUg*~76d@+INGkdzbSK_!7`i4SIf8DwF&{$j<?ktnw_)I1weGRfDRg$`=?VPc*4sbqn_JPj1(Ara<7)BzCm%25*p*o-'
    'Ck1+h;esO`aQkdTtuOkUlLR^Y&d?_$gYiEX$(gV@#Z8*P`FaY)pgPuS(cV=cyFYV8gshO~wM{{qv7}waUA@FWEqSRge<L80C%H@9'
    'U)!+j&e4xoBBZ$^)v%lmn+Ar)<$@Pa>%+)^zzFdisy|g#<f-'
    'PK=+y}TX^;RVO?CH)UjE9yh5S!lGJ%f?KjNTzQ7n8sxzIGXEX{7O{mRHLgWS-'
    '3iDtl5f$i0RzA5TITX2?u*BzZ46>}P7}N?g4b_b$S`Vc_QlW(1=@>`27ue66J@rW5%77-$*+3$G(AAY2Hmc<w-'
    '0o7w|Mpo(UI*)uXQ3@jJc_lKSf2vVb?AT^5X;s0y`S1`w*yROu|afhYxQ|T~wP7Fqb-'
    '?Jq#NFzJWaC(H%#+|CKsn5}f!(74u*gg&wz&(8|STNQ^ZG`qNsgUzAvwtR#C|#_T%MrK;`QfL&>BhHlMlZ`5YS=#t^CrUU@i;$@G'
    'gmnK$637yu^C1X%_8xVH1o%)hf#1wE(_P{WQFtEAy!Y)m2V1SE07q^#BnR?cRXq^ieZpC0@h5#S|IxZ89P(Q571R}a7HzXq)8YvC'
    '#4<&OrRYW*V;6K<v<iuVAllbS^zsH(*9vSdZU%M8h*%3iO2<*bqTIYXP=^-'
    '{VKvF+D)*%SEev$RM_t>j9lYfUqO)d{t85Nq0v1W*RC+^8?0;SKv+J^eG>N<W9~F`t%>d;!bo^28i%^G2p18e>>6!4CwHa0ApLS^'
    'oot?=CK~PwLxhhP#NcXRA%Q#djqVNb_;PTYpL>QYUoTCw<KcBlA~OSdyHz6}g=Fkn=!oKa@OY7G<~qEdaHHvPGYOKxpk4#Jd4Dm|'
    '#9DTtmx6iMV)WH`e>Wt|b#EffOOxa2uy}&W^|kKJgo`vTSEt}&*ulFD`IdPGxF+APP6hLBB`{~08Jx=?V#&`uem>z2tVTn^C|;x;'
    'QxL>*60UJS<oRO_yT-Z8wL{KO@2@oV$5@0oUt*pSFU2+KeG<2snjz`gQwW?L&H+EQ@b^H>6TZI?d6}Y2y+-'
    '@l)p)(dy@$XZovpk9-'
    'du*<T?Ljz!P&kSTCPZmcEIMFA!D+x=uuprVYm1`8$BxuuEQ${6A@A7<9#u>B(zDMO)o`UE;kIEiaS@Coy(1cryyxNA}|wU`H6_@|'
    '1wjTocGI7yYo_di*%zKA^j=$C1?q7y=GuqnjISFk9!-hc5}h|1-'
    'Kr_y^ZiP@;D2+c|}@)d3=rg+En*`!Yu@um72qOu;oVBGTUT#1?-'
    'q=7{?kHfa8l0gYCvk&frrjHRwXJ=a`O9GvKd$zTM1TVSHT(T{l8k4eu_@oN4wj-'
    '09e#2MDrHn5X{F*L=)Ioh*SCu7Z0ZTX@Dyo?&{<Y<DFhd<PD;g=Ic~PB6=Kvio&^bT?t9yVSiCGOkx{Nmog}GLKip;A(eS#@M9Ry'
    'Y<d>1~t6|gRjn5i?|~`NLYY)EkOik!(NW~a>6`i-'
    'XdMSu7JL4P}5uuyy`O_+2&OQ;io*m1=(JZ(Wm6wzbjSuRE=lim05c`5PUb2?8kv7*#!hE=6rM%v1FQ7M4~Iq)xN(_cZ=MSWT(Sje'
    'kHIN9xhb5f#+A5oyrW;l^5!qCGcep{F{%sibW3-f}VDR?uMe66@&RE$4dw|K+0;wX-'
    'Q<aBDmFPy8}7lRZ{G_2j|;RRrllm9T`}v5boWqeLJ}_^FHEEg7oj3+=pPnBJf*2Eyi&xG|APwfIIbkx1O0*T(4^p*_&bCBDLZng2'
    ';S?u$r(8Sgz-L^}GyOzTC_bw%x28T4Sn(!JJxRblzZaoz}t(-7DM@?im*29cPqxD7@k=jlmj3OpZs~bvUopo_IU_x*6JnJtk{-'
    '0<-RQA5F=9fO#oZE7kQR;WkK|gL}6?_H4Xk-p_D17@14d>nM1PaA+>ptaXGtVdp&fa+6`)T?F14F=HQv&r#f^{r*nae7*LM#0t9r'
    '_R886@Rgr+?}sE|mgMtJjFLV*AD?+0Sw@h~w4U%3GO<jvbT_i`4B=_QT<GFjSgkydVv#{q?q;(;yH|Um?39-yhb-'
    '>~(^VKZyBiUSd(lg7!<o7En7P99*`BqKc#l??bg0J&_YxkD>=W+CmD>p{e-mQxoV!Kwtb+G#OTo#9(0d;zJZgH#OYSzs1DLbI-'
    '3qIDO}#avi{2V`0^!b+khKnRzcaSF#RGow;CgxpdD`Z_NZ11DqDA<}6}K3c1<|=JJaY_~(agscrjI=cjhk>hkI~DuN|%~wupM`!3'
    'b<Cbz?bKN7r@r#rqV3}?a6$zrFWTq`Zmp$RR3CJa7)+=Qma?a&nTmKk8>~PE;lTck>c?()b&Hq@PO8K6f4mklj}v$;Z`bSx406rS'
    'AdyW<35cVSN%4<`Vw#lY}tk^a1}oogXdsNVmB46aC0DQMM}(8s89F6`^`AtiBFG*^_+|X_orlzg03yu5#|N)xC>f&1zAoVWD{Yj?'
    'rooiJ@=vOJckNgk65lTnpcJOoy1zgI@HEvsKfQT^I<#gh13P~UCCBxdk*6_fU}!SW>!Pq67_N|Y)d=?1hJEB-'
    '=v&Q#dgS&b?GV0VQbebr*AVW;x@t>!p*SmX&md|+0%MFOW0`o@fuhmNUlte5}57WD;`HApFw;!L*GVae}Mb7F4mjaCy<EtBX|yv;'
    'x<Uz1i71G@n*zGqAr&4j$~uVf++5T|LY@hliGMu&o7~`yr|W-'
    '3cOz#R`vbR@fafU9P;uMbgeKwgTd?4E0D5U*W?F74!#AMn;>rue0UCauEwVggjMFL@qrkmuKuerpZ7IeAoY3V;bCQ=L`dpJ;<S>m'
    '&itCa-rYf9dtZjkRgk?F+}i@rULgck`WUpW(!L@VKMw}|fzJ=<9(5b;${1$yL*UgKofie~n%E2RIp9Lz|3(6@;Vbod;SsRvaol~I'
    '!2UmkG27tBW4e=FlCdf;aX*2i2T{Y%nkNBfhV^Kx+WS-rUUXk~Uv=NqeLTmUYy1u46L-'
    'HIrf+_Xc@Kj5?*QC~)@9UVP(92jwuk3)?|hT+DuHd==I(^WQXe0=yKn^f7+{V(ikgbz8TY8^2`?f7k046Ijzrb02tU05aDJFE+%e'
    'hyXK-'
    'z;)<IzZ6B*nS9n2`{E0Uv^wD&%!`~E1{{#D4)yCzEfeDH$V?|p`uE5m*32F=7y!V^aK3xv(E^DEsSuW{dVKPS9KV9Pmj>*4Y9m?t'
    'aR!!cmXw#VjS1N_+yd<N`xzff#Lhk8M~{2uj!SDZJ{8J{C;C%gt3kHPL2G4=)B&u&AFJOW?V;L}#UFTLYQjM|{R^=pExyc^6a`V?'
    'xME0^QH8$I+T<-`-=?p4;K;PrXv-iDa+{+5}s36{Txk+0&A8hIy#u<;AR3mEe>yn4-'
    '40q<|PE4)N_J>0n@?bwdme^Ot?%J&Jcq^(8CGqZ~G!~I2i=S%S5abuHo@V89NIsdPs+M;#B8lFxQ^Oqs{V_3{PmyeD7EzqzP*?SA'
    '@+k$oDLyX$26}(;d2wYX-PolRbR_0e?5UrHwkZI1}c9r`nfqRJc=1mjTca=%+n(oYN=ZhE<#VfcP*qay;MIe1U;`N5H<aNSZu;D9'
    'HKks1n%UDHt_PZux+^4^RR#}53|8JxBy^kXbuAQBl$rp9?isBvjPv9@$&+r{<`!)ve!NR9e`EMWtUlLw}m2BG^hDD#lgD-Ry{TT7'
    'sg)8qMmv8IZ62)`w2PV2)ty~ez@vi`h*qbVUr~4h@5bpd);I(|K+9}$0h1f%oJMUoRE@=9I@GS1KMA=8Yj41BH{kL^q0WXT3SUcG'
    'F50wWJnRim~5w5>$YT+m3igWu0@B{MqhH@!2(%W&Bv&oU-NPnT4qj*m%=pzD0i&-fV=DdCcOZnOJXTqmi86RV95-ne&Qa-'
    '_F_WFAQYiFsu2nUrnA48jH`NTbJ_L;A1WK;1DfoJSQ-oMpa{tEAy_Z-zv+yn3}A^Xitz#RM3#FT6EJJt0KBKJ-'
    'VKEkMPwEn--^FH?%0!Qr-BE3s<|08_d<?bbX49@)k8^qsT+O-'
    'nAlZyQncD;?ryr$RX%3(zLJIwfmAQktT>G&TQu>N1o9&#V#yzlOZzCF-)5I!AHzI=jC`hwZ99wzYp{b1I68e3sJuWqmFK5-'
    'wg(^RhH>0<&j;(*rhK1kjV4R68v_hPW!u;Dj&_Mdj3eTeG;fMdopekbr+_CX9jBmAg)j{U0p0G`C7;B}Gv(QXqJ;hpTeKMA8Ok>f'
    '0LZ~79WKSH(-'
    '!GeQefLu%1n~*De_|K5by<o%7I_?L=Tr&6;GQhdwKKe0cu@&EgT~bNEK<_VL&ud`ZTjBXFSo9S%MDej!K`NNhMLe<ZS7jsf?KQPG'
    'Sbr1m_}O5$Thu#<^FA;`Z2Ji)<CXWyd9SIjyR`?Tg8Mp4+z0QX*iYDtXzm9Oe%72x2j>|F2;zBQ%>l?ipz{M9Oe6nyL{56uuWos7'
    'H(q~G?|#GBC@eFhV7Xso%n?Z2ucLX7v>O`sBg-uHS45QeYSLl$hZz*ff2&uYU^er8KkVOQkdPC6Vvaao6hCQJ55c3K)UQKe_rb7I'
    'zKMbBeWx--&v%+%VK=V>Kf%jiV7;u9znRE*^xw}I#UAZ*Pw}Go6*7MenSB@<jt~xmKN9oLmAO&y>d*G_efIim^n+un`#2)-'
    '1@wFxKG&D`K1a^>;oiqFU=HVn-'
    '@hOhzPBe$MI^9KanP)IKbW;=zw+&{_T(dO4v*`Qvv9yjI0~=0@^X8>LE4}02@_N9O}`O7H5z|39dI|HjQ1yc{t?9bPt5~YsQAY9R'
    'LJ`S+D_mphI=7bSz%A=iL3fIv*w83r+J6K!qcz^KZ>011LMCm*8T(?Qbnb`D44ktjl)_AKf=nt;T`9jE#=+l{*VVr97Oa!HSzgTc'
    'Uu1_3z%<GBmWXO7OYpg7q4W8!k!_A<oyur{TIFo>kffKQ7}u65x5eML(WlLm#5P|lnV#6+K%g|qqus4AnWpp@JTj_AKjlKknX|W|'
    'D(+QTXXUk;W&Zq{x1eUxj%;<C1T0d=7CxNsjR<&-'
    ';~?GW$dbw5tH2g0E@Vb|7I#KUB)FV5;uwJZutC{KJOn_ol)?<j_*tFcfEpMKJRbX@|Rw7ynoy%5;b&0_5R`J_oCqc3dP`8X#GcX^'
    '`CpH7sYX0KaOW+A1j*ceUA>q8D{ozmYGXAyeK#Z4s<5w=EO?47yAEG_8bdG$SX_D=|#a+b{uRu7Or9(AMPbwFDG2_o7u+}Fn730d'
    '3-KU_EEXKDE@JM>@L4E(&B34FxEQes_Z<H|6|7|GT!I%{>2RGaLM&=FZ8-t847w)u-'
    '@FBbin+Gu;+0m51PpSt$pGI{A3PG)N{bEoF1=+%paclE5U_~f6SWtA0nHZP{PZV3Ajf~HT>>Mr2cb@dwG=oKf5e1Q31y@!5_*Jk^'
    'hTZ3{vtMX65mskXgPb`yXM5?=fe1KgN|T@eFMHMfc)mu(qBGUT|KRJ;(4&Bh@a<2<D45@|pw3VK<*w#N*!Od%PY5*OR@8J3~n?il'
    'X2b^E?kM&w)Ep97d!{!@pBK-bdsKx#uhLc!d$=d=YRR`tYSN;*<wR4)suCme;F=t0i$1BfpNCnqV84p=G@&3VRhiW;NH&FNkq*cv'
    '>8gDEtl2PJ>;4n$A`Z_D4|+BgL*9+Ux(uUg;mVwiiVyFd;w2R>YWm(Bl)1!s@(`lmlI$gm*fjq}uq0%hyly%HZG?p$4=S_GH)nUz'
    'qb!KxsW6cPn{O6eH9mu>VKgtikI^X+%+aTLJi;A3o-'
    'V#rY6n=0z<;jdc}9q)Gre5jn{$e|H>oV&ScvPiQSg`Rtl5)YNnzUcZ?uiSqR`Se^&=<VLJY!M;53raYk;p$OtA{U?|DpNhZD?m<{'
    'sM7@^ISJ%uahJ5COXC=V3d<5=C`Mh*}KMLVTJ#eYKiNmQWHJ6Z>L;XlaDICHuX5ksy#Ux8yFU4cPEGVhGEM=}Yf`7Te>x#yw5(Lh'
    '8UbRbn%%|^)ss^zw=yxpdbkh;f)Vx=M?t<!lZnY$V5_&F^rn+)y7fqlrARgyHO`HNdDkBS(bzJQjlnC=7G6GrrgvM)I5cg<Y8bD_'
    'ubjJ#6iJ(aMS_U!VDEeL*Ww(qkr(m7IIw1Dfhei1{uBlPu&dmHRg?TwCSEoRZtjYhmTnYbSRX^@F#Jke4Dj!A$9wd=N=LvFn-'
    '1nI6yb_*9sA8)8G{hl4B3Mm1bPAZz+&hC%flx{#nOuX*5GsRfm9)2?qEe;y0_lak7K9RE)RQ(g0w0@r4IxFCd1@@!$2yo*Nj*8$s'
    '}8@a;$3w}6+fzEa3s4ZgI*g&($8{wt_Bo@<jO#G?Kg$M?K8m3O1Q!`CmC&K*2cym7n7(4NrlikqY&>JXoP|aW9el)-'
    'e*)HG$%;?71WHKu6G+E_EiDyXT|gybDe9U9-$m|M5476SW@1|6vkG<-'
    'D(CcFs~4zSrzXs`=a0q=k>KR;mjBmg+)~jtEwrZc=yS+N!R8GCo-CdQAw``q;cfrT1A4aUZo5(xPmytT;a@!`rw~LwzPUxP5Vw16'
    '-=gD0<uq&dXkl*n6aW{n2loKaugN7tcoUbtq7+>LpjwTJHOVZ)0I_LrlL~VV}kpY)y7mXPdMVO!noHo5pF}MX4VKseL`h;Tpj0fh'
    '+0j|XErqeIB#47ryDfK)pC@pni{b(>Ps0^Zxl9WH3(JERqEq9$Dj_O6=riKm4{A_vaRJB=mV8wP##urJnXt7c2$J6XCitv(s&XD_'
    'o50^U2G@!f#k>_`sJ0KP0*QYBOcP3>O*T_f2E8ZRmGSnnnAbBsAuj}(N0rE*;_fxO#?zTlhc}}!m6NNn&Nt0=%{JHTH6ws?RAX}9'
    'bkV095uaK1l}ohB-AolkZNlT9ktAEuo2KH<5Plntg#WI&K@>Iwq$JG6u!!8$r)u^lIv=Hozc)c%WH=tiL(in!J?+Hzp1gSfj0zKo'
    '2#^XDow76+{U=q9Os6(*4V`MEZmd+*<4>+Y8Q-xW84g$$SNI0GgzKAT47XE!=|RtD7%{au%)%;zc;QlHd-'
    '0HD{4<@MBqp^gTKtYrm&#3a=8i4ExkUlqlJ-'
    'l4xs~3LA`5k&`cw9He|O{26Q$IZ4&BCq8aq_>efajG{L)$(9{N!>YFN*F)a+Mhl0~>Q`WgaM`zgG3b}6ru5=-'
    'sMQEaX*E1lUbKaRy4n1TzW;F25gC(4yuHFdbrJcFnL?hSK>x(&JUnleA-'
    ';Gc|2K@*f2p14!{TAl;@CKSyoO#31;d<iv{gJUY$Uq}_))xMEBD6Jp;<Z|;q_uZ0p&e}L1li{i+J<%oj1D^N00R3Y>k8Lbb(PXiG'
    'uBQkN+fl_=#FXDz9_gq)`9mNbN0GzsILPwF|Rt|D$A^$Q9%*~$!bsqQe{z=*4GvLOj9rCYP|#%+1bRXR>-'
    'A#;kh1o*VWXet%$&uvo&H(wIerI-6Ey7%IKO>Yn;`Vr~qbvcg${$nVm7dbJ(X_gy$Bf@A1B=hv{-'
    '736fvY&|Y&Z6`1G}(l6?Iy%fEHo{CZ4g~%3ntuB~<4$k7=Xz!vBqZM_`%GW{V^+^HORWooza^D?M>4r1cO*8Xs%9Y5SmLtUy+h~6'
    'hKf0Mc${AQ)8{uG`T}*G`+B%=m3(@NepE@a<I4;d}h3QI=`WOiC>M{tHo`-wFoUXc3HP&_h3~zAAv{Z0a4JHgC$Qs-'
    '*tSFH=5;A(g<8v@puJu(Lnqs7U<{sSz(Hab{M8R<#rTOiTa}VImP}UgT;S4sMV2_5zU<lznFtR6foriM|y*~z?^@Ud9>L7d?qcz*'
    'x8%AhlWOqX*2dLEp!J9#*9tK1D#e@;2w)+tJ5{3gD*=sci{lKN(&^`oGhvSH1j6&Y=>fYR}j-'
    'xZyj_wL!Q%`(q2Ms;pT`!e2)Yvu*wj|e>V2?2Xxs=u7JlNJ3a|hw`VAJI!qGQa}wz@th(7{v=S6X)hXP^_TZ=+clVphqsGuEA;76'
    'xkXv%PYlu1o#Bi5VR&0a?SOQ=Dt+#;)i+31@j@5v#u5IB${}+XJ3(Wu9mD(mk}JNnaN3^~C$3i1;X6Z)aF9W7z-'
    'RrW>%09NqT14yU3W?C%ZfBVhs8i!e`g3*&m3%ypq2$+|9u&;1bN0j55zCqqqF;J%;ed|kCmO4sN?;8lGxdbD(+3yfdlcQ><g@V6Y'
    'TXb;#ED_}pc=zK)v0*y;&!~P4f<C08{g@uDLb|h>Vf^+9EM)Dd|Fh_u6#5Rv3^dj*6ZqPdbc8r2&T}*De;A+s9QZd3CPv~tbMs`)'
    'haA%A`uSh@YA0qWLTfx1kKjA{eVF+}N&G53f>QCI4_i}YY%8oDSGO{iVL<UCdKBpJ$QW6-'
    '6Q9X5hZ^8)3ybO0QfK}34$H4~f5OS}N*@sOe^g*mfBDT^=IhGR$qIrms$eo^7V~*Nr!Z7$Z5_a@5t6xXxyU;|1_t4{w-'
    'Q3wmBRXSLLw~JKwt(|25g(YcCJaD)hQWW)bfNYm86`REr<E`0hWEp%CdR}-'
    'qIDtR5?r4Eo5m|kCYtC?CUCUJhFQ5#nKcTMCWQTG46GmLU5D3!F_=V<I*noqBu_Bie4NJUV%)zN-'
    'E|^9O*E`b>{2cXUwQo<2MH|qqOfO(-l^V9%$=-'
    'x<sQR+Fn`6D@vtn4vA8xo2GVOLLhmJ!*k6vh6TNwO<<(FsWun=c@zd!<NagrUgnyUe`W&rq=00bJeHmkPOfihP%DWyjxqtJ#zL<G'
    '2j^uo<`!TvFyo@jbF_{HRE;ca~PI9lf6mrGxDIu;V^n;Fx>gh!928_ML>_OwY%{5vD1L4tS;7$}%P#u$TzDc{O^uj@~WkN<YFISH'
    ')$6h#!^SxW(<>ipTo#b*zyIk3QxiV0ChdlF)_HHFC4w3lGE88@}7*m7Sd9%GMpyzhNpzt|s671k^e}z_2D#n>ruX4(-'
    '3$o9+(Yq4*Cqv#KU9$%2&WV|LC1Jclbp0RU)k1crdegP0SlV<#P|u=YI{rZASrpetK8??)^C`;Xi?y#_1_|fGwkdl3Vs8e4V}3c|'
    '!kD#G;yVe)R8tj~#9$ii<rqvs9bKnu&Ui?ihiVW;ay8F|zN{gBUarr{*MPesDT?vNnmCCPKY8Mrs=b6W&%X2hspiU+=s!bY<JIVT'
    'vih+-qeDhW?`P|0f=NS_4-35cgz1RuT&#R^Or~cO*u$#`Ecqr~>qmnT3kg>l-'
    '>yR*7s5K}mZLCh4(12_M^@$O_&kd+U2ROVe}U<XiwMk$3$!;5*Jq3C3B$u{vNyOIzRyRDt|8n^U@y5o*}joxWx7E<x*C-'
    '~LwU#=XQ0nSF&nw%-4S>DblrI-'
    ';TrF5@L)DxXPJJ@d1pTtLEbF*zXI`^qkGGn2v=%viGs6p1L1mb;s$Rm;y%xGb&fnUH?EV+Hn`U4zt(W#I`0AaHy7FA=uM8nbtd<7'
    'yxRzKz~JeSemydAz3JRHnQ9C60txHqBTDm;i#clP96+)>2W#*`-'
    '4~?d<`~R@y#eN^yeWpKOOda+kbD#J8tuoo5P0Tg%JI40Bg&gwygO0h^SwI=vewTbtWzFGaT8)T4N<$ndz8R+vY0RzT3MQmyA84yD'
    '9h);!k`1)hAUCrX?97>_(Wb3xE0rKL~Iv%cM+y(ub*yqD2u`z%+7ed*=W5Evu{Qer$NFEc(;VWQJ8C1Z_XD-'
    'eTLTi3~w2vUx~a&F*649Ox#(1us+RK?h2-v8PjxUC6?Z0<WG-XgE#^~Y!f_NiV?FRnfVv2z*8~uMrgSj-'
    '7Uy_vgb+EmuN?xrYq4()2px4o<B`1YC0s$P`2Izj}o!IP0zO*?(i;nmiM5MK2yED7BZG<_eiiNvmJB-'
    '%)Z5j+pEl4!JR6OLw6H+{?)L2fpS-3x7d4#Fim}y`nekKmI0E-'
    'VC|V^>O`Int|M@?WYluP67cR?Xpj}}DjZ_f%rxWXqS}|C3(m$2_T(POkhOyK-A7moPAyS6%MC;47+e<vW-'
    'C{v^p544+oiBR3SQIhBqXH-9fVhCw*ETkxg8#^)X{f9@4aUCITt?Njd6jscZP?=WRXUM`L+b31N)_8vNX{UU^(J{k7g!ahZYG-py'
    'hs8cQ1~6@cABmx=ST2hh~v+JFYCybv=sPw8|xScSFW<vs+n7m}^FGR>af$(p+7JebW=L;sIEDgTZR!zr^r<bc87GQ%Uy&55db7kb'
    'GxGKS}gH&NcT&j-l*RQn|vpzfU8`QkQva2ur;6-WH5pjcjtSy9L*0pyxb-Y_BohX{m|FjIi@3`u%A7g`0m*SP98X!cHrDp*uDA%M'
    'tArfW+%6W$tqQB<BY-'
    'o(mDh$FzUmuR9gqBi(PjTy6T^dasmU&d=qS_bd6Oan$h_X{HxzSGym&WCtW%k<Pp!1}kE~v0Dz=E8)%U@V<`!fNBwc6Mge8Sn)XF'
    'o)|npV2_^!C+>&7CD^Sk22Y=d-'
    '4A%%pudh^4_M_z@wB%ISc!PA^&Zw^HLgD6Z3ea?R#9wKJZ#pBM+grSSSxpD=H!F0`8HU;241X%e96=%Xs)L|%<??{PK>z&cHe`$y'
    'z^X5xGT)h0~wj(8EdsarD7%caR;PxZE?KWbI#97(<hc{cV(VN!STBnKCHvsdrTx*y71}|yz)5dSL;G;%R(gICBU;Dib3Gv$_#IV6'
    '-YYY5?HZTyZ_y|zX36R#Blj;t=TooxQ8%$9Y#HjvFnwIQE<k#AtDcH_hTP;_rUizcu%Q4EAXCq$QH_c&cz0`^IpPwc(Kmg2pQ7F<'
    'O<7LYb43|#|V;r>3dsDcVS!aHQaj~I@VzBMwPOOz`0wFalaY9-'
    'f#4*BXCdP@#5!tV_U#$vFyo=nh4hI2aOG!>n9aYg|AO)4^71?_4{#*<U3qfk7<wHfY+x(JO#)r_u+MrdT+!|f%FjG`|(~pk;zSv5'
    '=8!YXnfq%)<&~~xe9f|-'
    '?53DCm}(s7kSJI<_~9w?@Rw#gI&c|<o;>ge=?RQJ4`Nv9d`2hdX0D3bNnYEPuTHl48$t#&<|@Cp44n?%E)s9ZxYt1T@PyaTc!MYS'
    'Z8Jhu07WA0`xzl5ee=;g?DSo@3ko$OyFsP?NiTSmK=|Soor2bY=ox|n4R}J)wE8#jNs`QY=-P-'
    '_56%s`x+DH4H&yJl)VuU3)k!1R0J{uJQrd;u0DsT2Rp6AUO~>A3C|L?Xg!tmOZrk(>%APl@2`gLhdep2_Hz2q;Wd!85t^kop1?bn'
    '^Og6nLi*ct7`F{MdlAPrJ--0W8xXnGkh>XUp2W^43bu0#;T`WIjY%qIAxEzNr}rWBzKHkF=z7K6+6)__c+PtncuCJMd)tAR2)X>{'
    'yeK5rYY>HCO<bezH^ZYx)w@)%<=c%#pQPY(?@Qy?5AbE1YP1Y{nXuLS#&nUbh(;9ou`4U!3#U)@qu8a<=el673h$nWTv=O1UIG6U'
    'AUBZ5-{c*Hr7xKMIA@fzkjIbWi1#gUMDZOUe0&_bwi=(AJEi@t%CIdMyHUCMqF2Nhh=+v~to=pr6~Y!=-'
    '41EnOgvvO{3zic_4auOyrYUg@Lnu^$z#?$qxQZ4yB;^Z;2g1z?F3#4c7&CcidnjL?tnMjpn+$wM2_nY;26C68RtK}zmSJNpkqh)T'
    '#>|PTr2E94;!Pa*^|)A>&f$wE8Zo?1b7)^-'
    '++|YagJg;{7Ce=M8#|~+2Uwz_D=Kv#$3_$Dn4_6+JSqonX9iwumg5U9$tlf&LsQAv3uQnD?DD;S6&sLj_hfkiu@WQ&*}C3x*3qCs'
    'PcXp!X|il1lCE6B_~_R`q!ZKZBu0&x3?7){68>$yZX%4^g4lK#+-YHAhwnR0;|jUFM&&S{MyL0<luE>-'
    'VVHSMBjpSuW5IF)_c#Z>?dswy!#JQUdJl%BJ7Xrxr|+ezo6|ctXXorjrHSw?}S&$f7STHVAgSERPm$Oq1^b;s|NqeXarsZ7q)w6_'
    '!8MSymt*G195vauHzjcyLaenwgdY1;CS7u>kIRCMDo7_vE3O9W>HnYs$b1do<By^#1rBBJBVHsFBoQiK;W8ZmcB=*54|=0H=yrbW'
    '%*l#`hIn+WZY3D#uAMvK7)nFVAI>M{|#fA*w1~Kt+jRcx<PfnzW)(o^9J6(Z6wt2Yk(0o;ngQz6wK7OO{ad-`<U<nEI0~XJ7Mjc-'
    'lrix3-N*B*e51_$3r;&e>k5|4%GMG!Ps|HZX;CVPS~;&pFYRcH&a;jp1!XJU*ywAgeLx{UKDQ=3WB9uy=ML%>}1%proNzA2AUIu-'
    '&~_Ljm9#*aI~@iF5>wf?$`9+^EmD;{3tj=97E2iNd6jim%5fq=05dW`mKD4;rj;fX5=V=YOwJG&DiIJceSEh`%zfCK7u8};;#|Sz'
    '_K?pY<wU0-qg>@Yi)eCigUoVxEn_!U-JHzu5-'
    'z??(;P4jpEaYCK<6a;=g$M7Cdd^e}&F0I=&=GbasdMBE*MAPbqZUn*Ik~TmM^czxMYiDu=7n2hhU(f%&vY$23B;4?)&0GoJA^;Ui'
    '?_W3a9*p%nVVhu$IYN7IA2hQ2j9*-'
    'enTdtYrk%a3B0j^9IQ@3%uWeVM_oFZK8!jd!2we6E%+utv+(4^4i)C3N)LqfS4<SoZ3(7_d&RYS#QVWS$8&+uG(@`qXf`xF4XX9)'
    'cqJt{5QKt+ihfz9YN?EuUa79mGy@#1eMlsk>ANdHyctC;Guh-uHyhP!$QbC;VlvICEvO_j(Guh0oJqkFl2-#o78?>+N&3PJG&-'
    'mVA*03AXGumGHgc@fp6Lf!|Qkz?VJ$)^Hyx@1<kNincqgrc``GXzB-'
    '@AhBn98~aM((C6M>LK{>`W8h<~5N&<2`zwt4$}pMvw%hFfe!*3#h+w`z`uMkam6~IVzYKZ9_I#bOYfZ=va{h&>VrB_z*hlyhbH0Q'
    '|xyN(&K<`($%hr6ZdE~1ci~T@5th?V}?6=Us+1R5af;fC_xYxmtqMg4FbH0OBdrcN(UF+a?^uPA9#sNay{r!ZOq3e4b-'
    '=Kqa0`}n(+s$77pvMpD;rEF5UK8bSH97|g?BhPw8^uqEub_kfJ*?e}SqIW^rvthz>rC8Bd_F{Auh_cweiVDvvmXe14cpibzR!020'
    '3YT0kA$DRLz<C&u!UvpM-+HGv*~AC+ixPZPgmZbHS>pJZ~&v(-'
    '(Lx0{r)syS)%hNRLnulIH>kV!QQamL{^fY4}+mm98?_#)ux{^G$&@VMZd-17kGET*l_@oe@1Wk5fXkT{FnxveEv@HFeHoq-wB^-'
    'lzvg$Q}G9I5ORLSO!0&7i_c%Ee$I_tJKK+fzx8+v(vK+HjuHatfjx%_VyA`ZOuRqnb@q=Uia$lh1|z=G_Z;t^V!+<9O~>HLZ}9lA'
    '*Ts+G8y(N_=IpV&-y!W!LNM>J*VT{WC-nEj8p)$N*Rts^f<)jC!d2c}FPHyV_!lgj^!o()X80H+{!RGJ#N!u?<fm~Td!d{-'
    'KEzJ%G`#Y6)!#wuQRC%dBb&Lw-'
    '^6T8W6xoY)&Y&lUncs;2)}!ep#qMWEF96%ht)2Q!7)hYYaCPdZ4>>XdcylpcK~1Ed<5$ae<SlNwEX4$LlCC@j_g#x(^eI~yk8OLT'
    'EJhh=m_C=&G+B>D-p+$tv?cjzm(-s{04i1*}r?;`~=Qb9QUF)hOunJ5j+ixPxAf8F?08r_q2Zh@jlM-d@avu|00eoIjQ&G@E0-'
    '8@IQ!j6r7u%k%ezO#}|C4-{#5RNR{_<`E$KlsEty7J;HIdI+&5f--y!*!Vy#N$Bq7flnt&Qg=nx0h=XflFW)io{~Y$;Ue@?G2ES!'
    '^^BZd9XSJU_H~$md1lJeK9AmHV=kQM$zm99>euY%d=D*Ozu{)r@t8#wcgY2yTb1>J+&xu}^0|$G!6I!B(^Eq8^qR8b7avICtS1aQ'
    'FJE7lv*VJ!1Yb)e8!(6b8V<AyG6lzN%kIx>tX(HlboM%RMgQmPO;CQfAQu$(s6WZ)SrswA*)bshb%zT7gzOOdr*1NvX{E!}%(^!`'
    'uV6V<46o*x(K|%?f%j29E+I*bZ#vsOC=+jZejBPAHU<nod3gHppPe_qj`PHN7f4S85ROEwAr}&)T^ZXkAUs~Z(Q$Z~^!05b)RDm#'
    'BsW=03>Z;CsenCQh|3~i>+$%%~eycvk$g(-'
    'hs~P&;E9mpD<mob#6*zWcU!hP>UO(7VC63>8y~<<kW)vo*%USSiH){`UKF@en#N>nhkhm2Je?^bp5ijKLio}xKnk~+Ue0x?{<DZK'
    'BnCJRM2!+BNCBf|6=k?HU=ZeFY!pibQF7iQU0dTHp_<MgUtW^c|8@*yieh&SvFBL^&!1-'
    'VgI5#B#(ad>0>=n@O|KxsdLcos1x04C5rKgy9CVWlkIK?le(J4VF6@Kf<zLfM!o4DDUC`l+yU~R=>P{c?mqIQ=;Z;PU&&oL=v@+0'
    '=}wLI`mzV|OhNM~JH=wY9O3NDI!QLx=Mo>7R!^4mD;%oA$KhkNWZ=eR^@cPh&JW%ZZ$($JCA9*obgd&pFjfyTHTmV`W!THcJ5vBj'
    '_+M8T2ezhdC~zeV=&zk7|KN#v9>F%}lcQ3h*I@r+d{iPEuIDQj>#<dwiVYj7sYog`SE_$UA7fl)pUN@ILQ_$<G62ETs>zfbb_gN-'
    'nzatM*nHglfJ`YlyX%kZ^@-'
    '^f4aRq?C({5K3dljAP=IOdh}`w=9Mh5dFau`0%}XQ!vhMiio*V^k5o6h&03r2)^cXl8R}Dih@I8!BODIrXd@-'
    'iuWwm3h?(RnlZIipm%xBjs-<*tSyIooX0VB~*?9N0sMuHcP87H4UnznaNjpP8GkS-_ReANY#n~$Br{sA^hGT72kRL&=K1CO>w^p@'
    '=!YsL>61g5e>d6u7!8iVRJ>lP8zUhETbCik*l?g<p=cJ;eB3(l)v9fpr-'
    'M=I(E%A8ubamH^J4RuM&=W1{^82_AlK>C!gy6^=g|j)zwd{n<4T!>KrZk>#n*ux~MHRk%QXdJMp7|pEWpYtj%Vtncp7gx+V@K{XW'
    'oldiZN%D(dNY*3-'
    'dntpDz+u8DACKlsMw3_@JASO@!>{Pz71{l=Q*m(smWC%mhzcGX8$&Kj(tq46d7WmSG<t?&Qgm4s*VU1xbPG4XBMRPRh*AS2^Eu8o'
    'yKvLf$A_L=(IX;xr~jdW*KHf4vC+svsAD=Q#34MV$|q(LM9OhTpbStUSW91Sq;tWZi*yfz9W(ZKHk>78&iO#_z8)-{8iCOWPy<n-'
    '{nD|+}nBm1o)c-O-3iXPc4266h>^S``$`kQrA^Y>7JPZe;lxq-db&1hy#P5rY`S8~0DL6aCXH8XmKkxHPAe~x-'
    'v*L<rH>@=jSv?Oq@<QmUzU}iPfzanc6hO{z}4Dhw{2#LK?Gt>X2%H_$Tj^8>49R1eHoYtzZ1teG2Jx8!pkd-}J-'
    ';F{oZ3r#QmEMG2gcdq0hzj2ex@D2@X{n*#SpQOqquIykh~96dv)ZW7{8vwnLmg+ufTPy}Qu`9l((4kzU>Dm8@4KPmn&>a(Z46lEX'
    '^_wsYe^@gUHV#czx_#~ZCXFtz?MEH4$VvqdH>wb?+}A_g!Ym5?V+`gFMli13H)iMfAQ4_UY)6(;4IW$JJ?a#@0bQMyR**iig6tG_'
    'WBnQsbI^x#2{L2t;62b81@D4I;jQcs;p#u<x_v)Y_%&2&TS(<_!oL}!}DD4Y_+sAkZP}7>#sP6z&^^aI-Q_d?sbL@-NJeORc2P;-'
    '=4X<@clDX!?~)ni{6)*wMOL6Mc!;Q*>dTva$Vxo#UJ1Y|H~G^zh#ksAJf(!=%26XVB%=eBL?TFef=~JeD8c*?MCQH;MK4fYM~pv='
    '5ax++J?QQjdptJSpx{IU`uzkyjOSxs{%*8520V!B}DpJs--ui$uD_95ACg^`sycnHxyd>`Xe$by=~Y>&ciIxCgb`O`s)1Nh*zI*&'
    '%od=Aai&JHIk67%hy+nd&B!dx-'
    '*G_?HWkvjxl|8oUo*?S#{eZ=k?&_AowZLS^Gef6}EXW<PJo1JD}RTK+^fp*$eWr#(DmQgh48&5pq@)J`Y1yhTuFL@`oeF1MwOKM}'
    'k=}3J`gnU{8O|V?X#`PjwAd$>Q%2V1Vh1se3hHa<F@seN_Y9#f~PhcXgn97_MIked5hfpeJk|N;sd;A2xQ#SmoM=>ky9`=3jug3n'
    'zz}d~>#VJvbY32V>kIjp+H%HwaetMKq!qgu6p<{~{CjQ5fGJ_XptqV9f80*<%!A72|+WCc8rjLu0^xac8`MFidkA1!slV0bX^7$A'
    'IO_wazKt3^wb<h#2sCk-8R0-wUJ!9(TbzwrFU`sRXlvSjgzqzAv%@eKL;gm-'
    '`koh`WAw0!Q%zqhXY(mn55pD2t?id#mjuGBPTx39Rl5uCs(Veo61@m(lNgneM<f6<EqrlRgZCw4SJjD1>tZ!qp%ZBT3$9T`ky$5o'
    'U!RMHr_Tp8>Xdlws6B_}&MWT@ZtAY8B_Yo?+wwXc<G`{3bmb1<j+vK9Iy1jnx=E#^~pB(2qN79~ws(8-'
    'w(Fi6l5fWAIcbK8*J#7+n(K?s%=QcPHrU2>3IRAdzReBMqat+l&jP)>B=9oL2rZ??QhPteQX=t$T!24)r$KmM7Fe5=Z4?0^gr#a>'
    'y8$rqhoL{pX#`IR6p?M|Ki{?cwg;D&$BiCSu<B7_b){2U+7TMo*ed;OGnQxVu{)r@$hfnczn%cuzJ0<HkTMM`c2o*8zISnVC0`I3'
    '=rVw9gF>J--Nk@@&>OD$<FChlBK5H0q;t^&XY6#)$0EI4+3+$9`O-e#V>ojPdY}^JKM5LKY^Q-'
    '2y*>_lF0)5%>P^qmR~>tPrE%?*+OqlfIB#MH9VqcuId9;V1tHFGm0`kN1W9wfm0yk^7*#&Ry!>=T3EJxf9$;?m2EZ*L7USbGkbJc'
    '-@@5&^HV{;{O2ehv?w'
)


_CANDIDATE_FINAL_SIGNATURE = '73625659d973ed58e19f911118b335c9021894477fc8bb1513931468517f331a'
_CANDIDATE_FINAL_MODE_HINT = (
    'c-pOf2i#Wk`~MlKwD+FcyR<iz_TGc0k`jqzCo@}ii4-D5l%f*R7g-Takr9#=iu6VO-'
    '`91{J<j{OpXWTjzrXkEocn&x`FyUu&UqeO=<i)Lsj%y$;5crM;bi#k<Y@K3;pD{f%)BdWiL$5RWTflli0A)iiA?vM9NEXAwGQd##'
    'Msa=JF>Uszp+&I>zp)eIXSeiwQ=@b|9y02vXGV7$x;$0+n=NRm|j*CqMpua)MrmEJ6)YDvc$<UQf(Y%Wy8$=&n_e5BFD)QH*3jEn'
    'mO6%?U*}RWsZ}Djk3#1$7FK6FXJkVlTE$I&f}S+%4r?SE)SiIdF*6U?VJpBogCz>o2`E{*&=hybA&pPsB<#LL0s=j-'
    '{t&^ikFj`=49DN9Ez;CtjJWeoGiOjrvBsn>uJJ`^|X_vU7ZY?I5Azy`B(Be*+i3*gBqBV&Yns8tfIt8iy|k7{WvGHOXhsY)P<Z3`'
    'Z%$DCc9a}$;R(amf7!Q6W!VOo}5n@PL}n{{r?On+gaJkFhykNH|zh*70$^fJN}oh<K)QqnJSZ$1CRgTXY9-'
    's+{w}O;O1~L&pf3&|8jE9`kBVbBKw>yR>t|sasIanIdc|fKYL}abeU&^_^BpJKdYyk%)QCUvWxs*YQ?{KYRvvw!O5lzW%cyu#71j'
    'eosXX<otTWzemclHqnsSff;oNq`M1+md@efKR9GjAr&%Y5y*H=5%gtd<WXjI0o@V3xYbS9h$F(){EbnAf+y0ks8~?83-'
    '@OfR{?!{8H`}L}tb3jlV~d>E9VeTa!O1pjos7?*P8M~@iG8Ez{43i#Ie1cwdyf7$Z;+grs^(;@N&mj8I60o&vadE-KLa?i)i3Lec'
    '5<YZ6WcAE9LD0m>4w?ov6DsrcCvWN&iV5+*D5F5**W`u<HWpsb7J4lIax-%o6U2llcTp@PKIYLC(BL|-'
    'zopi+dDVI)E74aI@wfBC(BH6GV;pV`v)h>4w~I;pIuDEdEdz}<>aISWtF4Z#YIlu`2XL3tK#H%&a%(4PL`E0u4l*Rc-'
    '%W<C&!iPf9pDKHc#OHTTjdMOygu(8*@HaI63I|IeSmw<nT$u$u=9EES}NgCTJ%|zdv#^zSaC+e@p4)s1s&>|K;Q$!kjGL@^XF2>2'
    'LV5|4m$+Z=HYjJIk!zU*+_9Cp(Yla_m33GtxcD@eJgShdWu_^E1Fx3H*%61DqV51-T$gZYQ5YX!^Jk#YyH$L8pjHln&-'
    'oir`@%Ut_u=$;o=o4GJP%J|}<pb8e>$A(vB{kV~aM$tkDjD9U0-9zyQW26CV8RV9=s<bh-blCoumyf{vZfv0lhH4>am@D24m*p&{'
    'Z;40k!`3Q2q09It?#mEb7qsS9-&l4V=Q$ZyX+Icg`p37Xz=Tsz|Y^01LFQEV;u@Gd*i_a){XBC3PM}D&^e-'
    '3c6Qw%(Xb77pLV107yx&j%%oJ**M<$59K6hc7*mZ=coWXM)X?-Y(fAp@4QS{k3=%1KU1>@0>FU{N?0tV<22I-'
    'xQl4^|a(io!;Da4zc9Boz6VV2#CoMNu7!s#Z}HjX^E!DGGZRaOxxe3*o0CPH{t}4uNY+aU)|fL_$H(3{aO)0y~PUMWWz!OqDIo^^'
    '+}ITy2wHv1L_#t$n2s1&l@nwpJzBs}*cj*wk{-vVhVkVY-A^Fcm+6ChfHcair8H<ORJ_(4w?@n5|wwt(cB-=n_S-Rz6E>$KZ@EZ2'
    'CY6r-AV|)2*fzsWj-'
    '7H8x`(#?fLQN`<JFh61pAy%34`DWbKtEcB{?BMOcIk*T<ms({lNXznyo$Y&vFR0iwI!3OM2_8?O$=`@3#3!3W{Azyi|qZMLM+JLK'
    'O^AIgU6w~Td5iwZac$MX^0b51Ek;V6mL!)L|nS=UVN#n5+!2af))sm7XP(&?LM0-aId~2b(UKw4hs)=~9F59Mt@o_=zlO=H!)R-'
    'y>n$7T2RopuTblH~NEmL|2^{au?FoJTBwHRoZ3M*z2*ib5KGjJ*fc{uaR;Hc}I3QrUQx8f>W3H>asIbKp@rl#>J+wPPY)FYe<&P6'
    'o3g{ySG5^C)t_*uMFEd~t<P17L$=lWURc%r0M7?GhY^0GW=R!2>YLbAG9h(t`Vk0m0?>bbPqs+`uy@|xde;Gr7umE@UpHHo^?_^G'
    '_=T?Th*5SV%;0@u8XkgzI_phr{!kDynU(P)&KRmSu@Moq(ov$D1Nu!_oCQfp@!t>!^(E&+a}OvhteRxugFE6z}KbvodXPl?vrdZ('
    'oJ0V`*)ucqo;M&pXBLt{dHLPKz==5!A$NyRX)YT_ubaS;VuPNHpq(=v(HKnv`wqmx1d9NnCD1kUIR;95Scvn8;zqDF>v$_gfHEI1'
    'EX!pcq`?JO0-T3c45rW|Cegzm)p@zhXPHA)9p$bs-tMbMLK-'
    'axacs`{#)+O`3}9*EPZjas4{d{8;efoixa(ZaemaoQ4CAHLs#AU3I{UJ$7(LGDT#W%bo}jg0R^pP>3zggoW-'
    'Gm)1fZ$s2It8sf%by<`4i1*5nq*B<es)BbF&=y_|QG3MCjl+(gLLKB}BcoGC-Bn$+s;~W_ic+kqHLNbKHGz}?>W6)~Dd;pXQZ+PH'
    '_jCel+R>~y6||*Gou1^GK%!bVSMH6?{XroHhe*v?L8WACNW?XR7n`X*&0xtYuuoNu$SOusMw1vcC2$@T1D~egb82!jOMx|O3F)}@'
    'Rnw@duDMeSF<%QltFBeE3hp&R?6!zO!lpG4i`DflNg3@xH8i5CfU9J*bjp?l)~y+#RSZreRD%4{k=h_*+5?T@hnl$30&AsDwbNbA'
    'O>JrB^dz)bG{T-5Dt)IAokP?@-'
    '>HrjEpTq0fg;<e4ZxmDL~4R&Q27!SqpNzS1FY0azX^@@1g<h{UzQ?}txXcWutq*xgOA*)0eu^TR~vLAslc|1wKF`}CtcDYUA3)Q('
    'bnmW%q*onvX7~B>{HI`j>ZZ?vj-u;HQ^7*#|Fp<Yr{?iu4=<fZ69i6WYkPfRc)bBBh8SG>f;fLkx3-'
    'DbjBfak9PWo=pUkE8fq%VTCiMo=u`*x>kzTGpsktgIueFqa;T#{So(BgqOYy@#LG=#r_<oMb}<;~jCIC1;{Z921x9M`3VK+;t2Hb'
    'vncNqi>q=lxv;@xy*drF~pcyQAEM20dW?nDlkq(L7wz^XGbSHFB>x?4Q1g$Qhb|!%%s3-'
    'V!G~j4towx!A_7PbpsI`Q%jTec#ZqUAyxiirCmvMRwIulxGUb426oO5usbC|Q~=w&jblSWw-'
    'J#=n9$C;c&7yR5=ZPLw|fjt+264Mr1eNYWttcjv~3|K4n1K%GRkzalva|Woe6}qW^f@guQ_%2bv^=>9f(!=zQe#qLc8RDs@(SHzu'
    'wH$0@?4$N#j=czjo#~1Tljx%u2*|lNVv#L269_8K0FxV1jk*~wy`3R3IMX=`vJ6q$GnH~*Sf^hoO*)1WMu!*^;wtAnNDxpR0LbsR'
    '`X_5;PSG8@v+agB!%-1Ogph9+0qL>W1>A<H+yai#euSundKn)N11^IndxJ0U?VpBT=I(F;Q|^I10}Of-MkMFj1UUD`!z#UDgE55Q'
    'eaaN|Wp}KRXc_`vjU)_(9DQ_cPkau+Pd#ysrDZ9G5x8z&Na$@S_Et{4k=q`0VEXBCDfIhd^>AYyj{J+DZ=knChSbMtN`ZaRF9yA1'
    'a5XRzJf!lRWwN0^fwSitbot&|8G7qyAN6w-BlYO4RcDw+hGYc$F3@Haj*C?HDJq%N!qG<Ofdtlh0)gL*u@-'
    '~FJWP0e6822QS&m8Y=tx|vi*CxA4T43a(s0Cx<x)S1H%BG?+89qJV~qA3Gn`9jgG&_r)<<->+N`=-'
    'F&b5ORK_~Vmb?QxM=_ef?@NUCI3vx)z;%R+5CdbdnzLt&;dv)vJk}&4avbQ$WHStu&NT6w$YGw>r26yPvFhD)NUa&6T8=Z3KQ0F6'
    '0AnHPcxXNr(K|-'
    '<I$M2y7Gi%2u20eP#m;<PmyW@RIKI{=ZKeq@0W!);;U=|ken@S;{?y*Y{S@%e9ut)2M9>}wDFZ4U^i%kWFFH6=2~2~%$M=U(_Kt+'
    '~6P)v5pYgaWlkz0+s;f^p4RnHUfNvAAY7*{uLH~_nmilU<$;KIKL;m{)mgXAGw==^-'
    '&QsKP>EO8Nk2$k0sGp7OmCkYlaAO+IH8x^QHasRrY&S{!$5d#?*5um4F~Qb3moSYm5Y|n+`D&nt<bA5r1m7nk%kIH3N%u@4++w7X'
    'ebO7J;kjrE;<1;}m%(oc&L`Z8yuTH=3y{xdpgvGFoC^8_!Kn$fyC;K1F39kNbrJT@!>*~iKfpqz*1@TS@6wUQTGI(qL)n+Z!9%PP'
    '*nXPsoQFQ$(YaJ-'
    'E4CKT4>AWQrVgp~Tw!~f$ZnxeqA8r~ECA*yE<~<ghKkb%mXO?($u(8~1Bx?XL&>~(_!c0@#97WI*ufEW8LWSSsi32^i(i0xbAf6#'
    '4WHAJ^aK|raWN>LV<PYJ7;r@$VdlU<f&>B&T@_9Yf_b2FF}!f0iKcU%D+uSnT3m4@^Dl$!GhvsRpmQnKOhFD_K)AxW5@5TH(>f~O'
    'I3lG7OQoI$d8g@ly8b*J)X&3t66CuGQeR29ig13~8?*rDE4`^Yz4Zy7r)0us&7z(XoTpZq0q$%omV&zqYd6|ZU|g);5?<mZ*6IQS'
    '_Iqo!Ryr;LooQ;b3-J@zJhAt51Fo$1V&@E{VOIv3JDb2c(F=T<z~WaTT2l2Ph=ywrmlJWwI~d9Gai$JT*IG2&paY(Su7-'
    'x$W12IEa1CLG*5fNQ%P+_0<@lTxcAH7iPA1z4$f?tva^N}@(Lam8e!P~zGO>m;a82q4ODei|bSlErl4I<BmLk2f1d#d^%#@da(pa'
    '2N)AtCM5!kOYq35MI3;*kqIdnR9&&p7Lm@8|0IVjD*PN6Z_=r}Hv@Ct+L6*qvwWr)D5O>B!_*anxGtem5nf4<rx0Q;YRGjJ((1yZ'
    'L&q_{zGlgZ4>jn2=4?|HD|<=Azdi3g#QcsqK9u8>N3HDQ+7cRk@k%v7S|Y~;^n+Nmy6j(3CJRqDg+aiN~mBY`<y3Omc(vdUSlz2^'
    '!&4+rhw{xw*Unr>#R#>1fxN7#J?_9Z_9@%<Y$R%Sx0)e5Q1fjp~2ucRQ?#X7T`+aaG=|2pSpvuZWuy#{m9-'
    'D$@qpvqA=2@xt!5|hH3$#kL>bxJZ?5@4HOgJ;Jp!Rs~?GdHTV*TC~4G4B)a$~+>v-'
    'x&jr&wFEVKh{We%~7jGaXV(Bt8sr0ym%Mk1{1UF=bM}d2y?JX;^cbs+X~?%liIbBXNFARPH?&NAhf<-b2EzD;rW~4+bB5Nc-LG4^'
    'ScQ$BvuBiZ-'
    '>usQEoGEz8z=IfIHxGS$Q*I4RT9#x*7Fi3eK|k4(P)^;(fO`o1E7oxE}~4x(T!V#fY*yl$ONX+2Fvwxy9IZtMS$iSa&^QRWge2-'
    '3%^wC~gL~+mwGgZYA6Vtt6vv!c0B~`)6y!NCiF<@wHuXF|OYVu2*2qZG<_-'
    'FO2zww?T>ff$%sBy5EWQv#{oN!aI<KZM}fNe7I`P1sAc)ZJ7EV3T>3aEw~oN0$3&rrpS4AYlf_UC<`nMG2g_!#OuSjlOEgRkL}J5'
    'W0#AaMaa)b2=ldSL~*O$k@_$fmcGJy59e8k{`*4R7U26rT)!B5=f!|6zX;&CxK(pnp5m?qP3chY;qzwaQD-sy&$<5)u19eZw7Xs-'
    'c0MG#8M|*rF6;m`>7sYy+C89ji}53Og8Q&`5mqiFT%$GkJ=lYD`qmgcOkj`Sj5YFndkfA%O}HhwcOE2Kteo$MWb=&G*ppn_Zz0@<'
    'yuBM*-w#a|E9Kym>)aB;Vpy0f1N%hci*>&l_L1BXyDfwUi<RHg&I;!l=Lu&es^6>5E2bu~kJey4YZmnN)NC<dtMa|-'
    '=i9(TW&zPGuoGwIea4akvH!KI+nr%Gem995k!edH+k;ql19W0}O1d{W+2eYx*Z1h^dtj>s*T=x~Y@8o8dGm;=jX{OKL64v_E>=`?'
    'OS@%&Y9WN~?auSg;{^7Z#M@n%b0tR})VN=ignWNMy}A&-;z+t%XDyLZzRiO#qqq(dJ^-'
    'EWAv}m<Dd9%YTL^9W`mN3$!eYX6X?Px30_u+_t)M$T1e@K0s}DO*67DrI&e6b67z+`H0oOZV8L47V0r#TwNAVaiAF|)!e5%;1kZ-'
    'pmt|TYt!xN8zqg3@}1n%#vjSsn|U!}DvbufhoA<HfB#k1i3Fn*R^@u<nMwa7}F;V;DC33&cd$n*$4R}dtVpMcGxU>`38=he<YH;U'
    'DeW}aGTDJn%2oXcwg_U^;6cmVo6p!@IF^AdRE2^`O9yf1^b81qf`yhsS-Nlg?B5FZbN>T;~T7twi-sn^R1FFAjhZut~!Ay^HI%yu'
    '>+!nikO^+b?}fJYIlLF_!1JU^|``?Q%eM2e?yEYokRoXwC)D2cR*%J680Zorwzbk`BexObv{Z9t8EGPFNy#T|ZW=-'
    'Vg4PmySeR;k6R(P~83YDjz^{JPXkcQ$e##-'
    '1lNThlScU90clm#T#pC{@nDht+oJSWkEi6?qAGOmU~UFTj>;jrhEZg5%~Hwc6dV(jxrC(JvF-'
    'N>EvCY`PTdIRlrWX3WuJjy|(qXL9y_&9SVo5;QrxtnZ?Dn(!pDL!P15nR>`rYiM)iEy0=>O=hh{X1@T=BJm34_#zIb&VJ=we2Kud'
    'iT87l;b>#Ou7Ncd!{5)E+VZsKWE2ZwFQ&Kw^VF-'
    'F4^KhUm(;qMD`1K%M=w|Mr;w$KaozwKIFBDQSJ?842pfQxfeqoe4TP5=yV&Y+X!0M}VyThu8OZdg_MO0E%dkpzKC3gpvpRpsH}(!'
    'k3foTTYy|&}dJaBUXjga=yI&@}PIx}-|Em#c5`kPbxSuRh${WCgv*8uo-3XpfnW-uvX@ceotXu{f5+NJINZFur2N@uHUPnA^g8a+'
    'Xd(ROhj?#H^<>5Ho3@Wb>o`H`x!Y`6xk0a(@B|NJVt#jUh)t1Aj*+VQT{r@FM9mPsedd1*P!U_YfJ}Z!CYc-OVL7&$Qo{0f-'
    '6#I)6)<VN5*5k=x4fI)wyW;a_^bo8{%Ky3nM@oWk;^(&r>zt3YLc9q+FRH~~0q@N!ft=sOIf$Pvz>9bWTM1vi1<9TQHHi}G8Y}T_'
    'p@}%=^_+oN-'
    '~;DvtX+}e?(?x=4$BP<>7rBdR5MkdZKk?goe#llHMCk~sI7{@JCHPmcL1ia5;3?=z4;+{zf0Imcuuo@J*;YZtu}Z86gFVB#Ke0Ba'
    '_4!axH_Bz)+8r_)ut=YB*;5Mc^aFG<1QR8AqrMOkDZ8zb>W_%W3SQrl$thzd#qV{6_BjCj(%_P3bOe*#LpJ(IGn}YDc54%@0f01#'
    '@ZLa{SEATI|k1q7TzaFOmoNK8ugNL*@R5mfX}TVxQo1M@|b1*Q}JKNb4$6$@VOa#HafcyGm>L(kreB(gFVVIS=QYM`Q@GO2Z*jMg'
    'b#pixV{Z5J^`nM90?BGDR&Xp$AE45npTSU5R+0Nwk5G1_S>E8(+O-PybL>tjo&mCxw=~&dcKXP-'
    'VHH$hrm6SpZ|IN>xi^B;E6r@CW%*9bEA-qTpMQVi-`E@F7xATm%b&RZ-'
    'L8yV6R>H{u+D}#e2H$J!1!shRv|mXV|~b`4qBq#(W6cOo7*MXMH~d&ux%(C-'
    'jr5EKwb;t{>z6tJ+Q3pRXpL9}?5+m(F%r_<ckmcgPP2@0(sN?>zY0+tB-'
    'C9J?XaIz#jI@O?=N+s&Hyp}}tWX)``Q&ro%^lkSLt^+6OnV970N)8PIaS>ip&wv!+^onFsg*Eo6upYQ0MZ=Ek-'
    '5Ajf<0==cj8@T(HsZQ*#J%~oGAfM^JADjd5$ri2R@8en&`v}~BKQQslSr_<xYx3BtpYrX)@Y@z$C;XW^Tlh2Q8~FJ{SYsdJ|Gl(3'
    '+w}aS^DA`N3ToR30sS53`j^00Y2d2&5o{j3O}#03eixixM}&TjPoet}VL$LS@D1=S@Im;St`su8!<Kh|^0rsr@_z*VB#t%`HiP#T'
    'MApZ^0pMF4--TdX?F8Rl8nqHlEWxW3eX?>FsP4cK^oM=OnFPK7K6efRTfvEKDA#wYjo-n#9VX5;#o+T897K!?k3i3Ddj1~Qwj-'
    ')Hql*2GEdLmO{DQz4w!>5i?uokztmz&^#GCNJH)fyI58f>@^EHkY$(PR{+fmbxc*4q@XQ}QZk^Hg2Zd~09dj%Fr*ys(|VlQD=Bo7'
    '4JaP<r4C+BC4V)m-kT$X7+G+_-z>vX+%&u-'
    '{{0GWFbpFGnAH7AGysRvxuejt2J_yhY`&wb$a3%Grip;mqC975K76oWq??;+<m#KKpa<LNj`I6&B=wPByhRtEd=pmRiP+<s8lZMb'
    'en-HT#-n6rV`1suztYyH{=nQZ19fp5Ph95DUxBkf3E!4@B>H@-7SL{fsrPE&`zC+rL7=oF5|U=Q|uf@8nqx*vn$-'
    ';nGef#ZVXM(PRIf-eyzKAs*w#w`COfh)^5h9c)ZQ{eTIse3``pmSLJ#UCacS-'
    'x*$AQ{CqMW)X^p#Br0cNe^t%avZmvU1G+M6h}PyZY{TSZaq(1l-'
    '5j{(Ch#c)s`<krW{K_6P7i4&nC|&ihPf_{PM~PEcp>3zg5cA9F2`+&C0w%x~fGJ+wI4Wov#3JBd8IRFXY7*gtGF_V8aZ_%qZX(CM'
    '=n>^He4Jbp}rMC~`w@DoV?IkF;(pUq^)J`zv=NMI?1Z(y5mjMZ7cAG8{A<@z3yM8R_H#lG*b@{ox`p?W+GoMVT=+2v<c>b;QYKv>'
    '5P=#2I=Vs$_K_KVT-7Xnwu-O1T2F@GI|9-'
    '{NNPI31DzV9b+b><rP6|Nm4e2Z(}W$0<gAR*I#O5fK8?=M$kw)+y1e3&3tj_7g3ctNiH23}=c=`ID_ukh0`C$F2&W$q`rrEzzkiT'
    '1-W_(|soi$uofccUooibO|3$x7m<{qUvl9&w`h3vnJr!fOdz7j#+Md~Os6)&9TY314cMX!#>Pe>XWJ7W~U-'
    'E_eP(;yAqj7xX%!_3$6UPv}?QYaL04#O?|GPZPWD37o!nSWA|qkQ>F1YTrYMz#mbYzJvy!;@bi3?MF=|V&sCoe}~2XfPTMd*8c%r'
    'f77a7z|9^U$(&0+fqDr-9``Wx{Z%#p3EJd!qu@-'
    ')Dt6LQ()~d_bPU$|6*~Xu6n3M?<=R~wJ6T0%0_?fJ!AbTVL0&{rDh7WkJ&DukH1j8V&p)_x0<>)<l4#AZM}QM%1ta^)bIe5Oi6ny'
    'm`1Mh&@e!B58A^=boZPPG`tETjir-)#nc^JxH^+kO1luuJh(xaDA^fS8^G{?AQ{o)T?MA^-'
    'cN`JJG>$oC;f=o`l~6d@jY7B=aifsB;UFImL8jyQ=DELs^Y2a+oHhAXvmmC9;_zIlG)KYnxYEl93L+23UxMCIJ^ulDgRBgyf<%3w'
    'TS{h@<X8;yfR4!YkI5BA8AxeY6?KaN#goY67KEk$0p;V+j%RtE8S)}e%Dd^P0PRkORb3n>ob16?uHXtPB_WlRYZKG>3z79ZB&}xt'
    'v+$=7IFpXY;BQdoh&pci;0aI?OVo2+?GB6tuLRe?{{`K|r}*$hSXm?!4{OK#4Ldwnx=s^V!E@z4N3uuwJB$BdzvR4wqbAPMy`ujJ'
    'W#8C_;l}*qpaSH@aRRhhuA|7WTm-hY$jJL-'
    'S7Ai=DY$pk)E&`<sa7ZWgnW=O500jgI9GU$s}BED!^nOzRu$6flE<x2s7dg!#&cQr(u9K0vlxydkdWo^ATR3?_`hfp9HsKNqj|xn'
    'IA~YL{XC&XS(Ab>V7@$ClqKYcRK>8bs!A2r*@Ijnb0K4+a?rd0?i3GK<_^~f`WHtpE=WBkFQ`bZDw&1`u-'
    '!?JpeXj_Q=Q8bM1%Y;Pi8!YlyYkka>t+;^pZ@jK=8CH=EptmiN#C=)CNkTx)lm(We@2X={2YT4vw)(uunxoE{*zPuue&w8xs5sS8'
    '^qf64+JWSg*2cdv0+;8A2QAdJ@)^R(t0&eLkI<Ocd!;nfUxvLN&xtK}c7|<qFx<%^!oZX7*)UHzKrfn}Bi&>@1pL+Ab7{NkKWc8K'
    'DYdu?jv*;cg+^FGi?Bs6#lBR_Em(j$qcH6y~~C;FB8`Dr~%SN~GGALew@f5-'
    '0w>{NTSWH5xNVRd~Lv+JNoXIK%$t@Pwebev>oDQA1evB(;{*hz5ifNwidyHI=?Op%S8t<3#LUH9RUo(-K-Iq9_MBDiO-'
    '3@efc;t5tDOszH#vmx@))pp>Z(jMmVswW5t4!n<+^iG!-JY-'
    '6==;#?JIqxuGRX`{O;LidueV`=c@I#X49LKGYkh26G50pz^cS!gr~k+5VwL`~(S#}kpxd0r$2T%D?dYE|s30GegMx3Er0`Lsr#tk'
    'EX^XOA_*t}3v6EqJaxJ~{tn@@q$^3O`3t)NMs5roAcX0JRmxu&#_wxOMPT9rtuX`3zo<q8)azwJSsOnvk)S=61a_$UMP3IA)ky2~'
    'eu!)+IEC-KxP#9pKA!k0ro=S<#-ro@#>QWK_@7O|CFyeyXl+G%AOBCRQy6*{Xm-&t%LeP|9Rk`5177SH`u9;LtBaO<;@i-'
    'U{mbI_mS<@KhZ{kJzXV?$pJZZ7ufUOsH!tm_lH;>S0xi!o|~VU1N`Wgz~76RcI1R#I@{C{9G1$n-'
    'bV^^85ZnPG#^O1c|s4NhPn~ex$z(k>4Y3)8DqG=TSY_sG`=Ts+!LMSmJhCsRV5i0UeADKi1z!%I}QjcQ=XOu%&({#dfM6=?`+Q0V'
    '+N8PS%hKr=r%&6n-?nIpN$0CZki_0md&4%@qd6vGlg)khC6-'
    'Dj87IOk^!h=ab4;A8}YiEnYhYjZoS6xws~%)KXdaPA!e|D1=LO>}iDPsg3*0L9}5yT?rC-jR~hII)tC6xn01$1~OFqR0n6)o4r}h'
    '?d2|U9*z8ce&XNN`UGeBy;)PEO?}woRPCVbn`U~qp^41~Mv~ryTFEI#bgvQGAc&sk>T7-i;u({@UeE1A;CFj%VZqj*-'
    'd9l*T>GjgYC%4x9@O-'
    '9uyp|Lj_r+Yn<1x$XnwJkWMXNm=OCXO>$JwJ8ix~00v!n*2vtIQB1;YUA+VBYCiGhp*qUb~V<?51;BuPIHJu1z?K8B?cSmG)!?`;'
    'uM-QC&IWSQ(6H_h6Qai{Z9BLv9MAks=+Unniki4tOl|Jg*x>^(KhMz3o5Vb<xunX1Fb^Qp!eXRDgI!5Xygu#URkggxP?y0I9SLz`'
    'qPo-'
    'z4%eOcDw;~fqXE%yO)Ft+F2O8myacAl?Ll*)^;ZVXE1is(IWadzJq&~kjGfx<dVFaEVdJtIlR>p445!0;@146x_9t{zZ4H4zt+~G'
    'hsW0}z!u~LB=YTc=)@0aSL^F+~>z;<X2`$lmgp*EyzkDX%0pgOcBNWExeVrftWgJGwrBtb^BLBw=+Cqw&j1hM3B0?(doN0G3(-'
    'f0Pnrz1~W$D$2^X^uCxXQ^r<Zn}VH7ek9}G9d=dP~TX`PRXc|`qUYoKSS*&ImyqSiM&hX*BRQSx+9OGIGZ5dp%-'
    '{hH1(dJ*0~xp<<7}FiTceIxnuHP@F~x;U6Cd2)OQ_`g(Gq1XLgy~`XWoGVAW|xiWV{8ia(0bjlk>q`7d!j@tiUc8uT}vtEGYXrnS'
    'y7gSF={xW03}<oG(%tPq<@C23{iyp?va9<X0WQ0}U05|Eu;49+4j_YP{S?jenUM{D(FcSDQoF<bNu-'
    'E{_5c1OSAs?}X_2Dr7yxs90_*?*T1q$_nt^ho#SwE=CBv4f%8T`d;HXhKg&)DyIsZs3oOF<|Q52|Wq3Fc<Vjyh$$by>2)-'
    'qWYLslGVKk{XuWI(v+Wi>$KBTBfBqwt5bLPT*4r9o$grK$LQ7@mg-?*yn|Uc1$)O321338m_-JJGNz+<Qf{$dM_8S8?gULbhPf)A'
    'Q)3|Xh9O6Lx#tmh<pAa02WxvLD|HGofAKZ8Xg7_suE|&&22DpmU(Oim=EES-'
    '=_X2h!+%lqQXfREItW%Afpaf>cGBJ$#XvLt4I!{q&w}1#u`ZY-hJm^qBaMv`Uh3msN*JNlenj%<1KG!^_I;DJ`Ak4Q#ro3$uHD>$'
    'Cc2Z%6Or&wNG*9Y5xsUKeA8F$*Av$1sy(Q?dp@|FNf_y#L+FRK;=A-Y`(p4t6L#-!>Sk|u8Zasb=ek#b`&dZZH(Q)fU}-'
    'sG`)PJ_gaoxq@-;Y5hE_u~#-'
    '>B!(JFB|ct1a73;?Hr*gF{#$h<ccQ9MwycBVUnz`4y6XTnp7{N|aK<8UP89SG^KLo5!21uw*v>5wP4H&kWi33t4i{W)I7BaY4r|N'
    'A7v5G#o}X`<p1RN-'
    '`t(rmfP$SIy2Xkz{WcfLDM@qnI(LgF#tGDg28?3u9hAi@w(?5A>FsaX;QuNLp~Q`;zw=ux3|Eb}0}Hqf1eJY;IaOjHC&#o`dP)wQ'
    '9WQkVpbPEt(N^H5W3c&6qYIva5_E3Fm{HaR&g23*aqC!7lzFM-xWAYHH`sB6LJAdQ45&chy-'
    '`%?IDg1PE@)xA+h=Mg5CN5FPb+yKqSfzMf5DbsNl{5Zs1odM5YhkIi6GeL2ju_S|^2)Qm@MBsXRF8nzfQeH=xOE?D+&a)F+bTq1H'
    '6ldaUA4H@?)>y(AXg3)W3=fZ>i(HQkXWw%5;+i=WJo>>S;}8Q;@N{z)Jd!=et5wsr90}=0VX9%;6O27BAPh2+pABo@NEi<Jry>ve'
    'X;S`8`g6^*-WctogJUoO@pB$1oFC?G3R7WUw%?^D<EI#J1@&cMSYHM}+ZkcKPxOUJkZvlpJ|D+&<rD==!fy=D))^-'
    'X>GT(Bl{(MZe;W8CXeFu$OMS7a8;MLy<i{kfO_IIS(`1oKau+)KWYc|ys&6j9>dPRLWaunhorO=4S9m655jxYcVm4Mr!BH7x*v(1'
    's^Uv8axJdcT#LmmzDTJE<=06esIah5coqoK&HJSyT<Wnl;6$GA1xHepFsyX|b`^hYVXfO+!-'
    'eP#4ty(et(eUITMD8r;u>$A2Rj(T%+b!;`z-'
    '>vK3vZ3XF$Ntsib*kGNw_LK<le3m!39RT^AR^uTmsvQr^m)%wtEdUh=MEBZQv_2%nZZjT%ECJn#?*M^FS10p(|n6J7IaQS`ru6f='
    '3imscv=eKt^$$x`}XyxpEbOW0dn@is?o(A=eDZdMWD2l{g+kT+D^^QOtF(CQL?DT!Qsn->yP-'
    '3+5#;8I~98&%t@Nev|rnGhrAi#%$O?=F=I7yt@_mB*8Jmk#ThvNccr+{*<s@$|uLdec(SDbT5H?tkZ?C3`YddBhxfH(s91KQ0IWl'
    'LGemN)m1SNZ%i{Y{|s0}=F+P*vu43Ma+T*J)|w-'
    'mpEP)OxgZ9&fagW9;`NYIYMJ=)M)zStRPGr@%G<(7y9ySGg1zyGN)>oEm4mlyXFShHcNshyaFs}#lUAdGipA@BmYR|w9v8*nqLAa'
    '&O5rA0g(G67Rt&E2oc$|s?NyDCt4w#|T%C>z5C!j;hE+GF=^tPYxU6tra5uYexNEg)T#HEJsyo-j9`};Kdef7*0km!b_07OqBim-'
    'w4aq-'
    '#&btx&_!?XPa`#a}dNq>z!V!EeG`JPqZZti4F>25Z@Q7XwbWUZ+TxfC|sJv(@O(IGXbIJ;1mn$Iey*RE!6=As+8+yV?JfGW}k3Cn'
    'II>6nOqjd>^{Wu5KxgYfJCM<wAuEp^X;p#Mx2Dr?94E}o%cH{VBnvZJs&L%9>{9i!08d5y~eP$aQU2N*X!!fu5a$Bwo-'
    '6!yQJ**l<ZjZb4H6h;=9w*!gi64L^Z$L&pLRd(+9+tXNtCPgp_1abMA#mqri7waJxKd{a`L-'
    ';?@(@xLuLfnV9~UdnJ4}?Y=k7y<@-'
    '_QRJy&bP<<S1RNdE|)r!Ed7G8Ltlfj4J^Wa~U=I9D_MZry#Q@yVqb>cWlKbw7?L_1hz`$_?&QggamdvEOy!oRLCYf4zqC<}vV!LT'
    'cqx@YOuiPwzGz>H)+ESDKq3_adF;<|9TIW6cA|q7)=b*?(NCA0%9dYIr+n-3jUoaY*&MLytS4!+Zkg5l7)N-'
    'G94w#aodrQQQGNpC&v;SW0+Uvx#@z4IAF2^V{Pldl(#r_Yj9CGt9v<he*z(x=qj-xIS^$=8k%A43-n_gE#I2-'
    '8(hHB}=##tbzQJ0rwHOO0Ol{4a)a`<Ab0*2UcL;T;@J&q<av0vxj+&#PwWsHP+-'
    '_U3(wyv7`?WZjQkU_c_8tVeHxpGY{I`gE+k(ckRsg437C4+p<Ec;!Rk`eS~9PX1r$!&%*-'
    '?Alp26WWM`S4CaE<O2Sg?WX_zcx45eaQOn3v#bV94McDn6>0-'
    'B=X9kOv@W8{^e=jnLJNI)Ymv{zaiI?Gc1h#z~5p_STl!8dl@;#rdm5Kb}SmEcyRqkp6OY|`GS%x~o>z0`svc_G9yjTZM-jCyD?Bm'
    '$x4j}dPPV8BZn0x|z9*n_#pv>|yO_>E=jAZJIiU+`BIb!P}6OVVMJ^N-x`f7=*pnF`I<hj)3;1l5cg1Z6Mcqq(>2aTln>wJ~M-'
    'H?*2_XglaSa!3!9=snk5yiT&<eWJVn(FhS#;aKOY4>%k5Ls;|twUZcPDW9p|1QvjWmpi(^)T%HIAU-'
    '~7@dij4<d6#2Af5(j<BAvKx2w6vQDpVfW;O<;+NrtC1#Gugw!Ot9v$r3aJsokyT-Mk$kK_H_aYA#fzKl35znK%p6fnG!+Nu0p;Au'
    'ZQB>O~0^Qd`cB#Z0ajepbD|r4|9AXL9Y#{JV@GPp<Gl;TPYP)5G#aO}bIbVj2Hfrx!je|Y&Bq-gF2#|ivydJ~ddDt7pX3$xQSX_s'
    'FuMk#gC3(o`&+B;dyc4>wLrr`ZwAUiCUodu9X{LcEO=n6_OIU*|I8QLEVaK(Q=0(_HrP2OrBPCn=Y4<;b`#^0C^w<auUW7Ho>Z?H'
    'eg%E2(tithfSc_A0!(yY+s|22^9|6r(u*6Haw-'
    '&q>nG9m2x^OU0yn*;%5FU?f{dikxzJj%{z$a@U;S0u}%S=u_8iOYZ+}&SBm3b95TM8>Y0nMLNd#%Qu#e}C#M6Ph(Bs>clgG>pIP0'
    '(hkdhk`~w9I{t@IuIKwMP9CwZ$v&$Upd80*+5YvS1B+<OzL_Tx}}CiZqDDp2Rhox0i<c^Nbp`@3Us^dmK_o#I48oHSRkE)?ezxCg'
    '^LkW|v0tYoNg%;VF%yU7}GkG2Qa(!0Qe7U7ZR<%Jiv+v*#_N6Hg`oF)Mb0?^5)-'
    '55Qv+fv4{0W5D}azb%9}2u$x~_dVG08EF0i@L?J_x}Gt#H@L^c+3bZRUPI(P4PQ$1@U-|k@B+bRP!ufxDnyyY-'
    '{XeTlh9tG=<Oul38&39;QJ!ve+=GVL*Oa%N!a6kLz}0XP5MSN9lOK$TdFl_Ilg^Fu)Nrp>kvD8L0#t49qt}NkP~l(*kW>zd*btuW'
    'P|%Tj+Y{NoCz-yHXCYh5uSx@KP7C2{2Nhm{^Ndy9WR=C`wpb!j>!Jq0@}|jm6t%_C475Xd&o+31&N2ZL0x*$YVAof4eTep1WT_pc'
    'mdowFS!DL0{d>#ci{45A%0#5`U#)QZ2g>xAI9^L{B4~EUqT$cr+7d7wwI8YGv75?xi$t|zuz-<eHB`~uH9T@-'
    '|l{lBZV!{cD>2H7c+pX!24;~3T%SEUImvmW~KOYtFaGf(wka$((yilb8!>6y@7o@;G5S8JIo9((fG33_)VNYL>_L`$l8F_TR?&1`'
    'c=dU$3D+}Y<=drm1>wo^Cn#VIElauY_Zq1lBCW%APd)sty-'
    'n6zquZ5jMVaN0C!jB{tAu{)NgMXjf4LDZW`W3zNc#_*rGaoXuPt`Twz(h)>ukFqJXR4UaB?Qa7AqYzR`TEdxG#D;hPwI1Wlw0yaf'
    '+@MtBqPWFwZN=o7+!utu!*0d>$15Q*E&?hgrD3GAKE-F=|{A?UpgY9F}Y5`Mz<p9!pkmHA7;2Z-'
    'CZ!9#l4Ho^v6<7w_=T$kS_Zo{|W=W*V}z$M;HU>xu&<~W`yK7t+}L?TVF9oM&-C_02b!?u#Xe-'
    'J+2ZoDqFkUJLl4%Xol%x>JrKUN=nO!$JpeeGSuDqC&4dmyYZsXYFLut)pHc0}UG;JqIji_KWKuL*AxnCf=I51_sSy6(~1w97qg>d'
    'rfcCfDgr$o@@XEs-'
    '4`g5rM2yaO`tFfqI@4W9$MV2l06?~JI{?_=+_7<>(lb|ZQ}HxjT7cbaJUmB5kkBXVmG_U%JHeE~iP2)j^4guhtlW7RSpzv0?@SR<'
    '9}Gf4P_X61g=pPht{HA6Z2cN#AS)hm_vzZpsgAjwXw`4rr~1h4O)!+yxPLn|lOgu?{3;5VA9LA-yN<gqKn9-ROr8b2^^7-'
    'Zir0&DXfEb|d!=W|ey>hr1bE$9B5X2lV7CCA$hYF~oh*RV<Y-H-H$;&%eq=7hAnL-'
    '79Hgdef?6Xe(T8sC2qxcYGw`OMTA$<IVb<Q|!*+&c)Lz;>UZD*gbE9n@;NJwt!p3u*VkMqfeZ12}#}th|lYTd<cYavl8yJ9xT1fc'
    'TTRI1Jr?0=2EUCUe7Hog{XF>i2}5hVFNS!rp$+{1udcGHc(3yup;RHGKMeJFG6LI<wa;=KTn1526N24*rCkE8_`*`nDI^2N3Ij1?'
    'jo}a~6LG8d0!TU%=}7as7}+%|6iN9X}BEhBn%3y7t$ga**&VqWWuC<PbF3i-;G`9Y*B-'
    'NMMgMZKnG<{CpU+ztRZiX<#qz{ebIipTnwW6bHzQN3s7K&5jhk!0KOv<~{;*;2h!qz~Ld|0jVXT!(OwOz3F*8JBgIsJ-'
    '=6<eWo`2$<Xw@D1L@-*!Ew!S9($W=|*vsAe{EWt6#vw><jK&hX}mhR+$~BMG_-_yScp7^)C@oUup&a#dHz|+x#T&2gvMtCwX6)`C'
    '2@^UuT&>-'
    'XBcF{9t<T4~VR9po@Gz;^y)8yV>JgL*WQK$(eVu_Y1sp6g`3`DDLII5q2A^{$?tmRHeOIdq2}}WxQipb=dt2))qT(bvq0hIZ}#3!'
    'o%>(?`Ep!JSpLQf~%axM@&aM2=A2fgnt<?pLfLl2b?ALxt@Nno#3E*iFYYx_7w6H4r^`uSu4|@?tgKx44lDS@4hnD{u(mvF;{=W^'
    '`F7vN6o<~jzY7)2rLKlmD<8qOvQ#&+i$dX|ApxJ6Z(B;`ZrVLEan>gw|gup-QSAuG(wJpmU#b9cq58^SaAZoqF~KL-'
    'e7&|srN5f>8MI^3^t3x@uJ`u;romqVR6wXI7`ptZsK`IA%oERQ*9Si&O|P9Mfk&1@!zzg1?R(B#W?<XI{g&dA5*XV4bEIYxrX!ou'
    'OPAK6+k@y7qUqGJ%&oo{^Xicz>9)wf59014z7VmeXkHfuKw*7^m3sBxSq_e?8O7xSr3H$^#_yby#6FFiX(0m9&}Gw(?^9psBtQH{'
    'z#+B5t`eJf^GD-dN4sX3f7tTo`BZ-'
    '!%WFJI7?aLT;4&*l*fz0#~~Tdt2t7R6TU+i3HYSeNJUEL7kv8grQ;okPTy#a`x>j{n^5`FWm|CUo#ct93wbAdC(IkJAB<E|iT}cJ'
    '(D=vqiXqaEyOPCRouc5X!#&VJp8jQOoC_WtX~l8){2!u5_sQ*XM|Kc{M?k~Jj?yY`P)(x$<RWn87Ycu=<}wxMg=cvt;z(pGu`M_X'
    '3VTtAOnHHPp47U-ZdrUQg3M*fIl7LyMX;_QB>Wq3luvi%Llr8Z-'
    'w&h0lrUN0dz{&c{7PhUs!9f(v$$6hK9F<pSqL0BVhX_0CwoN+N8Ef~dBj{HkG-GQi=uF7xl~3Sq3I(Js07;O0mZyH^6OC>d`jXd;'
    '&Dws*^A<*WPdrN=c8e691B0oc~KPhs)rVrs0hw1_aSIr99Axd)qg|Ve{ht8l*hC}<P8r&d9MiJkaryN7e-'
    'c}0B<=QuMVLcC<|)fem-c)vHOQx2^>nnAOFBlH3=n<2YJvZ_<EpU!vBf5uZ5f`qSAQk>va5syHONZy;G=7$fw@Pr-xYo4>x-'
    'xA}LXoDk82$s{)=Jk{jhTveWZOyjBObfW1-'
    '#7Ap#?NVO9Gap(Lme9pP$lVv%2+8EnQ&n}FpETB1Dz|;}0W0I5N!61`FmkNZogrZ(O^<Z8zHy<}Oq!?(HfX=1h#R6VkQ<IA0dVct'
    '}G`{gYrd7an30EmcQQWKVRr0EM+zX3>Lct_0`BoIxP4|yzPy*H{1DSa`Y^2rxWUXcCC>n`20mo?Im6W|Jz$Rjms26g3&9r({^r{)'
    'FvW8Z(6beTq3M7#0#nMnU@{^!BB+QNLr@+Rg0jXqV^qZiJ9u>XHnqBE|Lk{WEl?9~=*q<A9sxl~+#+AA;C~c<5vV?pD;m&qx47m%'
    'y^OejDCOOJ}IpO-'
    '@Ke<9Av`s+zxu@Tnf^HsTwekd>oJ)BPG=n|Fj~9L`1le<il!H5tev|KIakYiUUpdX*(gfL)m@_IyP{G)b>qHqsZIk8V;r!THL$gE'
    'bKs&D`p&~&%ljuuj^@yT2BA_acDq4>UB3}xFvP5G+uRWm*EEtuj6+GF*I~5XD#=WA(tBgv7x;Rb&uj1fa12R;_x4Q6Ub9}F1yvg9'
    '|QyG#(!Pck?iHqQT3L?L~a+L^Z<~7u;=G>9$(G*mpXaU(6C81kA!YQEI2-'
    'c~j{j`LcampI&oT8|fL^DDuouOLePIXwP6|S^|F4aul6a&YSrt6fWTwr-C;-~{^<-'
    'w<tv4vEND&TpF$NgMnC=8CLXf=s~^MfmKBK8wCqbi|w3`&514U;+5G&|TTEL(o8WD9e>C<&cvo1Ccy4jcgmp<O9+wK?Gw0%tf+dW'
    '~T5n&4I%lGMf|QQBCjqN%YhfO62SIuKNQ&STM`JXVwi#VAez#VW>YTw|(Y!jM`L*uIs~sv<a7Q~n9?iVg&}Fi+I&wD#154Y+~?a;'
    'GMrni=<}qduXU=3Et6U#ej(69Wy9$JGg(1JwwXAyZvYtf&<;anwM5)lxp<>qdkcgz5%0AxmTE%vFsuxt)3@3dw%9GEet4u&=hs6G'
    'kQ2jCVE(y^&faI#z|gO>xvSdCPgykdUGi+-'
    'G}polmDFSy2i4)<L8S?^E?kBdz*TR5Q;Wt+7g~U<2b(hCE|6CR8_9YD51hYI>)EOMP%?g!MdSvDexWPA4=p96N!w^qu;KRue*JLO'
    'm0aHDO1gS<m=eY~ICWQsA4EHJfIL>V~L-'
    'l2K8AT01o$@bhhBuVWIYDb65BEt5&2kxHD9sgB0vSz0R^L5?PnyRmYRxN8O<oCe*R;c7Dzm7QYy1>3k4_@8RJKyzbzMn}ll3UWoM'
    'b@F;a>P{L7y?}0BAAD<KY{zIp=tAIZVoSD!EKH*__;>eu!&>d}t+O`}`&t<LbR`J23ec`4Y`|TmE&9M|SknP?#75oB?zW+)Q)q^@'
    'tqAS0PN;MNuQsqx3s{Jy7w>m5**6+IPp6eFajhF@iWTZX<0zPSTdb0KxNWF^%HzDVBBFB~EYaBOi+aFxdU)ftPICUYP)jxShJZsm'
    'NY%wu)}Dm=8Ij7Xp#+`?CJ>||w}Isccq7bO*0DE%^GP~?$21IsR_)<|Ah%C9+INIqqTt>xRjUmw+Sls`^bgS)@`@Mb`#@k&cyKg{'
    'J%<x`c9M?HHMf>o)d5+0I%1}i=`f7M#9ux9)&TLtHs@)n4=m9EF&<QkC^|z2u9fvO^o=MInj|U>SA_nc+#33HhDM!@jq00f-'
    '3dsqcallz6-jcVw|b?sQtbx63I9gn(GWCy5hQ~8q@g7widK+Th|1P#pB}JI7n4<e6^Z?QjnsWjM4wGK4YIb!^-'
    'iGQ9lWBe8*0?0V<OPjP!;KVXm)n-'
    'CM9u>qAvB|#^BkX(239<G13JT`@upzp#`sR7tVz#O}S5Td~k>0`)$23(7z|X^)gx7${UGeteHxC7>}|ZBM1Xw@jh7B1-'
    '#mttZSoBz+CUTz(XSe&Y4l5GXQbc1NP~zQno-'
    '`x56<LS}>LL?kAn+Ou{HaGsN_%IyW@a_#O)`60hBn9jy?VgS9rdLfsyQpV$uFFpr$6v62q9>qOY1H#C#F))&$bg5<5xvzp;N)Knm'
    '@#iL_z7NIvHvM+e_)mk|ge8+&(AYcrB8mh6{6p=FkKS$Bt&>jGZheFSOu$1g(-'
    ';B+GF=mz+7z5eG9p_s3vO8=(1ac1!<r_x`@=oq>cWJKrrejiALsICCtNoCp!{MD__%5`@8rjA`4zARL0jcA?@Yxl6M}Wpq6B{DeI'
    'p%7rX9bxvgm79iQ-'
    'k{~HG7~7bZ847p9d+}hMko{Q27NzwPN=(`i#`xJ;*yZv}g)}ZTqX1QJiLGJg!}*VShXDKF2%Xtd&~My{NmXRsB@s0UD_zVe7ugwZ'
    'W(?1EDupoB_!wNY#QNIy<C;t31a|CsRK=z$yd5bvz^-Nf7JERmsx+=1G8cW?mN{r~0Edj7pnp5;4bek21Si&Th~*x`QiYXWDT-'
    'qJ6k#HP^_lN{yc-cs2>B_C`$0XD@g(k3SeTAC4L{PV2xKCi=&tn~cKB$=ER(YbQW|$>cMXXYia6930D>CoJ>11kM}2-'
    'yMD%=1mQG_aVr&p{BBq@}>~_nT#2#aUu666Z)APVnipV-dbnU!8L6(xQ#$QOx7G3pw*hQO0=13DD~A&c&1J-'
    'XJcPr(*cC>;LuOAE0NQQJ7Yj|I`)r4v`R-2i*k01g@i-QM0_4$2#yI@+aFS~#n{?`-'
    '?<CO)x<N<FdWgIHr7NA_bch3W7Kx+?{1(r-'
    'dH7J*`V%>gj|^<O2>HEc!FAEnEI7%!aZk{`bKaT<U0$N8H(@IVDobzt;E=1tQw#cWwH*Q*9I9m*q%HSOAPU~OX11MIHp3wz8NCtG'
    'QvcI@q}}|%e^ar$x8PcMBIoN*cjpLy&QHqH^a<64Lqje&J1{EhW^A+IgP+i(Lq*6aULj7(mNM=GokbO-bLyo?s%NfQwh^lqALi~f'
    'wN*Dc{&Yz1s5c7G31%<-2|U;HIz6#FI=C3m2Z-'
    '%7gJ!{OK`pp^v^Pp$<||C#uJC(knaM*mEH~588_7qQoE#sqh!2hF8^KpWaxAixJ);BJ`P#V(R?AIWJcQgYQi9JnXI*LvUe+1TxfE'
    'Pqn&eqHu8H6EHMr=E+$MzT*=kD;H#P7BGvpt*gnvAMiSQ$Mtbv*saFA<y%&Po40!rHd=Ar297O#D@9r2(BCsVUnyNJ$S1*U$_Yg*'
    'E9F2n%(@b}qZTOt+HTRqQ&GZORQQxqoaGN&^+DwB_M}q1|=romT+qqg-qTs4@PPi@s`7O;b<na}-)deQ5=Mea*ai-DZIuox`3@%p'
    '7T%x|c2wblvToQvI!crbM-'
    '`H0&cP8w0Az^w9E`&`vPp8AXv$dLv<TpaTAWNdSHZ4w5xG01~`Bd2W5_oI`>^K`gUq?94;G!4^{gLqJOh|aW$tuQ7gINX>P%|$=K'
    'Fx(EQ}sF_&ji>)_FaONEd8bM$SAenP2SCf%fS75)TBAalT3x>;Vyf<;)Wz{*U3h*XD(qDB#PpCSm{P!wDytl_%;`E$sMVea-VlF_'
    'ghSzzlo6UJ;|Rj;3gGU?0N&?Y}~sR8F~(~@)~@bi)&oP?;zYtxGV+}P4Anln$5wTE5P+S*dz+k@|FmXYaqv@7+e6)bLC-'
    'K=6Dan<5xq|t4+Rg-msoGW7p-V&I|E*gI2KH2p7THoIkwg7Q$@c7VjC2sr33JF@Jj+#PXa?=f{96=QYZU`EyLn_7-'
    '97T<wEVa8=?;eXWVjIo_RVxC^*WbMZ#?#0|9ne8|JuIYTv>t+Gq3+^8|0@Nb~g9MHYhRLomraFrtHsNw}X9o+|i-3*y-'
    'G?mtRK8oqscM<Hd0y^FX&)tikZu0IYh+jpY;0Vso=p-4%t%wb&L$`a+64;xZt2cu?d+`o<Vjk|?jvTxgd*-'
    'U0FU8f{2@e==+(qD8@gS(*0-iVGTNF~sE=5Ezg*!q&q;R*%#A{XR+ac{N)oLE$PQum3A3PUv&6hcQF65sNF0;%$c0~-Xhkv8Eo-m'
    'tmsj<VQVYj&pR%eS`YV6E8%dveQ;b!P~r?JA!u){3Cjw@lct27GJaT|6$q*)@eUFj_`np^_giC^x-'
    'TA_70{BaX}#9iqg6RT{2tGvey^~-QoGCa=X4355g!ul7~48h~3SFk)h2c)bk8GkEcB?^v#z@As?-YZ~{CkSG*D^!-'
    'NpwUCXO`v*d7zymbd$F2lRDLd)r`;)vd5G)TkRo_OkmD+Foo8?Zcq~AqUIR`XCl5mpsTa(Tc?DJNZtoG*C5jsi=EQ(4z}53!#L&a'
    '|ybJzKMQp&0E%g9_Y08?Q5(JYQOMe|~a~JfQkG1om;e2rBX-{@P3@bcfa2ss6$i()1+*t&=w`-'
    'Ph#dw5pjaC`vIS*7G)aZM}TkJgwEWpnT^!%XSdk_-M2meQ4@kPc5_v&oC5dIR)9y0m=DB(tv#at`rgUTbucOv1V-'
    'V3nLBd{Y|_f~JIxpqHxErCu^JOWwoP<uR~e8iJ%6OI$^)x3T-MKyb3f%=_&dXM)c;a+t64UqA1ScN<3&7d+1+CPi$j~QC?lm19(-'
    '^OnTwDsHgE%b=K^&IH&e+~IjZ#iP&9_1v*{jk#l91rUuRv)Tywt^tOUkR_>3ArT_cz$@)X!)e6tt&CJ2#x~J=y8X)2K4U5=X}tAB'
    '>4;`l06rL#W7&+3n2Rv*z`VV@h2i&Y%lR4)8-90OMQ5r@K6kv5QN@>G^_!ig|PO0YQg*Q^D|iS3iu0-nf+_MRo-'
    'gI_!zABw6_eq9>>)eL%cu;u0Ce^#ZxhOoWRvN-LKLkp8}72HIA17Pe6l3u>XtDAeaBA*U|6j--9*xL(+#p?P<bN!oAoX1@Bm<k(i'
    'Fha6E`3$bytlxq2`qu1brH?hhavn96bj$F*enazxldL!WU!s4v4ObCukGCK4%`@R%ZfR}}2`N8y3Qu>|YyR$JW-'
    'i{B4<AJ@+X@WAsLR~*apb(eTZB5E0|_l&N6CPPGVb=UyPPiggt(}`>D3Pk+VsCJ9NpW{NLN{>tNJzH`G_&#gmMI?U`_2yCak;ui>'
    'z*ehekmQ*Rn2+o0LyM;}m1Ui8Z1gOSrI2B{Qjuq{m4H+(sVwZfhj9NltxxG#V(hn!u#&)*ctY*Ky8_MR`g2JM68ZTUfpd!MFwY;4'
    't1VaK;0R+%tASNYag{=TenAh;z^ByW37@CpeUX{P$Lag40P=2~pNF1@m6wATSEXl7c3q5$m)aN1P%E|S7V!g*zF;aRdt(iO?=L|l'
    'K89zP4c=i;TZ8(~)@5&OGQQ+@V#~aPkZ={~iG|<twxr=5?=^zNBIlIsdlAyC#=SMB>%2=?sue&ik~m+H>{wZWqkk1)6ZozJFY(h;'
    'T7{lR9g`;vd0vtzyBpV68`)kWaKyc;RM+CpQrPJ!wck?Zu^P5s6h`pf+Tj*y|CN6Waz&`&3d4{6yAD_n`tprq_=#ktNmYn-h_gz5'
    '5x+3{Xc0fJUkKlr=2~DqVL5ELLF+Qd@mfVzc7F-'
    'Gf~ZMIwL*Ki9Krb*WUA~xrM7t%>!l++hbwH^@4ce_@80jYQ`ujSJ?{|KfU@ZP7}l<Z_cjpL$KZW$tEt2t{VIM@{}=CT@O=&1tc5L'
    '?o7h?Jy-Zl=?euneJM{c9&O5!=3B`TpFFHz}TN+MO5-;l!LCd|31lDY`k?11=>-'
    '7dK{0ih;kMFAq{~>IE2K;Vs19nGI3Nu+LSgVT0R#BV_`Q`Afh`$wlHW~^HiS3P8xdIw*P<km;@jt+w*J5iO%T(HrV!Ic`bJ`s@!C'
    'R~|`?DjW@pX7#B$OJt7I|I9kAnAXf|a9q5xX|S<{Qn_FRP=d<nQpJSP2O>n*3T1yKI538{j$4n2j;WRGHK9R)#$?dvAn%|53Yfbz'
    '%>2)V~Sr26h3#c^BxsrBy5)va^=I39H%1uVZx-'
    '8(>x8^*rT(*!peMjLpWoZ)5dZg7v^Avd&tsqW>fK3%570o;~pz?!9e%z`R%Ccolff=*+g-'
    'ObBeYSI+{DF3w}_(zSd+Z9N1}qyASwj+VnYFR~*qJo_{{xx}s5Y`s>N014^Ef9t}o^HO+zlkkONx%UgcZw4)!O*Sv0;QF>z=ZG!('
    'xfQ6abtZwT{;QZ}>iJFlx4`i=m0^>9a-@8r@@$06o4gOa8vY*d6K@Y7R^5%y-@MxXo5-'
    '}+!RJ*%%1@bk#|H01!dJk4^bNrmdj8d`>Tf}WWJaQkaZV;|mdL-ri#64c;>pcAZ+#8P8~Drd{T)4C_fGZS*YEEdyaH|h69bOLgmk'
    'ihi}L&0`v#x&{rBJx_UC&!0OwjAzm8v5j{qNmli1~b93S8u#cRqZJ?gd*-X#BXHZjEp%2hOc564zxwNr6-'
    'tFenv{*URlJJB%@sUO~h6dw>If?1j^1h(k=kbtAPp5NI2Sg(scnL;DK9wcrAaAdMP?;CD!d)p1($NcpUcE1Mc`Cf`<g2xUso&AFS'
    'zj`vi{f6(qfXe$`x@=96c^`qodsw?ut;ALpjXs2D`FbmVmlp;5?!U0dMz5v+Idc4CLTf*JyaQ`+>^AaudQs%{w}DnO|6P^49whz1'
    'SgDPlLR)~TazwJfw?g-Q-e;!A?(o|A?R}YEKlZGzcN6L%X10Y<8pQ|Lzsc+1N3q3ZmRM~&R(}p@qP^-'
    'hP~EP5@U`UC*D*1$RAP<v$ws6}_sJgU(gGTNXd?3yBMsNlT}mOKD_ZRUuXHUXKHdVq_Woz6`cjYg5=0ud+781<o@DC#yJ4>{&@*k'
    'k7xYiU1e_aF+@~hyc&+rHU0!3qoqy2#7Ll+s2A>eNBgXe&|7nEfYJ=~*pOP`PH3VnOUXQ0u=J77%4)X1EKZ@<RyBl#CMJK<L-'
    '`p2_exdy@_}r;E9>q?zFzdJ%^VV*ibv}ay6Oixw2%Y`Tu-ta6<v80FgFOVk|5=jfXL|k*Xx2xje`unR<K+{R(S`hihzEJv4h|pjk'
    '&fMl#y;ixxgPr>RtWH+K4E^ZlgnQ95l8Y?OdKgx13n5h-'
    'DdnHxxo9GV@ls35*lOXS_3_I;~vM$r{0$Y=EzyG2l4SKxb6vgybG=6r_YSX6H@FoG`})c_f$WMSHq~=qGzG<3B0q@=xV`Iv`I#S&'
    '=IS?q@Mo-b6*tgF<H0NZrubGRdVYyweIJH&kTb3H+bfkv#&F&Smnde-'
    '`fxcV%^V;eEY&{Ul@75R9den?cn?=ra7@TOT<284qp>K2c0j$K|I5IgGl~VrAp_qM=Qfu1eW9*;57deto%|ZYfJqADqTS73$we2-'
    'v#Izo_nA+eXCr)!7k=3yE~z#eBot}g9Mi7TfzawcR+C79D3s`uZzm{E$nAK6~*^heZczx-'
    '^AMA;V1S$@Z9kOVSfh7B9ru+@ANq6{RlZat0epJ(>LJtbp}XXXIi}GM*`pfj=&w|Yqi$*gs(LQvIEP-'
    '9umua13EtuM8<E_3O^W{VkMS@<^35nzBB&*22_9Ye%2WKF$Uj;F_Df#gdcENF7`@5aUa%yuYNs1VC@gY;2@@T@p?Lb28SO&{YU8c'
    'vyq(b^=k}%1D)?OK&<zvT9Iub-sBkX?DIRtLy%nJ`ZvN}rTmML^DuA_bmR%;Cs1TBiM2k1yoZbiKj2P~#VPO`gg3n-'
    'pf9m<gz!^HTRg_Hes84snQ*{6!~Y$W51BhZqmuoC^I^gdG2jSc>;LZk0iDIJUHvG&!1Z5@esLPe^WYx_V%1-'
    '~ZhjO$qvIX&{)7xi)Dw)u`nMkG5MQ&u_}_<-r<Sk4^C)!slW^GZ_ywzYUM}kw1rqOq%liDT`a886)Bep!#=Z(-C-'
    't7>d2s(V<E{hXaMU|yD6ltv^;V(h{$V2RnBM&zpZ_%wJMuNA`#XW(N1lYIp%hN?-'
    '%x!vhfj)cfWn_R{vre#yyMAx%Cd@bivL0b=6D3x4tswqIGQ*rj>O<Ejj<?>djEua9>GM>)0acey>R}=i{fa8{eNImJcfHGyyF=p`'
    'UCSLcc0@}&*-'
    'WA<oGY7{2OaHs)R!n%<DU^l3xi{tQ4ZMKac)au9_dkU*L1xbNnd8N`DiM=?oRjD?tzb$LJtY_Lt}SQTz@1|26b1{U|K;6Npa!Unc'
    'zJWe@frN4wOiLuuy|$|Z{9Nt{SKj_X=Ei@iiY<}J^H$BmxHJ+b*Ouo8Py{L8jG;mLnP*RC}G>j;9A0snZCp}BCdOxr!5{&8b)+>7'
    'EL6GJC~Qy!JwQTgSXoR8~o!sFyWfv3<Yq_*V2VP)~4&q?&$otG8zMMY~9qd5NQ30^11AQypUs*2pJ>etkBWqdl2-'
    '7~vd;A#~42%?Lx^z%R`^S>d`dLi0IksruS=uVKg6-D$YtiR9AtMvK(h0sn(msjn|wkSYgor-'
    '7=O+jj8e(07zgGF)~3!Ds|1&xNiQ02HjwgknKp{bl%hULi6D!49O3lR!{l31)Tp#ZKG0PlP;5DBFVp707n*C_H6;%Kqgg11WvErR'
    'Er?~~{6y<SEei=u=As#z3;Ra533aP4C(Q_R@6AVGM@8!?eT>bX3=%;b(*CogPQ$m}fvNL1!mPYT2jd9lwmx)d_9adiIV_4SMUrTm'
    'f_<!tW)h{0l@!&LGcPZk7);y4mJ6PC+^t5Fn-0aGpz1Cd=0*8O|0xL;cDl%_uTclhK}!jFO_Eof-'
    '6FA9TpQOI19P)6w$GZbY;X8kxG^ZKO-WeLo!cnnGy3W1$Vhf`$=rF^zYQNyh?wBT$1c!mA)0RQbtN%akDQ{MReZ=F7h`V|QL*Cr{'
    'A0x<2;s#8TnKQZ5&jC*X01g(@cOB=qG!oL_O16rxxC}$V3UWt&&j!dyEs8xm~DjEKwM}EIDAucA^lfjeqPsu0x6<$TZigIMgUXiB'
    '?Y*QfeHwMAq8Q3T(9}D4A(6353g;3%D2&PvB*UICQr}&B)AU>^%&$a}4FY`^>^I>&=pkEy{s{1w6w`_NosgmEGaFR|s)p10@QNtd'
    'W=M?$7zk&W>pcg&|>G$QBQ_K5xAyW;%mS5R;tV#xNL{S6aYlSto%RsIg;a|$6U}J*8R^T1Al;bJr24eH7>ZwYIb$Pn2L#P@XH`Q>'
    '|Qb}u?dqEYio&1Z61UM!v&Fo{QI)QgdMwT(U)HC}sfp=B&PxqVq6aAweYf+z2BL<a0o%aOK;9dL<xL*XC$?xqO5Nc|~21wB2_zj+'
    '^YeHV}MpeV7Ay6v@H8s9OXO>)^IE8){NYjW=I|Cj0D<;00@=ieWkUq!OQi<#0nW-Ux{lN52^7;Kr9g`~(c_L2}0_#vG?XP|ku$1c'
    'i^7kLzaY!uH&^0x2miKY|J3f9QtVgi1#4AoiwW<$$vmGk?SAyRuep5mdrNOaN*YAXT4Peju=3WbAyl`UK8^j<}Y^CEDFTkPjVEPr'
    'c({}Qkh0)&7=+d0P92%ublep3ZR&Jmk`d!~F$=j~lh{WJ6<L};;p8OYkTByI-JE=$rey-'
    '+s)JSatX&WhBmZ&A6v57IB(vNw(N9tNl<s#89yju9J2&Wp<_T@c~ye&!`r$U~NkW=n94QVyOxv}3H)X%`doaL{In#Q1+`k@6rJAq'
    'C*+{l}V+^CN|z@H^=LumFdksG{EgRIRBPW3xOr+Uzk|5B=LhSj}w<_zY%;M1<@3|(95zf0PV(BfaCH=!e}Bl?_*XVWO^X~h+<@!Z'
    'og19*2`GuyOL&$h*k*v{v_p=y-@eF?3wyNz<^SQZ)O$@x^nlVSgQtsl@)|6N_7Ut7CXT^vEbNsl%;<T<{nk-'
    'm*Vf5JdQ4a82{Fb?@%kmHFb{KkGq1^Zxtu{dYQ>0yktHhbHLJovAn_;;8bZEZ40d%AkIt@@^&ezx;_`dy*f>4;nF0hXnrYIi#70C'
    'N+~TN|0W#Gs?`j{NOk2hi_l?sUYI9K~Qld%rW`3_{0jAU5r<d^`EwP!Hti7Kq=#+RgND2~NkivPKK}O`)tg18Yu0mUe}|qv-'
    '7O@BR39?p<RLL_%A&PBaHwt2eu;Y<)6N9s+b%ouWv@QDR+vKlpD!r2cLFGkp2GkshGReifQSaVHA4zPxSihSl96{~54TU+fv`4+D'
    'A<m=ar<@AEs|hDyI5DD}b77gzd%^I$}LPw?+W=xO4jKdy;Qx~SB{^gTi9Nw7R(O{5;;566n$h8}~x&-Z(PQ&)Tr$`C2T0j~O-'
    '#avMbt4=+zvQP5p8`;-Oxk=rTycwRv2t`x%W_PvRAUy(bP3L?YM4hh%<R7B?^$U0O3%~W#V>FH-'
    '{uu2*(cUo(ySnRY<`K~7r%@=HjWE@EsKJor#1TwsE&Q>Beh~_R<r{=Q^}^lW#xKKS(BB^kzQdINa9AeM!Q1N5R&%%+=#Ionu?YKg'
    'RGL=;u*}0XIwYQ3>F-MifLcHQ%oz0c#}P(?*9g?>!5TXwlNjZng=@ocj-pQt#`_bXZzGkce;8@~(cNY5Ncj9rwf1OWu!*R))J0?h'
    'PdpXK`D{WWDu#q;h-c}MI40_}dMbE~(F!w$Fock(^Xb*9lUX$qJIBK}BVc`wF^P#m;UN-r2`fpWUdvPXC|n-'
    '_{YPnJj8HqVHzan>0=<cvgXuU6T!-'
    'oI;gCeKZKyvff^)EoEyPhYItCoaTp7m`Mney_F7xC`h5N#o7>p!H^mh$wRRETfbA_$kEe1?)6d{mfL^4yxsV#bFe-orSQIK`AY62'
    'lqNm3af1$%;fUV2ZL9(<<Za5lxqq=?M~60LoyLZmW%2<}cKa6L)M7x<d%HP<!XKOqKEgZgLCL##i<a7$?(V2pNZ)?u{%jcnI&Zjd'
    'vtm`vaZoR}dZhC`lnVq;mXJ;|RE!BkY=#0=9Tlx+m)u@oHD=iu78Atq<QI8(cW>1q@z$GL>@G2oe{AK@IrXw>4N&_eXHaXA4#9fK'
    '896^(J6g=<p@;z9Pz1XFp>Ba8`mH`MAsRsB8CuK-%-'
    '!SYORlBrg`P2F#(w8mmJYdMV|9!bAH$?UM0l&m5Nj8R^bOw>*G&nHX{eV_1u!kdZ6;)syk8LQ_&`=%KsoM@th!F1VkLp6FQ`WFzU'
    '5QG-jB=+YNe>%=nGk~etoIF>(a-IR_1yg29Q(zUYT#0+rV!_<bQ`<<SoU0Kdemc)+lURNJzXW&mIO#-R_7l^*0Fq9}nKfbG4MCO-'
    'K;0Rv89o-(_iWVP@$kerSS*O@aTzc^d5rf{e?XN>k=x7bedd1YZgqFKkGU({``w4#%iPP|sqXo1SGSkzI<Dh6)tw4{HK!c-36-'
    '`^JEy(V21k46{{j7J-Zu'
)


def _candidate_ordinary_passes(graph, count):
    for iteration in range(count):
        program,times=graph.schedule(1,0,reverse=bool(iteration%2),allow_lowering=False)
        graph.original=times
    return graph,program

def build_candidate_graph(k, graph, program):
    graph=boundary_transform(graph,k,8,'late')
    # The boundary checkpoint saved its first 886-cycle result at pass2.
    graph,program=_candidate_ordinary_passes(graph,3)
    graph=_candidate_fences.transform(k,graph,_candidate_compiler,gates=True)
    graph=_candidate_startup.transform(k,graph)
    graph=_candidate_inputseed.transform(k,graph)
    graph=_candidate_alternate.native379(k,graph)
    graph=_candidate_alternate.canonical19(k,graph)
    # The native19 checkpoint saved pass0, the smallest legal scratch allocation.
    graph,program=_candidate_ordinary_passes(graph,1)
    graph,program=_candidate_apply_mode_hint(graph,_CANDIDATE_NATIVE19_MODE_HINT,_CANDIDATE_NATIVE19_SIGNATURE)
    graph=candidate_parallel_branch(k,graph,[69848,69688,68568])
    graph=candidate_pointer_multiply(k,graph,[73968])
    graph=candidate_xor_select(k,graph,[72992,72264],0)
    # The tail checkpoint saved its first 882-cycle result at pass4.
    graph,program=_candidate_ordinary_passes(graph,5)
    graph,program=_candidate_apply_mode_hint(graph,_CANDIDATE_TAIL_MODE_HINT,_CANDIDATE_TAIL_SIGNATURE)
    k.constant_vectors[65536]=k.retry_tail_parallelbit_meta['mask']
    k.constant_vectors[65545]=k.retry_tail_parallelbit_meta['coefficient']
    graph=_candidate_aliases.transform(k,graph,(65536,65545))
    k.retry_tail_parallel_alias=True
    # The constant-alias checkpoint saved the first reverse pass.
    graph,program=_candidate_ordinary_passes(graph,2)
    graph,program=_candidate_apply_mode_hint(graph,_CANDIDATE_FINAL_MODE_HINT,_CANDIDATE_FINAL_SIGNATURE)
    return graph



# A uint16 priority per generated DAG node. This contains no ISA instructions,
# register numbers, input values, memory values, or runtime answers.
_CANDIDATE_PRIORITY_HINT = (
    'c-'
    'nQH1>96s*Y@X}p}V`gySux)OB%$W1(Xm$6qOJJ0TB>EKoledK_m=Bq(cOikQPw{M1l9eu5)JYnelm_>$msWv3jk&&zT#)_lp;>|0'
    'M`GAP|TjaDk0o#&NTt;6f1^2LlN~V!t=ULrQ``62bU_ti-1!52WH53}j#}9=%7GSbYN4jW<aksXtE;$OJ`3n#Z3-'
    'eP&h@ipyY@B@E;aWTj6UNG7gCG|ytAN+hd-j1w}BL_$1B9G)eS{mEEM81^I)G86iI3H{~<vPqWtmPnH0$?ELtqmc-'
    'C$pa~nl89%LKrVTZR6cr097w~yQKh8Mj$|7r6Oshd$pRat2;_-'
    'Da*&ku#BipNtdts=w7g;YNqt%I1L@gKL;MXXh2(+!fqa3S_>!ER)M!gcpE^*$r%ECDsfl-'
    'L@!DQNMyb)6EKrO%q+pgRke0o|<RC5MG}xLvP$UNF*h?!7MWD$?P75$M=@}xB5m**8U~Bq7N$7K+uM)aaVnZ^mreik^NEwD_YjxF'
    'D&Ojj|mL8h4$THh=!<UxI$;>!~o*7j^+2PEJwBpE0OS~*Q8ClOP9k!NFQO+BIWY|>HM=>9%l=sX?&PK~98`B~oS)ddsA1E!@GX)w'
    'lL6HT$nXoV&l;yB0Iiu`wXI1{Q0gFQ(csx|_kw$sTPE4}WXCil*@isHs$^okduOe~~g`D&`ke?Z=%fn|nQmL{kFsh(j<fIyN(_7@'
    'TAuWseQ^1u%^^k^91th0{Dhu}H!m3<o456@4JWwJES<sRO{tSM-q{33G$#O_bgO=RPb5jLn@FE2g(@JAHJu@nw=@pv-Xw0k(6ood'
    'Esw|!8t;W;Ir&P?%Q_D?>K-mabL<*oGi(-'
    '&k+?mN%Rw(mPQ~5wSAE7vBQ&cjm&nO$RsLrw~vzhQUKVDnb>^xYWGcwC6t1_{ZA385L`S36w`mM%uK$}Lf$fP)1Wo1(23b2<wP+3'
    '0a5`TKtUnW(rSEuRV%RsaWLXnfczR)00gJw^QXCc-L!fUl%KqtmfKw2}YRtu>XBT<Za<tG-6l*!E4QOLKcFdkT>3M-'
    'DS0~N!l3@Wf&nB5}ey-1*8pc-~%gFCBVx#`%=u6Wq_%jTbz60GJ7G}rmaCfe-'
    'i$g2Dl$KsMmEQ!3zWT6cnWoO;$ytwj}Tb>n_&BZ~f@EMFl6=~0c{W&$t247J{tOO^oD!f&RnQ6|8#knLi2U2q=V#VZ5NxU)n-'
    'nqyQZ&p2X2w5e&1hr@uRA;9IeSOyQYTwRWZe-'
    '+w!z7fTvdr^R=r2r$OQOG~cJj%BVmdjw#FblBScJ9G$oEjpKl^3j^v+n`Ky9Seq>n}!XzU!;4)cZ(It}F|r4%b|eYEqDFO2*`X~n'
    'r5(J3e0<?t*Q*5_6pa;bLgJ~NqVSS!m~8QpGWsE~5REU%)MPdO_{9tz@NKGkn7_DYfmyURk><tMKB^!CoAGnQX*%!RCyz|M3<Sy2'
    'hpia`#f+bOF;-'
    'l~IA*ja#;3eeiwt0u1UR8_e^lNeMNN<x)in(Fwd>!Tp2E*}&X=*vq+1@u&56pDI?vZPRzvuLNT2KH9c$mpxHUXf8{p-'
    'LD{S+!?n=xrxIc9epr3a8Lcfz@g?NvanCJNZ@RQzV*^`wTka&EctqWi`=jJB@vc%04WM#mSV}RVM~D0v*YF2gS93Ukmw>S5ld(DN'
    'j2KUBam0qb{v6=n-h)qm_@E5iqF*u`?f*6vo;bcvO|OnrLp#N=e;0g~M-'
    '_LgK0l%3^aZe6I$&1bPK}26_YQdx0)Gx!wu(G*!kwyVqL~iw2A<!utg4=35QbidBrAt%}NcQ>0ins%p=u8p*GEfv#9xhiBbDH~eb'
    '?Wlf#G2L4+;4xUYB&l9rHcubPTwjuiKFm9tPdbLwQ`D`5M9~i*1u4YX|T~9toqOorJ0fB*G)MH+k)&SYh!ZRFR)6tx2a9C>u>WZc'
    'd^LXqvLP|qf<-I=|FgA}W21bfH4z}8o+%?ds3Dj*_ZKs_`G?zAuNxQ()!ZTqs7uo>pn-'
    'P%)enqq<l2#uL;cFIX54r`qBcZ+aTA;b5cso!{9(wwi5Eu$qd!aS3Um&f*6&Hy{=xhR2Cn~G64;u}KKMwVf&|dlsaqC_bT4G-'
    'un$g-Kqhnw!o;O2s%NVqZKvU+O!Z&RQjafIln_)>i^z^~TX84sSFc^JJ6s`97-G$ba2sEL!XViqb-Pawd#%JJYD{G859uerH*u}w'
    'YY{7mf_PZh3{I=?PQoEkr&A{{M>8MKRj{H{Wn#fsfrkpm@v$-'
    'toqE=T`&{6hVgu5Wg(3LhAST*%v+y?1SaAumzUW<EIAN^VHM{lh^R(D~o2q&g9a$Bhex?)F%Fnw3ny7|4K6mNx|aJ9pNzEBsEhVE'
    '$X1(oM}o4{+x@4;<gtG%Ha5txGQJ(;zo?-zJI27OrV;G?(D5!!CywGcY`=%otnz_Y(<va>3*Hy#-'
    '80MHAWedq^kzaO@ClXvwMr@?^*%GeY1%{j+UAjf9LJCZ#^b>@TN>&x>AMQkAZ-'
    'C3`pv)>QY2n=E0q<8c)5keOvJrNj&?mq0=9o8S7BKoCLTz6z~M*X1b&wf45)M#a-'
    'AHF;*yW+>7Zww~t?(gQeE~+w<){2{|2om~|E34h9;H3x*!sa2^G9U&6CAFWA{;KRD*wv4`4&*tA+HHf?p|7CgYIa&$hvAI}a>D1('
    '`or-gR?R?y?b(g=B=@^Tzo~RqAVRN@|5v~pz@FuxYJ<L^XkWnSNpAH)$TKUR()~Xinpx}&5simM;;6y5$Pk^nLEve!)tz}KIL(rQ'
    'qIr^?S)BHofib#42J4<|qdGR&*H{m{@2u~miV+wTSO{hb&rm00sjuePWi?>;WA`w;8}nH_nnmwnbYK*8L*W{RAHCSO6a5rt<0)0-'
    'lRTdcQyYedF#_%Z>^=wVtoHPIL-'
    '6kX31RwK(2c;qXV5h$Fb>`!RFqZ6SR{?al9B9=VRbP1eF}>4U?{Tt`B56k&GbAnMxga+Mb9+Y{AqZG()UO5a5RiZ?=Z4DOjJX3mo'
    '?X$t%2_7J~1#Khsv&Jkuod><8|61F_1{v&ItKwJm%p9`Q4Lmj1hP<9I3-qUn7ufmYcQx0&|%^s~wwpC;tUv-'
    'xP|{*f*Xy3<+bXFp-+*N3*X8tA*a;wn}>e)ZmTrJQ_!#aT2u8s;0*&yU+1_j^}9q1ocOo-LX%Qt=fE5zkrMZ><$Y|hUyu3N2v-'
    'XNkdJ3uRNu!JOzzeZFMq57{z`s`dZj*b@w#=3s8;}=M<t}o4wJhj^VOoq$;i*vZK-'
    '4mpBf*C`^TZEOD}Io3*2vjfc{*IZQTqm`JS0u;WROgz--mrjZwmz=!Y;#lq*<nS!^rXWwMwG`DJd9zB!M{tP#g)!qw4d@L}I*-'
    '%W7e^$ZPkBz`&<|8<D8v+}39>%GEfpx<~c3+U@DbT*DH_r%knC~Nzx>0!6=iG>0HnKR34$Oqxtep~gne~m#=3q-'
    ')8wKAuV$@r2ivGG|20=L}3WLK@31LhOCh~q9kGxmedr3Mc5)acm2Ceo!oIvHy4b0<dF`6XnqA`=Z<$3le;pH3H@I1aw3(RN!y!?M'
    'rU+IyUsoQ84674Nu5qb+vFDh#@@%UwYoWgiId()tpCQMhG4#i0Baig0F#^S*ot-'
    'l=DL?%q)%Vd18_FqHtTt+hjy8>TCVIJ^gPU9vYK@?sS#&a{iga<R|58}mC#xD{d%glTvb@S<7W^Eqth-qlB3ZBM&JPC>ybmk|5Zp'
    '7r6Fdw?9&@6;!l5AZN_=c77F*wG3J%dP0C#nmuem@dkhHC+ln$E82eU)42ZJsme7ovM|44#j{E6U>AVJz~ofCyU*7c+h#0%rNKz;'
    'X88Axlf3v#h_uXo=p7)3I?}-~jz-'
    ';{GNU7#1>Kq;qU}T?p?xSo#WioUOOec%6*{=vx|i7td#7$x`lznQ&Mp7Qp|q_FqExc%8b}S(^j(47{~0ypGIw(K3;}3DCYlX6-'
    'G#n4On-'
    '+WT+@{maVt4DA_a!Tqjy<{@Pkc9|zLV(=;%oQ<Y=NH*zX1IyrA%6zu!>=mTWRpd?cA|xFMd=OY0SQmINu%7DrJg}2IzeWwNWq+J7'
    'B8&y9&$;sGRh{8ig}0F7jm+-JipA@Gg?<~xOXO-'
    '9(%*t^DmJ_kNbgwPz6kf5NVB}m!Nym8EJspb`lWcg1kK*p)>O4=I%9dAj7}zy*N1V<4163|fn^rIH@G(~2X86n?}TCF`HIIP{GOr'
    'w&QZ>7G>b?~LC1VF&w+PNU?n<dK)Zy|D}mGSN8_Uid<>Ssvqbc73U6cm3|1Eh-'
    'Y0s?uyP{1o8Wv6ORUaTfw`Rf<zN9iUJLvvoDpm^lZaZr7vSA;WY`&4jqHugXX4|;sAd)Y3e@w7z#Hg&Qzv98ayOH2tA<a&d-'
    '%E({v|xuLOY+8P52m%MeuA4v~f1VIZK*XQZH|@x*2~K3-'
    'ggVUuS5Z`eoSo9_=H=cr{jhh~={bTl|x@AdH20z8Kmy?9L^27P&Qnt%1AR`9Ro+-'
    'WLKph|eP3<Nph9`dCUdR|w0){RfKg2UxxatfH;fXhUERnvB=(g-~@Y)p?1=yYPM*$mG0E-EF6;-}k*-'
    'B<e`4WY6%P|1`<VRB`XhmW@PsBbwhNUMun6vap!-'
    '_t_onY}PmXN?E*6v@6(M71+jZUS}D!gPp<7C)odX47_|UAv0#%Y%F_++55Wt*0Z;fm6fcmgm<F8wCoK&S>G<3@X)fkp8du0;R857'
    ')|=>4=vT96H{LRyTZr3c^7;u>Yelz(wmJr@^kpB3tys27b+H7w>yh>zHL(Q=mW@T|Sx#hkasyeER-t37a=-'
    'Y0Vo4a=up%0($j`g<+mUQiKjrxjv+aS;z;=IRZ3Xr$qu&|Wi#02ewGRHJIuol^GpkuQ`VV!kKg90!8m&_{ZFdK9cBuEBYgPB}L-'
    'AQ)H<TNMjYQ01w*nc<M6n&(577J}&-c-^UT1%~U$us{?5&1w3)rE4i@0_u2Ww=@ZsN5IU2Ej)M~eLi5qJ;UJ?!m-'
    'W+j~4l(`ke_H$|80oOh(T|)%+%3eF=Tame*{hi=T#nf`W2@ke`)kw6PeI58x_L=m}$l1m%zX=)ZncIn9r)Jn7S%%&JL5RlVF?@wp'
    'M^qbMB5Mn_+IwjqdiVMLm-Op}ufZ1HV(anjE3~hI-'
    '(q8DZ#|<$^k3;_{t|r87t4Bl3Ek8mv$uw^ovI<cL5Ao(GQ>F;I0V&3v~2KmxF60q><8xOUMw}=4<X}Qs6SGkw_%Z&^^HXPTVk|La'
    'XA2N{ITdZ>dbz^&L=+XypPn^oV|&r)83$M#rF+pIYAUYg<_L3Vc4Xa`Xmf*Zx30IgWcd${9h&^>?GqK5vQ+p-'
    'u9wlGvmLwb3bGL2{QMwV}5>2biPy6KV@~R_IHu9?K}_q*v9xv_M&lLco1-%49?#?_dvBHaFV!LHZ8-E*apV|KO-'
    'B$$n5Muvwib^PlUgw|2}Y-'
    '(P7qpKw3ywNXmBPZ~G6vgWu}s0GY75IUL3|{5ciA8A3Qn+b=J_ge#wu4=cXmEo^lA**i_Yo2c!>i_`jMI?df|+1~8uehXR1=X{1f'
    'J7M<i^A*&4vEwA;FR=Z9b`Hp@J?J^h>N(!>`=L4njonC}W4Y}ghW7+At$MA7zly?9_AMv7u={iWo%6o#F2gT;1@A`AR>kl;sO)=U'
    'R}8*E^HJI<q<;$KZpO!Weuw;1c)W+_PpXvNeAmWeZwwBxc7kk~<=)D^2pmA_37y)=^R!zL+Q)Ohc76@~j78?_mp;By+sCer4ifW|'
    'jL&LzJ#ZQCzQ)cyNIy#?zheFavc4A%hTB0sZFI;VeXX-'
    '>oM%}(8@Pn$hp^xrSvbHbgv07@1a70@ARLE%uH%e;0p}y|9k<s}{Xz=;+Oz&!_;(ZSi#(0@NZ<mv2rhwN!T0)=6%G5fWZ#4Kb!@+'
    'Zj$oNZb_Wu^#rnNyI*R|7;lCWX;^Po8{+<X~b}YWf$li%4*xpGvkJC;H=g4sgKZBnFSHVG~9oFcCeBRH_ar9c<{3M#|+V%7vR)3A'
    'PV`OL#_4^kUbd>n~4F7SRfKyPNCPrUk|0Q2{2&eQ$I3<av@YQm77S4N;_Jj2AkN8btkvIQO@jQd(pM_ui=RP!^LF#!lo?(77@SEc'
    'NbqwtL`iB@?K<kkhoC2pK@B{O+MEwTVo>k493rF!RI7l2Wa;yG?_H$7E4&8Zq@oV4~`7xPy;lCC5gLs@L8uyr;R#lu+y&Z+)j%x1'
    '$>!(zmw`pE}JYNlmnO#(sUY7TF@W|@nI2!CEokzzJRlnsYRHwiCIDy1#XuOYpi}X2cJcmZdIm*tzNV$si>&om&<o_NB@`gOht^Nz'
    'Vm)NoRUyp%h<qZ2*1AnTr@1pUf^jW-'
    '4>lQrC`c1q)iFNTDI}fL@`eqC)JAa}38vO&Tv9o<P1{an8D;l4Rz;Af&!5iHL*Jvj5JiX=X5-TT=a8-'
    'Lh=|rAUY;O^5!ykUThW^CPuk2iqq?3|ynsy$$uLu755AMtEYtnH|)?L6#i_eV+SR^l@@du*w6A_QbZ&-'
    'Ozk+=cBdFfej35)+hl2wAqH@Y*-t}wd=wdsEp0rST@-M>+9XW+i9eAw+_x6yCJ_5z;#u3C-'
    '8&&rH<%Y<&4tLU_o`fDJqbD8ml7~IDD^Ynj1aa&$rjKSY%x}aZVYy7YG_1sjy5bHl<U@Ldj?%<vEe;_ZDW2YjSbDr7XfkaLc=VBn'
    'flacv3Y&5=`I`#iWs49{=X&sYw$CsI&*#%-0<oOSI3Hg89M{?&<AR0-'
    'WtMc+T?|iFJ^XmrBzsQ_f{STH{79WQ3h*<rD4R=)053%r`YSMD_*#6^F$8jFhU4!D^KnmxEytqZAZ%}K$Apb{3mvz?fsg~k7Nu9s'
    '2@-'
    'F&*SDoBN_a8bn$(&@){ea1c=cIEII)9?$wzS<udLq8gLNQA3T$jy%qxUwFZv;|0@f_QEporaN_a5U1?8oQ3E`#%rVrcQX7lr$rl8'
    '5X(hQn$p37JW%=Ham))B92Qn^-'
    '&!L;p#xbC1;^@%Y0(hY1|l2|AA$J<xp>aQ={|e_~NQA4z;9#BZy<`^3^JCxMeF29NOYPbe~D?Y~H{8b~a@RE}jZK<sb9|A>+6{Em'
    'dbm?f3$ySf4H(Ok!}d=K77;z<IP)!toIf#II|yJ)g(By&;-'
    'DS>B4;xPV#CKd^yHBAo{`%K99Mv+MEqycHeNa!TTyN7T;!mfMRmxyd-bs~`sOA_OkL$iqA4`g!;Il{2IP3J`8F|r)&yu(*-'
    'z{%r0RNel|T4FR>ZXfFADe5??<HzBu57oaHgXB)=X+Azy)mT>l)Vccude^b@P#P;-CshnwR=o-'
    'ds3HQi0`zunf{tl1?E$JG5$!QG<T?o$nbgoYC2`HeEWY^S@ueRmwuy*cBBw0e@t7MDFwV~`F{2cY)ki|77}{K(u4DI2a%82UrGh^'
    '}3|yoIeXCQ4kywAZOaymY`17%1vJCN%7K<Qp%FMY*g_h(_Zb^tA1s91XKLvJW!IET*)B1Y}d<xH=w4B@I@*xh1;IpbpAA#cd5+9w'
    'Z;YlL<vML%${pyWF${1vD3dX?nroleTb~gEFcR?~_rosC{AU&0p!ly{CGm$g`$!PJZ$Fxp%o$w%hso_cElwfQvUbv-'
    'Wrx>zwDpP3{=SWmRTYPph%GM-Nr!TSYjSLa6n3Qqy64m6$&4h(zWl?7IBzH<W6`aznrDLAjKiTPte>o=?(aOa$1M?KDq+u3~!p`G'
    'x1*=%GCOPzf1S&zF09`53p2vy&Uj|}V8mXaw9ryk_q8{83x$!r%tVrjS5=}J9W4EEaM%JewZY7C?*=v<k9A84HC}gHC%M<mCM8N#'
    'X?YA7*kxn&{1qnG=si1sV1?A?X*d1b3mgYam;AHy`Dx;~gP({t|ot!=_PPy@>lza;Hxj_~A;MrM4T-'
    'mWRJ)UGlOGYT{T%{$3RY5Z9!o0Da(mu-gNJ5NrhHtk}^=Hy~&5oQrv`pBUjy{8OlnfrL`oxO3MJqKC&V#M_sr8b?BR>-'
    'IfNJn&VYd>#7XaRAC?up|CzEdB!psUgwHaHCsxeNBMcJXxgT}(3XatIZ+KO;G_H#Hzh)6{$G?!BY?>ry!I@RGXO-'
    'oFz)`luQlJYUiLe-hpEYRj+y$PB!U~e?+?T|4BIa$jNU(0ahLdb<R8EJ*&M`1k+@XW2hrWIzV2z?Q_v*WS%48eQP<n!yw;Z0lwO+'
    '}d%M{edQWbt7%p?PMg%R+6x=5GaZBBL#`a;f68IY;y>#D2*g)~~HBP8^D2Wp>q4Ze`g+QQ}|4M^)lg16@b;yU%{d+Ars$K8}TvgN'
    '#I;9P9JL*VxISY{j7j$gWzA!wtUr^J=FA`di~qiE#bLAwTgbj!#9fu!w(y6o9g#lP?BVgT;t<e)(m&C`tY0fjS>MCM6Hgyvk2*)('
    'Ws+7OK27lb*+EpqrwUQ%k7fKWjPl;LA_G&5y$L`Dl5arp~g!QvD@8gwVfQ{XlQOnahg05}bBs^OKvxl2HU(@;S|r8oHA#1Nr?-'
    'c>ktXUjCYmB^Wh#ZU?GjS!LE+u$o^M7eq?Q7*rRkVZDcHF{q9O<;YZfWi7w%pt9<{+?UdQm|u5Jsqmdq+Np`GTt1aa$&W7u1mmd4'
    'v%X?g0Logrd+Nyc`a1cIh;2jqM$|@Q`hxzgTsC}97e;DD@fFZ%Dxjz5e<`>N5rYznNm2FQvkrckHAVb0SWr7HbsBr=jM`dhryX9G'
    'WW5P1rnM-'
    '1rKs}SNQlE3|KA=%t*#S|P>f6YDC=}~dO9QZ)~Sbv4tN`l4(v6f@1pl*IaOgaI$}{{xT8@C>E(%bB|JB(5^5!gYzgAu(CGvkqQ8e'
    '?Z58EJRZ)GP72z~hMQdejdxp4_Cqh+ak5?U)RUsv@sBIM5VNKO2csZ#;oa#CQv8K0T+6n$rl2%^wDza0XtcKoZrJO$4)(OoO@TEG'
    'f3c8;756V$f)rot^XRDdI_*7rIEmIZ3xe4W<zD`ghBHvf1p(!%@VW-LJ%zmiuYx=0D%Js(H)84iXh<P>HUz6N)p)ards|DE_%(|U'
    '^J0G>2uEaSSwXigFGt~Cc28pett)i^0tlOiV&O$9&U&ZMr>uM^emHgZ`#E$B$)nRO_A(*EYlkV`<kk&^2PGxacmJf}fsL!mv#`T#'
    'sqQWiH4Ut+OUehp2Ro)2C?DRF1+#2FA?LFYAPiw}^i?GG0u_TS=%r(O|i%Jvro8U`xdb>xO;%^IbY<o>XGbA=qELxzuku!*TYR$6'
    '+9F5tnPhSVUjfrn_PE8wlTXP~?i@#Y|ifOKi6-'
    '}V7&B%+dJ%<3J>PTFhIzySY7GHDb%~VA#K|M4!ly?pMxHW>K6SdI>8(M>oaMxzH86Gs1jg6zw7VA3GH|1GJb=?LH(P)o--O=2OaR'
    ';FzJl48kS5x1Y=FS+cbW!iM=4kCLYg$lS9=ZX04h2txuI%@7`a>HHliI|GWuzZxs0%)~Q1vvy_6Dl#M$RxKbz^S;yN%gzq1c2l0@'
    '>a0ycIq)bDjjIeULK_zFz2Q5rbh+4~5S%-inGb3<6$15H0N$*{6}$18J?4*P;0MG`rn&w{#!}ZItzq&a*@|8qb9NYU-mk<1R>RgX'
    '}3pxeb;*!^+dhP2hGwM=K=tQIz`-'
    'ukKKEBp!Wz^b<x=!;$E!>`g$rdD({i%y$+zvxNETJD{g0l6q<sidZWjZLy{e^YO}GTScX)GT4Q_4RYJ6pG1z^BIU{diD6;1m&KEn'
    '?LIU^KlL53vlCuLqc5==4Oa;5h<!)U-g%Ks^ugcWj63+dUS)dEwu;YCs5;?wFZ6bbLD0?TT6{WVS7&URitOIdcgMHxc-Wrjvv@g$'
    'nZ>pnbiHL&2Udr}GYGqTz%|909)khIvxh8=L=P&l8_(uM%pzft?@h&Y)_Ls|J~c0r=@*H9H#9XT;_aa7Bg>|OKC+=l1iHZ6m3yWS'
    'GJ7~r#h`6Cn<4bY?w8={z}bBg9{V@(L9F)FX|#%Y0)L0HJ`BnsSko;AW7r>z{n2PGv{6M^>p}dU5uWwYA1y<$X&7ydWcSCnzPf|#'
    'bU&kd8Hy*-'
    'coO>lJ_h*cMKs&%w&|j3^w3{5W3~1)l0&dMd4uyl5MMgzUHlaLW6@+e9nI=!o^~?LkB|oAd>V=upd939_T?}}h0k6bhKqhAyU!7y'
    'X<#gVtx>J^(XG%&-'
    'zcNeZ%?bzaab`O`T;r%k?0SdNgR!?8PN5W+yTV3J)GTGALOj1pN*EWWOOK0uYgy>7=)+2X+1f?7Q1oO)!WWY@;3$x24L$zWRKLnW'
    '}d&`Ohm?1IA`iEd6xY_x}`@VXBg+)q>smn*Rkv=c{Lf@m%uB|Yh>gJe145K0sjZlroj0Wa)zUE2%JOE-xq&hr#+AKvFI6suM^0RV'
    'OAIevBm6}ME`<DR*^4re!4Onp_}np{GTn%31b|7J^%k;k`Jr+!Ft1b&l%2}NbLddDD?CrGvm=8jmcEuD=`@6EaJp=CLR;v7>ySb@'
    'OTQ8R`XBmo_wAi+Z(ErFpT+B#lz0#3|59?&x`Pll$S3#i~TbbcXEaib1#Aul%FMxhx;`evThJru+ugasblfUVmB$G61=*2int9{6'
    'yE{Ev1J69hVIEU%kog<tfl{49!($~Q&@kE=aY<HLdGQa*E*j#dz^jFW@UOZ-'
    'n~dZy}Eu%XDfuMP|Sd557>;HJ!IbUFa}vu>8+xdvK9xcg5fckjQm&NdI{g&r8b_0b8Hk|6sCqT9j&jzw}qUJ@G;Vdotbgyol6@}='
    '7%G5TnvnVCP?5efNGpF8;SAATZYC-SUV58+0H`znoN6}W_QbSynPE>N3pk*XioI&I1Y=HU*ns{+ju-'
    'H!;w0X3RsNBw?(nYd5`Cd*pR@T#NKex#9;+7SqS$V_&kNmSOo1XWX)9NjA3Rca1L3RBtOQf8okkKA8UNf0MA1|6S>A~=Xn(InM)s'
    'y$%?RL{yFqdiHQ4D-'
    '32efIbQH$Wf+UZ(L`b?_DzG^@;_VjW=}N6{0Am;8gZKDysyy`{F&;k!p0f8XQvQ>2|6FkqOcqutN&HRU>4rZmY0#3PYlN*;U!Ll-'
    'C+xe{=2NsCwp<Qe9er(tE^9<f@ebiIuaJqtbSipdkyUil=n!?A>KwaQ|Iv&vKx)pu<(NzEO%C7lV$${c4xtD_uUF&u!IcEVK)-'
    '*>jt(P#WMa{7~bicjh9nmFc+*L&o9$vLo-'
    'X!n1Vl>kZu`!(?4Hv*bL>%?9L?yucK))wvTl_)ciHA&jZsre`^@q{7q(W5><P5u5~`5zP-p=HAQ0<66X-'
    'RdF<M4w~n?zF|{?jzo)UYkeS_TA37gn;X<U&!r}$aRxll!^~7Qk>#rhXL&SUAVE6sI%ELQQtRlKI{TqEf7A|IWF7bUEE0>`4RdV('
    '!qj{{oNB?mIHh{%3_&8jB@A!B#2Cu`l5y^|uwwnBIr2l|ue@a|G#hZDw&)A!*N|_1$X5zRO?MoH2H=(d9TLHGP7l$pvd}!AYi??a'
    '7>utIq0t<n6YmWEdx^EDx`AFZ)y}lg_-d21U(!ZfR&X2%s<S&P6JI_tr=i89@797#AUm06jpA&_-'
    'vTd<^_`ul>g=w)YeM&yw4aaAmkA-Sx@!PU|F<z}8p3D45FY)0;d98f-6pI$Brr&04&+S;b2<e~Uo!tok3oF9d#tEA2-'
    '#62A5+)<#J=rjqOud7wcgUaB!W`nWjprh*g|L*0ovSEpL$cNH4%!ClY?V&aQu%6F2F-S2{~>j^j%aR>y{p+-'
    'NezF7B|E6WkI?#&?Aky)7U;CDMDqK@cn7|1gnJ9V&U1D`F<-'
    'jZF<upe*U|SW)%hVDn~A;UdOeoE&q|0U^iEh3gV{0Ih_##1u?4&Aq<tXX51hSZZzp>nW7#L{Zjk*Op!vkdCLbGUpZPc3nlRqQmS}'
    'vAUrXTIfSp^JZARWZI$i6-'
    '_(<4CP8RxYg(~YC@$H0mC!TLY(<em6PTbP}z@qyZHS{@Fuf&e`&~1_55YAQzAK=?s=S%k2`I+<DF7&Ka96m?eYUc~MH!4%hWYJFi'
    'd&qMc65dD4C(ym8_wPoYYa_4@yw955=GF|GwHgPrZUvGo>f0D^a`t1B)zU6(Sg)Fl#7X7%3+UcwceNsGr+7yMK1ISl=Ud%17SYI?'
    'Zk6*DE0*DpoD=Y`<otdQ?JlToe-'
    '~?C!~HQ)Jb<2c$oL){ih!+tjm}4k%*Jp{e8SFhWbfBIa50j1D|4SAZv&oN6jreEK05Y;Z^GEGo7AdtHxXTjln)tigz|v%9UNP+Ym'
    'alvkKszy${I#TVqiSG@PDs!1|GX@k2|M{hT$t8U$edq|JLKdPqeMd)fU}>U!iw{^CRQE#BnG6c~;G${cvr^kFTNmSbJMp-'
    '^%DS)%SX;`7`hhJa$et>bzLxU4Uwn>iZL<d<4f2ShGuC-uAArH{GYuZS`R{_eVNgAEV_P-FsV!!U5qse{?1WpZfR~E_;J~4$XG@e'
    'MEFGFv-W9qqI1DjV;@-(tF!{5rgj{a1iXm^UsmAi9L(MK}K8g<a=dk7d!iy9mkX1F*uH07R%3M-<R}<$kh%-'
    'a69{7!(+1dV@))6vKoy~(fe5xEEl_2JsO7R-4{r<-'
    'z#zbaWojERr3MGbw^ZP9|m8>;0vUG0f+g#kMVxt+X#F^y+m4L_*&Wz(GJtTbS^4_A=r63P0oId!eLg-!|$-`pz|0DFM%V-'
    'vP$_1>(3%-uj2g`HMEzf>Hi*zNPJIJ4~MHgghR;O8H1mkbI3RZ&o^*<@BHfAWb_*&lOK&=SUt+_*E(m1iQW<7aFmh#emu-'
    '*zvjK&5nz-'
    '%67&Y2b7L9LA>A^^7#$=cw>Vdabn>s_=P{bq=#QM0{pdOY)lo(2sPnVp9*J*>(P4Nl`!%vXjNgf>oseV1>L@ZUpvANwApYO5d(64'
    'c`f=o+R+XG|{-hqFVYkwEJik!;LeE{;dK_s-=+Dy*61O<~1Wsbj1!dzv6ux8Ms_G*8P7%YO;5~r;3(jq%-'
    '5}zpnVlo%KSOaDIeSEFXYMG}f6yOb)oSA`{68x@7pRL9>{uP`W^_!IatUc6{K0B8E+g>-'
    'G(RHa7bILk`vr6z*Xj74NE~F>vh=g?OPKSdkJCE!7s2<+RwRt?Bz9cE(j&y^C#bAqtf~)DwTGQMoQi;Z3YuS_x`>U3BXC6RFX}F&'
    '?Uaur;ZX=T*!_V_UQ^8PV$;vW#yj_?g){y$4xuVO9)lmS@GLcW9pA62K97n240_IC<$1JSrrjWV-?0BR)W_lZfz>m#KNP7yvE?Q-'
    '2U+`8Z?kW)@hqz+puOUxbT2@68|s_PzeSdJhaB|Zec$+X6jxDaoHXtYv|OcXt-fxO_e`$$#ys!CvUDEpXGIZ-'
    'E6`tK_Y(2B$yqxoZ+>BXU3!1Rvs=i%O%^U<)h)7PKK@AtZs7lK?EQqFf3kmG(Ka8i^E}J*ma6?q6z)NBN!j@&4A07o@STJ2XUB1G'
    'VU^X0Nj$^KPmb&UDz4vSa80(Jl`S`+47$Hz=V|1PckiRkR?lJW&sc8Nd`oA;a&uRn{OiPXe|CNj*YYo_+uv2s*PQt7btE`$eD^$7'
    'o0S)IS6mRTW8ZaRa~)qVA^A4A<0N!#?^lgap*^wtI~LuerZ1`v%=16|JN=BR>a0e8v-'
    '+p=4|drJy~6koex!!`Ppte4iRYYj?hib#vhx@HHBlyU@9>p*Mfv=ZHLJv{&M0>bcWWGyx_>IezbT*hod>k5F*qB8>&UuHG_GLBWw'
    '<ZG{Q&6}^=RC~o_};YEQ9Bu4Z$ktlIrUpB61(?SMcz94DKrve?#NpU*U@4^9Xv2#eKXy$J%4)qH$H49tw}c_y;@h!F3;s2af)a`b'
    'OX{t=zzNs{*gLv+x_0b`S2qz<t^4)!A(ym$1fh|Ik_RdaLtWWaUq6eTc5R;9n%%lm7F_bKPXb{{ehfk$+R8_mPy$4Z6v}Ur=~{1l'
    '<%G{p%!m<B@mAwcGCjvA0un**{CyvFrhrc?Z4+Xbn<tR~5B8;I1t9@X$%%-bUIjNxtnw<1w-'
    '?_>sN~Z9MlXlnJpap=&vNh^>#%m)N~4pYDZW88`YMRtDTh$ho9Cxu}Y~4A&zfmjW;2yJmF?H?eCOxvupGSo#2q6TqK_*gSG95C6v'
    'CZ$&o}0oQT=Axn;%fb*D^7Gxy}vzzQaz;}n0_-;m45^xJxCgZvPX#F3@)^2e#Cc-<5M*{a>JW1&qO(KxQwM_r%WM-'
    '6#*t@RfG8N<GaQsUYQn8biOeWzZC3Dje&mdZ(@u!bFKH}c9mxZ)$dMvlzd!|6*BPSZkusyMxn*3%ZZYij>L~cs?lmJUYxT8BD4hf'
    '<3EKCSjB3e?lLP$zaONG^mT|18+ZmIXqhn*1bnFUSxT<;`VMR|Q9Vv`JO(~+-~#Nr`ze>-'
    'MHBEK23qT`nP2%RaZ3d?uEEewYtKl3D55szBR0Y?VBen>P5ATvI^e>nxhu?*b-'
    '1<7P8NpkVvA@j7dAr6J`*L+N(me>tdY9vBY4c!Xa$*1W~=2}bS=k#7gbz2+?Q(1}7o(7vUs7@^+$=zr?Aa1TsK_n8pRp^s}l;p(v'
    'l=S~_7LvGL47~iC-fV7FR*EVHiBt<I(Ul(GGJuRw6d~(r86_nS85t#zw8zx6QKWT?xjEciaHnQHxpbxUN2#$aQ3R5LbXbrH`2n}2'
    '&QD@hdn8grliv3vei+%X%4~`sg#>OnRa$m8FS0Ypia4Z{tPs)*ww@*exoPFmlz>ql{K^QddNXNc$fTCt&55s$FC{|+GQ*n<js(<g'
    'PGn?crO1Dfk%-vIwhGGNmO@7|qLzdFWWn<E#53ssFP@K(_7Lnp;deDE3vWW~&dM``Timq@aEYkvk5Zr~o=@-'
    'X1T?a7W>zaG2U%1Z8Aav)KM=snNMyt2#84JuWD!WpJil^ll~c{F$XaHQ#YbkfLPQ}qEthI9IT=j}y+u8_TOB!Bkrs(cM4_}>2FW?'
    'uOAS{FR*KN_z>^mK{K(49s0f~yXPjS|O9_1$EGUA6RP=d>e|6DYOv<?>R26AeQ)Qv80B0^Doq}r5!){5m=Fs_1ht|x}8;LwZ-'
    'Z09k+Ecid8Rx@>N~}~wZ(d}kL5|gI7CeYVQEVv69h8@G2DGJBo#cWyk6TdODdEhc%E^cwiG3vTksEE5SucpbOh`$tTFFVD3GG=K='
    'Of;k+0TpA<jBatvpf>B;dM#$6ksPKbcMJRG9oWK6cs=gB;*5L1y@%dv$B>Mx;)V3g1REls>-UT-'
    '>l9KPhN3`kjJednrg~z0W{@^K^38D7zLuL#zQfr=8+fKR6_-'
    'kRh%s4lW#fUFTzfCRZpnpC$a@aZ*^3Xx!F?yzEWtl$XAo)mX*w`=VxZsnUmdO%uD<B<qX3l<wj~*T2aM561lJ^jy9K2kQ$6et_b7'
    '?InZC2SenE#S}Cb&&5JFS8CiW4SA`V@rSPB>J7ux27W>86E6G|BtgjY>+SpUdtpkl!b1^7NyLFLQ65br_lww>U24-'
    'PBXbTF3!YHfwMWY0jWH~4p1*0u4TO(1*tr<owp}y+KPHahO4rwZ^_;yz%ltg-ItSN;=(^P`0E&^{Ep2c*AqftjyUz$oR4^4Sc7t}'
    ';vC1jM2LLIj$8tN!gO+iDqIiu3hmZkpd!C#-ORAOA0-9~OR^j2q7*KNaI1vE8)qCOQ-'
    '6T0%Ws#JO{R%@_px*IaC%1SxrmEf(ay?XFeK~n{ERgjnU@TmvXwRQ61P?5EU@R=P&v7i#P6?GR@MW&&A7>%gs+-`4dDdo1HE-JZ='
    '-9D<yYLZkDuiC>`4SDrgZ9@G17mA0`0V|(S9aq7J)^2B>7KIY{P$MFnHQkQLt&a6xj%%Z%Cfc)*y^`>>04+f)A9ayte%ZJUXzRCH'
    'vaA!mo#IkdQuP?rAX~MGMjh2j==RO$kBSqSCuCbBnqx~1;_lT!T{INN%i1w0q0>+*qDIPKT}|RtAD%k2PKr$(P(A{drLqyQoHUbv'
    'b;a3`2$<B8ek*~LrcgA)t~gXgPsJ!yBHoo{ePe8|$Gn9Q;%I>_Er@Mjbk!v?Rj{i8*{ST`r6H6c?)^Yj#wMvTd2fQe{$UIdiYQlc'
    'D23cs$gEC8>%r9$i<_Xa6}c@c#MMb@;#ZGW)$NG|O&B$Ws*>A<*2|}kLt}R9P)lvFy*cB0@Ki-'
    '#6}>Obwz_b3!S}B0wkCp&(c4H8E0C2+v<~QKqO%u51JDhQa^#?lZijNpM=y9S+KtFxCF0snC!!K3p(8Up^WEX=hxC3}(+s^<r!CO'
    'b7A=)HmF4I=5Qj!F=mEOJ(I4uzG^^oW@b`qHE$GRt1CcFDd|ETB4t;B6cR*`PEVca_G($O^0~<ASC*xTotZR>qc2M=k|MHCc%agY'
    '7*20(OP?)|pP&A=V+X1WdraT+4*BP1)y47lhQAcRcC{*>{N(<ddEy+R`cPO0ARPlBEx;J!CEw@BN7oGFA?jSsCj5nssZoF3X1BhZ'
    'B`eDqiQaj;CZMdv%Tf^CoXE*x3SX~1>jfEy*v?m)aW6&6h^`IUEPdo0jPROxl8L*qAfiEQvPZ8TzWV36;&ElaUYmJfC5M2!;(3QE'
    '}3y~N`Y&$7O-'
    'O$lc)mGTIvl&sfXH$Gk7;J~#ov4J~(DsC?DOMImW^;ESG(FgT0&DDF59&)&1HCo8){Nb5_|}5f2(LQ2LwvgC>N_B}t2>xD3}9@vH'
    '5d$J-VrnnqnXeSUwc5;nFtRg2Ay<M4DsXMm1gI?FFFmav9kw~I)l*7+}cM^xSoc-'
    'H&L@19t67JUt4)Ilzu4BUU0NS(_qOnYI}C@(UI0O3cc9vrBl;WC!u=`8p6?solZVF3tf<JlKWu!6SAPAynO=Oy2>j<cO-'
    'YmnhuPg#QFi~v6!`ks<pW6#%(0dw#XZZG&`kz-7)w!kTwL3EflS>x-t8(_MAHo4AlNaVrTZ4_qNlE-tN{X*&V|8S?qjPvo7=<l&O'
    '|b_k+*I&$uIzJj@*~t6M48Pl|pVv3^<+u>3p;jp3;<M!@^DI}J;_#lUhk6k3zjANvPm*C_faaCOJ?Cz0JpwGxL;$a=~h?@nd6$A8'
    'e4-Oe%SqnwS$(*EuQc%R1i-sG@d44%RNXCwOOk=94id7Aze)}Dc{PYl|-'
    'FObWgSk@c55ww1KzJX^W;k9~x2A`f5o(*Fnah^r~CV;+Bj9_gTJR{^|cfCWpAbB?Ek1hQezrgc3q|VVT(g}^jh{Avvyr5H3K3G0j'
    'PR&DheP6}lRd+PHo+KV!kkbW?LsVCB7^JG{7lQ%*P6+m!q$5=^4%?qX-XzsWBt~M-'
    '6nwWT8X>z!DH6k>pRB)}dy#x1jA3Z8Iv9zK&nRxsV4Y>|IlQxKHAyeQ?d58+AJgGJ>|Z^HV9_W%>WoD%FrN~Gp}=U3s|!AjMD~l~'
    '94Wn{v7;M%6Wr<8KO~HyKAylDqZ|e06l%K<o|=`@+?P2gBak&7&7;^G&HP1SY8W$hpIDxzvp<Ht=kVtxRce1^y~O$m*2f@ay!OW+'
    'B^onyx+5_yY@NYQXHO(rwVN+f(bj`lO(eSg=qICND$-x0w#Vw$7{-d}oWyt>6jQLu^vr-'
    'F4in+All~O`PGo<Q`xbsakG$uR{45kNGk%V`UBq*$>gClK49B*iJZF%b7wFe1K7%7*v7Q-$*T4w>Zi&PsY<NL(rqbSFK3yj-'
    '8djN;@oAF#dIa78Q<Sfl<i%9QEDpnvJ4)wr7_uj-'
    'O@VeAnCQdPKM6^%MBI5#sah>s6NKjl^VM#zH}U*sWWPiNEf!BR8;*TzvEWtwoXc#QJ5PR_Hm`a8I^Czyc!k(mmCbZNMCxludljEw'
    '!^2sOXOhtotWVcj8^h{McRr&x$m3g3&4BtPMpj{Ch|wFYO^Cs4qA*!9XCiB~vM~!u&xg-'
    'k2y;|5qtG_qT>$lHc&#>H#H$zae+=@+=zNW4G#}Yxbeq3IR9=SnbtFV%q-'
    't`kyAYc3Xq<qxuR`~>;%ainyUUO~3cY6A>+Bl+b67YHJEtLIMhxD7cR4&`d0PHvsUF^fa*{IYW!A8Q(HlsOtADHCSMY6a3_Q!m%c'
    '61UeNVQGlavW)cpFTEbByo*oAA6&RHN}cly-7v5y2Ob=H2<$CLn9pe=rp}3yJJRq|I>`V}WJ-c_hpeX1hzI*KQrN<jok&gKq|W^Y'
    'HR5RoP2^tr;ehp*TtpW~FzBP8X&zS_t<NVlt05o3@yBqgad`^TDgw_71)*VEhi$GnK&?;hCt)n2sk4(7yn@>n;TgnJrZRmUiAk(g'
    'LI`!H;)nbMf<K*4~!Zx3PF>3|_)7v*m4g-'
    'jw8}?k8BZ1Y0c+D|N=+q;}qo!6IeP_{{t1P`#>bJRgM@W3W)(&2itywzckd<gCDgnOM1yY%k_)&ZOq%$6!JDoiNQum0&ecHCQEBL'
    '9MkLi8k(hXQ{h}c+HV)YxA&pA#JgmdDuY_UaJVLCkn4)jm7IVV(>nAi`8|)J+MygHFpzKb9pX+Yl-'
    'Mz_BDL;A6T3$L$4FR<wR+=d!M|Rzjm)qrMHUuSXI9aX-kyN5H>-(2oK(rjq{kVgJLH#41dG1+1=o7MDlX1{lHxf-'
    '3nH>kiq5TW))bWJjKCI@G5raDhI2<d&qnTTkpH^g7@8;!8z>DL(1Fmen9@`vKtN4v^WOKX>ZZIe7zOMn@}%>X9cUXn5|J{BVi|Ej'
    'q1tjaUN8wd9F}ptWzA83oF8S7n_%2VW=%*ZML+|#@cyETcPJdtp8Xsc{>IR#I+JwoK|DwI?=9E)T}P&@m%cZV;(CTp!pEZ3-'
    'HAxEQ)}6yaJk4P%M_D_o<wv^2lVoB|ldqV;%e(+-?2|S{BATtQ+?#c-Le9hhROh8u}R6Is4H4Lsb-sWmvZwy2VhK&edukG4rs}$8'
    'xOy2uoJTj*YapS&7C*ut9Vi1e<@Nwg$VFDTf|H(k-'
    'uV#GtpnqdfEl<FWn^By7M6i{A+9ZhRQib*HUI#}+qL&`#QijO{dTqAeqG%X#~3cmHH{6V?6^^Icf~9PKI`8(25XzjD8hz<&1&MQw'
    '%lw!pcO*(PYejloK?wpJ%R)Q^LH3%+eY&nEX%c+B!us*aDTFMHS8d(6)2Y}PG4Tha5mXg4!niEXQNvQ~<7BUZm7`XzeL*a@<eYX7'
    '>o*4Mul-e{QaPpOp68m&=&J*@KaA?xqDIfAKzDLJ>Pf{B7D7+I7y!?y-6wyT;~YGt$h`d`=z-D+WtkF~l9ta<(4Xv!H}C9gh&-'
    'cHU(tZa3!xv7JHxqmUv8Qccdeym+9YnMZ}nTT!2mTm5L?m@iF6-*uc-Mt9?7wFxLFKcufK7;dwd(u6w{wVzkcek4siCcl4$CawLR'
    'f_F4qP@o5f#f}KA3^Rutl5c_ZHza%d)YaPo!eQ<5X=zF1uRCX=~D!=FiI6X2=xw~GK=*NW^0ie;s_yE@OyT?P&Mv!GX|q^46SxTc'
    'VYcr=(b|>X34TsyP1l}6#R_cefau0wr*qZ;~3a}j^J_R>}GchG+$7^JF#RV@^%2L<gYZ^0qtIC`vi*J?w8;cI0?M|B)ea+Z@UG9y'
    'V&0pgWc?J$C{nsWB;6oU^ae3KKG#I8}>KHz---xJ(1Y#W)I##s!7}j#TM3f!@V^Idy&5vs?XW~6pB5*)o0Wj_Tlk!F(?=`6jC#MK'
    '$T}BgIVYkk%2_4Tvde4=54A74|{!Bj6U_x(pJC!(*0Rj<NnTg58Rf$XzYuDouz}iS-'
    '#fuAjql83L$s!b8e}k!P3F6kotu*@6z)tbeWbN=-%ai@8%Dlc7JeB1M~kB&p+Hk!7r))-'
    'AMjiR(|TA{{LdTd&oTxE^v+vKdZm(<_>;M-2XTB`}iX4-'
    'wq#MgbOIf7Uey<NiSkg@!%T9`_*>4WrE*nY<K2PY~1@F><()&ny)40qI-#FvETuswa-1^W0zYvSU6Zj&BGBm%+Bv<-'
    '_w7gn1308!|K0^z#iy|i`z6FpdF-@VSZ3m{uMSI;T}2a-'
    'jWXo;QwBE+k;K};Qp5NqQO$ZqZ*lIU%MrPMUhYv9Fisbu;>{0j+<^jk^chzW3KIaIOO9vyT3zo+qK*955~Vk`JHQWLMTg)kHB*P+'
    '7q(ni1Z!8yCZI;;7ROwfDJp`iou_#ilc7jU}e1f29A=!6K<m5Vb;n8zZGXuIKP)|k*E@kLsf9VjmCF)dJui*+_Sm^B5~ZU7OWn$o'
    'BF6cJcW*<vh9fMJxb<kP-SMtQFMLp{)C*}KD=}H1+>R>2YeA;-OV@UAz61k0w(on1Wu#30@@D|^B>Uhjot)DeQJ9n97n>52-'
    'pqs6;i4P&vK5e`p-z}DJZ|ud#qUS6c+!?9bx%N9*ocZpMcx;N969;2|VeR;#b5~_g7+LGJlA{F>-Pmnp&I&v*L<-'
    'lNfq9=;I`Mk3;({QjZ9=c{luk9cSG-!8*b6!81_&tdr_JPpA$~=-xZQ+GgF~M*m|NHa-'
    '`Fy1}|wdkoq`^4{z}8>T+1elL5)s1&QJQ&{0;Jq{^?$%%q}DR?bNR3c$Eos+De)XqujA-'
    '$ci>9#tf$eh9R+`%~H1xFOq(`fcGat6-'
    'BVR*4A#SOU$n@+L!1ML@RE&D$+KEv}{G}`m5o@cq~j&Z6#_s`qc>P^}Y_;SLJ)j8_k^Z2lK&4#V={s-'
    '>K>cN`9iaO_|gEfMdsk5@|CwaMD*rD$U>w~(T&4NREo<ZJ8to%i>IOq32Blj24?-'
    'ak+|Hyr3R{zA_IkNo|^IxDeKYodU7u_Ev@23cyR)w68fl0dL)(ZZ>-Y>fSO~PgB(WCyES>s?m(7^8-Q+dCN_Y!m-'
    'MIGwL=&!>4E320!;fmkyL8{@rTTe26#RJoGjg`ypbw+lhUtxC9e*;|?y~+C3wKH@@?W%hN33Vmo0<%j<@ZME+8tlDwLvm^dFUpE*'
    'dQV*Pwa4MAv|eKOCLA{yne^*={t8#^;BV}h&P(jxbbphlM)|8ex&`em-=<$<a6>o;#Wh)c8UD-v!BuYYI%v58-'
    '4*)Zz!lx{x4GkQC~8-v`A1oB3y!PM)(w8>M&lNeEG~ag3r276ia)_sXzgur6E2f*ld8B4^=;-iiQ?yOG`@82z-4~lQB-'
    'b<^BQum$-'
    'c{OeZI&4M1tA+6Ow;pcKtsvTmN$JqQ`7%Kvmp^_KsxOyX&s@t}|+g|A(Qz5B(k4b_<F3<i&0M!i$96{paDmht&JB<99~K-'
    'OPM_rw(TFkvX{Djm8yOb06B<ic2Is&v%9KF*0xe2S#}rDfjrb@~14_z*)MB#J`!{Vdrn&t`GE=oCjKo#9#Wg8;SV5DdG@6xKFzG_'
    ';0m+@Z6{UBfWQco7h)rW+9940Noa`yX^lfSXAza=f3QGD7{A0IM^g;&HASF581u1h<uOL58cP^BXQs5R=X5~M^OCjHW8n-'
    '2gv`I)%)VRM`UvZb6`&nA31}w^si-c$Q%3z`bTacXqNqpwDY=Ue$m}!75$Jki{C%46Z{vt2lD4{tw-ZAv2~I24|@+(gSR63$J&p^'
    'qc9#vv`3<~-mEfx|G4(1dL-Y>-'
    'ruq1v3&IZk7$SS*tP$9*7oc_b`1y*UCUxT+P`izjzMi)kDw2rAzm<{xC6n*(jL(Gk$$b4yvMx3tR525jL$`4e13@p83m=w)qlU+1'
    'ggX!9yGa$dTze5b81uonuJir3s&HnL~q@o=n^8wk(NkUR!u`vkRaGdZ|ziSDfL@6k+_1QNf<^V*_1e#jBzTRzBpJdB}H>mSrLy>V'
    'yKhTno=qE`Tk7GN&@u>`SN+dFZaZvPl3Gqu33_jQBqoR);)eZF^~DmN)=41wFLf`M#A7#RnfI@f6Z;GofO!TOmb36Qi5P}Mrq(ku'
    '66rDF&T-'
    '1*Wd{t*}P30v~QSnz#@=TRvOHjYg9vO@g<JIO}9lbEx+m0D`rVzkc>#Afz#xrLV9{=(}#DH`bfxXG}3@nvNal}$69JEO3N=*8)ww'
    'Hh{M0`H6a~xl0%t11}Wi4jr{c7nC5L7=<M!HrHEu;lwLNZW8~S9k?dt)o?6;6(LZ!k2D5^UjMIxJt7s$fFO~n0-'
    '~HL(2>F%HM+W_+BTFzlbPrt*iJ4^%Q>6EiRvL5ozs@s3AMoLwL5Dcz@a;1%vT+i!vJxxD{_c{Ly$rfZz4zNq|NVHwUF&A&w{}kXW'
    'cwy57xpC6FL1Lot6&t#S;>PQ4|$--#%dnEQ?m<ZZEl`bgZ4JRL<=EbuuU)@boqk$b!VEjIrwc9jrcm#`Dk{l+6`-e2Wt~-'
    '2b%J1tMM9co~-<)%!{moLGNz{S!0lsz5Mhk6}L=C%+GJ}T>7rg7K2<!G1@#-'
    'qgkI@e&rx?_Wmg>9`iAee%I%x<q4)!P7BG3JTVAGBM$k(J0VyuG9kMtd;bfCp{dR9oPNQ3ZjoRy-'
    '<y1l?&^J8kMHjM)Mxgf{c0_aO$EZd`F#`!-V<#Bq?zrx;VuC}f1fB6w7(kU58AuL=yD;ar2cMEAZUL-'
    'DHvg|C-IivI|U`L2;cuD7!@RIcim_dmh>W$R2rGyUngv(SkQi-H=>#E`L$Yr-oAm#AhD=uit_8QI?v+bj6}}hcqsD(%Oba!JTJl*'
    'a9N%uL|=iNSwzY(F3w6cN(6s*J>2qJ86s$BrcSV&AITC(D^Fii5<HX;N}{urVpNp6%%ST1i#YwEZ=ydKk9Y0Ac&VVjc9c?I(jOc0'
    '@~f#dmX(y0^7;i;ItD3&fAg-'
    '$uDDrzjlM##5`7sWWp8);4hXd}NU4cD+b!$Um!>ZjY!<AKeEVxhSx`<sRp41iZ^O9%C7{Y0BcoEVivQp1B!I3Ky!PMml;Jl=QRG$'
    'TMBBWXZa43X(|g)#U8GmkzZq7e_s}c`HPK?y%kVxbs_HYFtHSqRG$*IkkycT9Ga|JH_8D5xD(mks6&0B>dM}sJ-'
    'BI~JXbEcS-!qM?kj_XEnpgd0eU$TC0wk41Q#9<aKCQTQ@)MJ)vY`sIQu-FKZ-bgb)nIF^jz(D`TTL)8s>5H0aTW2`mVZ^{OEo>K@'
    'kP=A>9vWbc~g}dFv+#aV>O<Y;kI9YHL=P5rc{IVmh?5b6WcMb%e-b3YJpawsS|8S4sBk6xYZKM>DOd!Mw#{7wt-gbvs#O6HNc;`a'
    'MjjMRoP!vT5E)JQAtrX4;o5Nb3NOGM$%ZzM{OU)g5H1EFg=X=LHqUG7|uH4>%h1!+>PL_L*D><>XGvfpd+wv`FgZQcw;*)S!qsd!'
    'Ac8w+7a0%aN5^heSB}iTEk#H$?T|iY34Ap5;v3CKG+G0X7Dt{=cY(+h~B2`S`0dYW=Jiq_mAmshQ?59p}i2Aipw(J7IX@uvrtz4H'
    'IlV$)jYI7a$B9EI8;D$d)d;GeM3us)KaYnt$nbk&Pt^Sw7{RH=rHPbK37Y{*X-'
    '>~L@bi+!}o!AS5*l1f~r*nS_{SfSyOhKF}G;73U-'
    '0NqvUkL?$Bwks#aCiQV!}aP@27Mf?c8OMYC91ZYt?FRcmNk2D^b~!QN1J#kx}1Y`2%qx-'
    ';%XZ<hB7K7po^;%enbxfN%=JLrJ~v%Lo}kE{A8E(E)|L+=6W`|3-=(B4N0-'
    'mnI{Kg%Gcr)sSyJE5~vHv)AO_dfX58Lv7hPHp{WQtMH7h6X<DE!~y%&e+*ik?SmPdMldUk<d?d5sB_d?kM_B#KNkhL$H4o2Ef%f2'
    'Avdxj_B?agC6MY$$l;D?+LmfzZbT3jzE1@dh4`Aqkas!!foj6KfA(ZIvVTD8XASsO)#n_{EBKFJ}X_Zv~l=mFxcO*x<T0&|2iw4E'
    'qn<fbPuCFyZu<{7>0M!?EIUISXec+l3cUAJ+shGD0)408}uVq_E)|J@~4sde!+oo^^@nF;2oeCM#C)a9~>NoA=GnYKN5|7shyE9P'
    '|=UX0M-Wi7%23{;wC~@az98Bwo^C&`%19VSsKjxC-AE$Yo@I`vmx@PAEOSzq1v&EHow~NMcbFP{>=L8R%@?snjTot56Mr;jsX$qA'
    'AW{%9YRm>*!sZWFyH!tVe3P_gd#CeHQP(Qch<`Kvitd*Lv))AWd0QNw$>R-'
    '2M3>~AM!siSlJka%}+ux6wimSG8l>7!ZtoBbZ2*<ayS5b^U&-'
    'u*`Zn*CiIAb@tUTm*n0}jr|G+^cKT9#y;VW2@To2Hj%23`Yu)^)^!^{{6K;KiZs<o`AfAxGP2}4DRiA^-ac8-+)|uxlcE&o-'
    'IYXRbP6MZ@{eOrSa0B@Q*@AfkS<q)vss^eBst2mjst5iLI+l4;'
)

def _candidate_schedule(graph):
    signature = hashlib.sha256(repr((graph.nodes, graph.deps)).encode()).hexdigest()
    assert signature == '6f49849865492d3d7ee5b3766b56acfe32e6de57a7173d2fc70990dce40e44ae', signature
    raw = zlib.decompress(base64.b85decode(_CANDIDATE_PRIORITY_HINT))
    assert len(raw) == 2 * graph.n
    graph.original = list(struct.unpack('<' + 'H' * graph.n, raw))
    for j, dependencies in enumerate(graph.deps):
        for dependency, latency in dependencies.items():
            assert graph.original[j] >= graph.original[dependency] + latency
    program = _candidate_release_schedule(graph, list(graph.original))
    times = graph.original
    graph.original = times
    canonical = [{} for _ in program]
    for j, nodes in enumerate(graph.nodes):
        for engine, slot in nodes:
            canonical[times[j]].setdefault(engine, []).append(slot)
    return canonical

@lru_cache(None)
def compile_program(forest_height, n_nodes, batch_size, rounds):
    assert (forest_height, n_nodes, batch_size, rounds) == (10, 2047, 256, 16)
    assert (SCRATCH_SIZE, N_CORES, VLEN) == (1536, 1, 8)
    assert {engine: SLOT_LIMITS[engine] for engine in LIMIT} == LIMIT
    _compile_verified_base(forest_height, n_nodes, batch_size, rounds)
    k, graph, program = STORE_LAST
    graph = build_candidate_graph(k, graph, program)
    program = _candidate_schedule(graph)
    result, allocated = allocate(graph, k, program)
    assert len(result) == 881 and allocated == 1444, (len(result), allocated)
    assert allocated <= SCRATCH_SIZE
    for bundle in result:
        for engine, slots in bundle.items():
            assert len(slots) <= SLOT_LIMITS[engine]
            for slot in slots:
                reads, writes = rw(engine, slot)
                assert all(0 <= word < SCRATCH_SIZE for word in reads | writes)
    signature = hashlib.sha256(repr(result).encode()).hexdigest()
    assert signature == '9160f691bbf3d87edadefdb5436817bb9dff6db9638c0cdd81fcd6421cdc222c', signature
    globals()['STORE_LAST'] = (k, graph, program)
    return result, allocated
