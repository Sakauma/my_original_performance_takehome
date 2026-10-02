"""Single-file 891-cycle compiler for the original one-core take-home machine.

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
        self.logical_counts=__import__('collections').Counter((e,s[0]) for e,s,d in ops)
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

@lru_cache(None)
def compile_program(forest_height, n_nodes, batch_size, rounds):
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
    globals()['STORE_LAST']=(k,graph,program)
    physical,allocated=shape_allocate(k,program,'area')
    physical={base:reg+8 for base,reg in physical.items()}
    physical.update({base:0 for base in k._initial_zero_bases})
    allocated+=8
    assert allocated<=SCRATCH_SIZE,('Static scratch allocation rejected',allocated)
    k._final_physical=physical;k._final_virtual=program
    result=remap_program(program,physical)
    for bundle in result:
        for engine,slots in bundle.items():
            assert len(slots)<=SLOT_LIMITS[engine]
            for slot in slots:
                reads,writes=rw(engine,slot)
                assert all(0<=word<SCRATCH_SIZE for word in reads|writes)
    return result,allocated

class KernelBuilder:
    def __init__(self):
        self.instrs=[]
        self.scratch_debug={}
    def build_kernel(self,forest_height,n_nodes,batch_size,rounds):
        self.instrs,self.scratch_ptr=compile_program(forest_height,n_nodes,batch_size,rounds)
        self.ssa_peak=self.scratch_ptr
    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

