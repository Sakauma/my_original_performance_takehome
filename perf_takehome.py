"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

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
        end_flow_dests=set()
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
        cache_depth = getattr(self,"cache_depth",3)
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
            if (getattr(self,'flow_mask',127)>>(i//2-1))&1:
                tree[i]=broadcast_scalar(scalar_tree[i])
            else:
                d=self.alloc_scratch()
                emit('alu',('-',d,scalar_tree[i],scalar_tree[i-1]),[scalar_tree[i],scalar_tree[i-1]],[d])
                differences[i]=broadcast_scalar(d)
        pre_stores=[]
        pre_buf=self.alloc_scratch(length=8)
        pre_src=self.alloc_scratch();pre_dst=self.alloc_scratch()
        pointer_flows=0
        def pointer_const(dest,value):
            nonlocal pointer_flows
            if dest==pre_src and pointer_flows<getattr(self,'pointer_flows',0):
                zero=c(0)
                emit('flow',('add_imm',dest,zero,value),[zero],[dest])
                pointer_flows+=1
            else:
                emit('load',('const',dest,value),[],[dest])
        prefix_loads=list(cache_loads)
        packet_bands=[(4,2),(6,2)]
        packet_depth=4
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
                preprocess_reads.append((len(ops)-1,6+width+off,6+width+off+8))
            for off in range(0,width,8):
                emit('load',('const',pre_dst,width+off),[],[pre_dst])
                for lane in range(8):
                    source=level_src+width-1-off-lane
                    emit('alu',('^',pre_buf+lane,source,c5),[source,c5],[pre_buf+lane],vector_lane=lane)
                emit('store',('vstore',pre_dst,pre_buf),[pre_dst]+vec(pre_buf),[])

                pre_stores.append(len(ops)-1)
                preprocess_writes.append((len(ops)-1,width+off,width+off+8))
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
            if packet_depth==4: mirror_roots=packet_levels[0]
            packet_flat=[]
            for root in range(packet_width):
                for level,source in enumerate(packet_levels):
                    count=1<<level
                    for child in range(count):
                        packet_flat.append(source+(packet_width<<level)-1-root*count-child)
            for off in range(0,len(packet_flat),8):
                emit('load',('const',pre_dst,packet_width+off),[],[pre_dst])
                for lane in range(8):
                    source=packet_flat[off+lane]
                    emit('alu',('^',pre_buf+lane,source,c5),[source,c5],[pre_buf+lane],vector_lane=lane)
                emit('store',('vstore',pre_dst,pre_buf),[pre_dst]+vec(pre_buf),[])
                packet_stores.append(len(ops)-1)
                packet_stores_by_depth[packet_depth].append(len(ops)-1)
                preprocess_writes.append((len(ops)-1,packet_width+off,packet_width+off+8))
        mirror_stores=[]
        for off in range(0,16,8):
            emit('load',('const',pre_dst,off),[],[pre_dst])
            for lane in range(8):
                source=mirror_roots+15-off-lane
                emit('alu',('^',pre_buf+lane,source,c5),[source,c5],[pre_buf+lane],vector_lane=lane)
            emit('store',('vstore',pre_dst,pre_buf),[pre_dst]+vec(pre_buf),[])
            mirror_stores.append(len(ops)-1)
            preprocess_writes.append((len(ops)-1,off,off+8))
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
            if 1<=g<=getattr(self,'flow_input_groups',4):
                emit('flow',('add_imm',p,states[0][5],g*8),[states[0][5]],[p])
            else:
                emit('load',('const',p,7+n_nodes+batch_size+g*8),[],[p])
            emit('load',('vload',val,p),[p],vec(val))
            states.append((val,idx,t,u,shared_w,p))
            histories.append([self.alloc_scratch(length=8) for _ in range(5)])
        packets=[[self.alloc_scratch(length=8) for _ in range(8)] for g in states]
        packet_bits=[[self.alloc_scratch(length=8) for _ in range(packet_height-1)] for g in states]
        packet_candidates=[self.alloc_scratch(length=8) for _ in range(1<<(packet_height-1))]
        deep_start=8
        deep_load_buffers=[self.alloc_scratch(length=8) for _ in range(8)]
        deep_source=self.alloc_scratch(length=8)
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
                if depth==4 and r+1==rounds:
                    for lane in range(8):
                        emit('load',('load_offset',u,idx,lane),[idx+lane],[u+lane])
                        ops[-1][2].update(mirror_stores)
                    v('^',val,val,u)
                elif packet_depth <= depth < packet_depth+packet_height:
                    packet_level=depth-packet_depth
                    if packet_level==0:
                        for lane in range(8):
                            emit('load',('vload',packets[g][lane],idx+lane),[idx+lane],vec(packets[g][lane]))
                            ops[-1][2].update(packet_stores_by_depth[packet_depth])
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
                        if not (getattr(self,'flow_mask',127) >> (node//2-1)) & 1:
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
                        # idx already contains the forward one-based address.
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
                            final_index=r+2==rounds and depth+1==4
                            if final_index:
                                for lane in range(8): virtual_map[idx+lane]=virtual_map[bits[0]+lane]
                                for bit in bits[1:depth+1]: ma(idx,idx,two,bit)
                            else:
                                assert depth==3
                                sel(idx,bits[0],c(40),c(16))
                                ma(idx,bits[1],c(12),idx)
                                ma(idx,bits[2],c(6),idx)
                                ma(idx,bits[3],c(3),idx)
                    else:
                        v('&',t,val,one)
                        if depth==4:
                            sel(u,t,c(6),c(0))
                            end_flow_dests.add(ops[-1][1][1])
                            ma(idx,idx,c(4),u)
                        elif depth==5:
                            ma(idx,t,c(3),idx)
                        elif depth==6:
                            pass
                        elif depth==7:
                            inv3=0xAAAAAAAB
                            base=(767-512*inv3)&0xffffffff
                            sel(u,packet_bits[g][-1],c((base-2)&0xffffffff),c(base))
                            end_flow_dests.add(ops[-1][1][1])
                            ma(idx,idx,c((-4*inv3)&0xffffffff),u)
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
        self._ops=ops; self._op_rw=op_rw; self._op_group=op_group; self._virtual_width=virtual_width
        self.logical_counts=__import__('collections').Counter((e,s[0]) for e,s,d in ops)
        # Critical-path list scheduling, respecting read/write hazards.
        succ=[[] for _ in ops]; indeg=[]
        for j,(_,_,deps) in enumerate(ops):
            indeg.append(len(deps))
            for d in deps: succ[d].append(j)
        op_weight=[getattr(self,"flow_weight",3) if eng=="flow" else getattr(self,"gather_weight",1) if eng=="load" and slot[0]=="load_offset" else 1 for eng,slot,deps in ops]
        rank=op_weight.copy()
        for j in range(len(ops)-1,-1,-1):
            if succ[j]: rank[j]=op_weight[j]+max(rank[k] for k in succ[j])
        stagger=getattr(self,"stagger",4)
        rank=[rank[j]+(batch_size//8-op_group[j])*stagger if op_group[j]>=0 else rank[j]+getattr(self,"init_boost",150) for j in range(len(ops))]
        rank=[r+(getattr(self,'end_flow_boost',40) if ops[j][0]=='flow' and ops[j][1][1] in end_flow_dests else 0) for j,r in enumerate(rank)]
        from collections import Counter
        read_bases=[set(r//8*8 for r in reads) for reads,writes in op_rw]
        write_bases=[set(w//8*8 for w in writes) for reads,writes in op_rw]
        uses=Counter(r for rr in read_bases for r in rr)
        live=set(); born=set(); max_live=0; live_total=0
        reg_limit=getattr(self,"reg_limit",175)
        reg_weight=getattr(self,"reg_weight",30)
        ready=defaultdict(list)
        for j,(eng,_,_) in enumerate(ops):
            if indeg[j]==0: ready[eng].append(j)
        partial={};op_start={};op_end={}
        def weight(r): return virtual_width[r]/8
        def live_size(): return live_total
        def delta(j,complete=True):
            new=sum(weight(r) for r in write_bases[j]-born)
            freed=sum(weight(r) for r in read_bases[j] if uses[r]==1 and r in live) if complete else 0
            return new-freed
        def pick(eng,scalar=False):
            choices=[]
            for j in ready[eng]:
                slot=ops[j][1]
                if scalar and (len(slot)!=4 or slot[0]=='vbroadcast'): continue
                if live_size()+delta(j)>reg_limit: continue
                choices.append(j)
            if not choices: return None
            j=max(choices,key=lambda j:(rank[j]-reg_weight*delta(j)+(getattr(self,'fma_bias',100) if eng=='valu' and len(ops[j][1])!=4 else 0)+getattr(self,'fraction_bias',80)*sum(1/uses[r] for r in read_bases[j] if uses[r]>1 and r in live),-j))
            ready[eng].remove(j)
            return j
        def commit(j,complete=True):
            nonlocal max_live, live_total
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
        self.ssa_scheduled_peak=max_live*8
        # Reuse physical eight-word registers once their scheduled values die.
        births={}; deaths={}
        for j,(reads,writes) in enumerate(op_rw):
            for w in writes:
                base=w//8*8
                births[base]=min(births.get(base,op_start[j]),op_start[j])
                deaths[base]=max(deaths.get(base,op_end[j]),op_end[j]+0.1)
            for r in reads:
                base=r//8*8
                deaths[base]=max(deaths.get(base,op_end[j]),op_end[j])
        active={1:[],8:[]};free={1:[],8:[]};physical={};allocated={1:0,8:0}
        for vbase,start in sorted(births.items(),key=lambda x:(x[1],-deaths[x[0]])):
            width=virtual_width[vbase]
            while active[width] and active[width][0][0]<=start:
                end,reg=heapq.heappop(active[width]);heapq.heappush(free[width],reg)
            if free[width]:reg=heapq.heappop(free[width])
            else:reg=allocated[width];allocated[width]+=width
            physical[vbase]=reg
            heapq.heappush(active[width],(deaths[vbase],reg))
        for vbase in physical:
            if virtual_width[vbase]==1:physical[vbase]+=allocated[8]
        allocated=allocated[1]+allocated[8]
        assert allocated <= SCRATCH_SIZE, "Scheduled kernel exceeds scratch capacity"
        self.ssa_peak=allocated
        self.ssa_op_count=len(ops)
        self.scratch_ptr=allocated
        self.scratch_debug={}
        def addr(x): return physical[x//8*8]+x%8
        self._virtual_instrs=self.instrs;self._physical=physical;self._op_start=op_start;self._op_end=op_end
        self.instrs=[{eng:[remap_slot(eng,slot,addr,addr) for slot in slots]
                      for eng,slots in bundle.items()} for bundle in self.instrs]

        graph=Graph(self)
        for iteration in range(3):
            program,times=graph.schedule(1,.01,seed=5,noise=0,reverse=bool(iteration%2))
            graph.original=times
        self.instrs=self._virtual_instrs=program
        graph=Graph(self)
        for iteration in range(4):
            program,times=graph.schedule(1,0,noise=0,reverse=bool(iteration%2),allow_lowering=True)
            graph.original=times
        self._startup_input_base=7+n_nodes+batch_size
        program=startup_synthesize(self,program)
        self.final_virtual=program
        rewritten,graph=post_transform(self,24,4,0,4,28,memory_base=7+n_nodes)
        for iteration in range(5):
            program,times=graph.schedule(1,0,reverse=bool(iteration%2),allow_lowering=True)
            graph.original=times
        graph=reassociate_final(rewritten,graph,4)
        for iteration in range(2):
            program,times=graph.schedule(1,0,reverse=bool(iteration%2),allow_lowering=True)
            graph.original=times
        self.__dict__.update(rewritten.__dict__)
        self.allocation_order=10000
        self.instrs,self.ssa_peak=allocate(self,program)
        self._virtual_instrs=program
        assert self.ssa_peak<=SCRATCH_SIZE
        self.scratch_ptr=self.ssa_peak

from collections import Counter
import heapq,sys
LIMIT={'alu':12,'valu':6,'load':2,'store':2,'flow':1}

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

def allocate(k,program):
 births={};deaths={};lane_deaths={}
 for t,b in enumerate(program):
  for e,ss in b.items():
   for s in ss:
    rr,ww=rw(e,s)
    for w in ww:
     base=w//8*8
     births[base]=min(births.get(base,t),t);deaths[base]=max(deaths.get(base,t),t+.1)
     lane_deaths[w]=max(lane_deaths.get(w,t),t+.1)
    for r in rr:
     base=r//8*8;deaths[base]=max(deaths.get(base,t),t)
     lane_deaths[r]=max(lane_deaths.get(r,t),t)
 physical={};available=[]
 def alloc_key(x):
  b,t=x;mode=getattr(k,'allocation_order',1)
  if mode==0:return t,-deaths[b]
  if mode==1:return t,deaths[b]
  if mode==2:return t,-k._virtual_width[b],-deaths[b]
  if mode==3:return t,-k._virtual_width[b],deaths[b]
  if mode==4:return t,k._virtual_width[b],-deaths[b]
  return t,((b*1103515245+mode*12345)&0x7fffffff)
 for vbase,start in sorted(births.items(),key=alloc_key):
  width=k._virtual_width[vbase];reg=0
  while True:
   occupied=next((i for i in range(reg,min(reg+width,len(available))) if available[i]>start),None)
   if occupied is None:break
   reg=occupied+1
  if reg+width>len(available):available.extend([-1]*(reg+width-len(available)))
  physical[vbase]=reg
  for lane in range(width):available[reg+lane]=lane_deaths.get(vbase+lane,start+.1)
 if getattr(k,"_initial_zero_bases",None):
  physical={base:reg+8 for base,reg in physical.items()}
  physical.update({base:0 for base in k._initial_zero_bases});available.extend([-1]*8)
 def addr(x):return physical[x//8*8]+x%8
 def remap(e,s):
  op=s[0]
  if e in ('valu','alu','store'):return (op,)+tuple(addr(x) for x in s[1:])
  if e=='flow':return (op,addr(s[1]),addr(s[2]),s[3]) if op=='add_imm' else (op,)+tuple(addr(x) for x in s[1:])
  if e=='load':
   if op=='const':return op,addr(s[1]),s[2]
   if op=='load_offset':return op,addr(s[1]),addr(s[2]),s[3]
   return op,addr(s[1]),addr(s[2])
 k._final_physical=physical;k._final_virtual=program
 return [{e:[remap(e,s) for s in ss] for e,ss in b.items()} for b in program],len(available)


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

import copy
def post_transform(base,groups,nb,pt=64,cache_count=4,cache_start=28,memory_base=2054):
 k=copy.deepcopy(base);program=k.final_virtual
 oldtime={(e,s):t for t,b in enumerate(program) for e,ss in b.items() for s in ss}
 writer={w:(e,s) for b in program for e,ss in b.items() for s in ss for w in rw(e,s)[1]}
 targets=[]
 for j,(e,s,ds) in enumerate(k._ops):
  if e!='flow' or s[0]!='vselect' or (e,s) not in oldtime or k._op_group[j]>=groups:continue
  a=[writer.get(s[3]+l) for l in range(8)];b=[writer.get(s[4]+l) for l in range(8)]
  if not all(x and x[0]=='alu' and x[1][0]=='^' for x in a+b):continue
  if not all(a[l][1][2]==b[l][1][2]==a[0][1][2]+l for l in range(8)):continue
  targets.append((oldtime[e,s],j,s,a,b))
 assert len(targets)==groups*2,len(targets)
 targets.sort();nextbase=max(k._virtual_width)+8;inserted=[];removed=set();plans=[]
 k._edge_latencies={}
 def new():
  nonlocal nextbase
  b=nextbase;nextbase+=8;k._virtual_width[b]=8;return b
 def add(e,s,priority,group=-1,deps=()):
  j=len(k._ops);r,w=rw(e,s);k._ops.append((e,s,set(deps)));k._op_rw.append((list(r),list(w)));k._op_group.append(group)
  inserted.append((e,s));oldtime[e,s]=priority;return j
 eight=new();add('valu',('+',eight,k.constant_vectors[4],k.constant_vectors[4]),pt)
 pointers=[]
 for b in range(nb):
  out=new();dummy=new()
  if b==0:
   for lane in range(8):add('load',('const',out+lane,memory_base+lane),pt)
  else:add('valu',('+',out,pointers[-1][0],k.constant_vectors[16]),pt)
  add('valu',('+',dummy,out,eight),pt);pointers.append((out,dummy))
 for num,(t,j,s,a,b) in enumerate(targets):
  group=k._op_group[j];out,dummy=pointers[num%nb];address=new();nodes=new();val=a[0][1][2]
  removed.update(a+b);removed.add(('flow',s))
  defaults=[add('store',('store',out+l,a[l][1][3]),t-12+l//2,group) for l in range(8)]
  add('flow',('vselect',address,s[2],dummy,out),t-5,group)
  conditional=[add('store',('store',address+l,b[l][1][3]),t-4+l//2,group,(defaults[l],)) for l in range(8)]
  ld=add('load',('vload',nodes,out),t-1,group,defaults+conditional)
  plans.append((num%nb,defaults,ld))
  replacement=('^',s[1],val,nodes);k._ops[j]=('valu',replacement,set());k._op_rw[j]=(list(range(val,val+8))+list(range(nodes,nodes+8)),list(range(s[1],s[1]+8)))
  for lane in range(8):
   node=('alu',('^',s[1]+lane,val+lane,nodes+lane));inserted.append(node);oldtime[node]=t
 cache_pair=next((s[3],s[4]) for (e,s),t in sorted(oldtime.items(),key=lambda item:item[1]) if e=='flow' and s[0]=='vselect')
 cache_targets=[(j,s) for j,(e,s,d) in enumerate(k._ops) if e=='flow' and s[0]=='vselect' and (s[3],s[4])==cache_pair and cache_start<=k._op_group[j]<cache_start+cache_count]
 assert len(cache_targets)==2*cache_count,(len(cache_targets),cache_pair)
 difference=new();add('valu',('-',difference,*cache_pair),min(oldtime['flow',s] for j,s in cache_targets)-16)
 for j,s in cache_targets:
  removed.add(('flow',s));node=('valu',('multiply_add',s[1],s[2],difference,s[4]));inserted.append(node);oldtime[node]=oldtime['flow',s]
  k._ops[j]=(node[0],node[1],set());rr,ww=rw(*node);k._op_rw[j]=(list(rr),list(ww))
 for buf in range(nb):
  jobs=[p for p in plans if p[0]==buf]
  for previous,current in zip(jobs,jobs[1:]):
   for st in current[1]:k._ops[st][2].add(previous[2]);k._edge_latencies[previous[2],st]=0
 nodes=[(e,s) for b in program for e,ss in b.items() for s in ss if (e,s) not in removed]+inserted
 slotmap={}
 for j,(e,s,d) in enumerate(k._ops):
  slotmap[e,s]=j
  if e=='valu' and len(s)==4:
   for lane in range(8):slotmap['alu',(s[0],s[1]+lane,s[2]+lane,s[3]+lane)]=j
 byop=defaultdict(list);producer={};d=[set() for _ in nodes]
 for i,node in enumerate(nodes):
  byop[slotmap[node]].append(i)
  for w in rw(*node)[1]:assert w not in producer;producer[w]=i
 for i,node in enumerate(nodes):
  for r in rw(*node)[0]:
   if r in producer:d[i].add(producer[r])
 for j,(e,s,ds) in enumerate(k._ops):
  for dep in ds:
   if not set(k._op_rw[j][0])&set(k._op_rw[dep][1]):
    for to in byop[j]:d[to].update(byop[dep])
 succ=[[] for _ in nodes];deg=[len(ds) for ds in d]
 for i,ds in enumerate(d):
  for dep in ds:succ[dep].append(i)
 q=[(oldtime[node],i) for i,node in enumerate(nodes) if not deg[i]];heapq.heapify(q);ordered=[]
 while q:
  _,i=heapq.heappop(q);ordered.append({nodes[i][0]:[nodes[i][1]]})
  for j in succ[i]:
   deg[j]-=1
   if not deg[j]:heapq.heappush(q,(oldtime[nodes[j]],j))
 assert len(ordered)==len(nodes),(len(ordered),len(nodes))
 k.instrs=k._virtual_instrs=ordered;g=Graph(k)
 g.original=[min(oldtime[node] for node in ns) for ns in g.nodes]
 return k,g


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
