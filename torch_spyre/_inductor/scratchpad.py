# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
from torch._inductor.ir import (
    ComputedBuffer,
)
from torch._inductor.scheduler import (
    BaseSchedulerNode,
    SchedulerNode,
)
from torch._inductor.virtualized import V

from torch_spyre._inductor.ir import FixedTiledLayout
from torch_spyre._C import SpyreTensorLayout


class ScratchPadAllocator:
    """
    A trivial bump pointer allocator
    """

    def __init__(self, size: int = None):
        # scratch pad is 2MB = 2<<20 bytes in total. preserve total * DXP_LX_FRAC_AVAIL
        # for backend usage unless specified otherwise
        if size is None:
            size = int((2<<20) * (1.0 - os.environ.get("DXP_LX_FRAC_AVAIL", 0.2)))
        self.current = 0
        self.limit = size
        self.usage = {}  # each record will be tensor_name:{"addr": yy, "size": zz}

    def get_lowest_addr_in_use(self):
        if len(self.usage) > 0:
            return min([rec["addr"] for rec in self.usage.values()])
        return None
    
    def get_highest_addr_in_use(self):
        if len(self.usage) > 0:
            return max([rec["addr"]+rec["size"] for rec in self.usage.values()])
        return None
    
    def find_free_block(self, size_needed: int):
        curr_lo = self.get_lowest_addr_in_use()
        curr_hi = self.get_highest_addr_in_use()
        if len(self.usage) == 0 or curr_lo >= size_needed:
            # completely free or enough room at addr0 
            return 0
        elif curr_hi + size_needed < self.limit:
            # enough room at higher addr, return next 128-multiple
            return math.ceil(curr_hi/128)*128
        elif len(self.usage) > 1:
            # find a "hole" between lowest and highest (assume a block was dealloc'ed)
            rec_only = list(self.usage.values())  # simply drop tensor names, not needed
            sorted_rec = sorted(rec_only, key=lambda rec: rec['addr'])
            for i in range(len(sorted_rec)-1):
                frag_st = sorted_rec[i]["addr"] + sorted_rec[i]["size"]
                frag_end = sorted_rec[i+1]["addr"]
                if frag_end - frag_st > size_needed:
                    return frag_st
            return -1
        else:
            # cannot find any free blocks
            return -1

    def try_allocate(self, mem_usage: dict) -> int:
        """
        Allocate based on needed mem_usage of the node and keep a record in self.usage. 
        NOTE: 1. assume compiler always allocates inputs before output. (but need to
                 implement special cases when tensor is too large later.)
              2. Some unresolved issues still prevent the reuse of main input tensors.
        TODO: may need to utilize info from previous Op's sdsc.out.out.out.json  
        """
        lx_alloc_to_del = []
        for tensor_name, needed in mem_usage.items():
            lx_rec = self.usage.get(tensor_name, {})

            if lx_rec and lx_rec["size"] == needed["size"]:
                # same tensor name and size is on scratchpad already, reuse it
                addr = lx_rec["addr"]
            else:
                # new allocation or overwrite the existing one
                addr = self.find_free_block(needed["size"])

            if addr == -1:
                # no further action if allocation failed
                continue

            # Assume all tensors (that fit) are moved to scratchpad first, dealloc later if needed
            self.usage[tensor_name] = {"addr": addr, "size": needed["size"]}

            # directly add the lx info into V.graph.buffers if it meets some criteria
            # TODO For the reason in doctstring Note 2, filter out *non-1D tensors*
            buf = V.graph.get_buffer(tensor_name)
            layout = buf.get_layout()
            dims = len(layout.size)
            dims_eq_1 = layout.size.count(1)
            if (dims - dims_eq_1) == 1:  # effectively 1D tensor
                layout.allocation["lx"] = addr
            else:
                lx_alloc_to_del.append(tensor_name)

        for t in lx_alloc_to_del:
            # TODO also need to consider inductor information about buffer re-use
            # HOWEVER, reuse info is only available at codegen stage, see AllocateLine.plan()
            del self.usage[t]


    # TODO add dealloc and defrag mechanism to allocator later


def mem_usage_by_node(n: SchedulerNode):
    """
    TODO 1. we assume there is always only 1 output buffer per node. double check
    TODO 2. node.get_read_write_buffer_accesses() may not be accurate, e.g. paddings due
            to stickification on device not included => better to find input device_layout
    """
    mem_usage = {}
    inp_bytes = n.get_read_write_buffer_accesses(include_reads=True, include_writes=False)
    for inp in n.read_writes.reads:
        mem_usage[inp.name] = {"is_input": True, "size": inp_bytes[inp.name],}

    # process output, assuming 1 per node
    buf = n.node
    layout: FixedTiledLayout = buf.layout
    dev_layout = layout.device_layout
    out_name = buf.name
    out_num_sticks = math.prod(dev_layout.device_size[:-1])
    out_size = out_num_sticks * 128  # 1 stick = 128 B
    mem_usage[out_name] = {"is_input": False, "size": out_size,}

    return mem_usage


def consider_for_scratchpad(
    n: SchedulerNode, alloc: ScratchPadAllocator
):
    # 1. summarize both inputs and output sizes used by this node.
    mem_usage_n = mem_usage_by_node(n)

    # 2. adding lx info to mem_usage_n summary, -1 means failed to alloc
    alloc.try_allocate(mem_usage_n)

    # all lx info (including inputs and output) have been stored in corresponding
    # FixedTiledLayout at this point. will use the info in generate_sdsc() later.


def scratchpad_planning(
    nodes: list[BaseSchedulerNode],
) -> list[BaseSchedulerNode]:
    # Nodes are in topological order (guarenteed by caller).
    # Work division has already been done.
    # Stickification has already been done (therefore all ComputedBeffers have FixedTiledLayouts)

    alloc = ScratchPadAllocator()

    # find out the last time each buffer was used
    last_used = {}
    for idx, n in enumerate(nodes):
        for buf in n.used_buffer_names():  # just buf names
            last_used[buf] = idx

    bufs_to_dealloc_at_idx = {}
    for buf, idx in last_used.items():
        # if last used at idx => del at idx+1
        if idx + 1 in bufs_to_dealloc_at_idx:
            bufs_to_dealloc_at_idx[idx+1].append(buf)
        else:
            bufs_to_dealloc_at_idx[idx+1] = [buf]

    for idx, n in enumerate(nodes):
        # release unneeded allocations from lx before actual planning
        for buf in bufs_to_dealloc_at_idx.get(idx, []):
            if buf in alloc.usage:
                del alloc.usage[buf]

        if isinstance(n, SchedulerNode) and isinstance(n.node, ComputedBuffer):
            consider_for_scratchpad(n, alloc)

    return nodes