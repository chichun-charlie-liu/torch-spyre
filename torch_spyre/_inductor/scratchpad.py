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
from torch import Size

OP_OUTPUT_GOOD_FOR_LX_REUSE = [
    "max",
    "sum",
]  #  "exp"


class ScratchPadAllocator:
    """LX manager simplified version"""

    def __init__(self, size: int = -1):
        # scratch pad is 2MB = 2<<20 bytes in total. preserve total * DXP_LX_FRAC_AVAIL
        # for backend usage unless specified otherwise
        if size == -1:
            size = int(
                (2 << 20) * (1.0 - float(os.environ.get("DXP_LX_FRAC_AVAIL", "0.2")))
            )
        self.limit = size
        self.usage: dict = {}  # each record will be tensor_name:{"addr": yy, "size": zz}
        self.lx_usage_hist: list = []

    def get_lowest_addr_in_use(self):
        if len(self.usage) > 0:
            return min([rec["addr"] for rec in self.usage.values()])
        return None

    def get_highest_addr_in_use(self):
        if len(self.usage) > 0:
            return max([rec["addr"] + rec["size"] for rec in self.usage.values()])
        return None

    def find_free_block(self, size_needed: int):
        # cannot perform defragmentation yet, will add more cases in the future
        curr_lo = self.get_lowest_addr_in_use()
        curr_hi = self.get_highest_addr_in_use()
        if len(self.usage) == 0 or curr_lo >= size_needed:
            # completely free or enough room at addr0
            return 0
        elif curr_hi + size_needed < self.limit:
            # enough room at higher addr, return next 128-multiple
            return math.ceil(curr_hi / 128) * 128
        elif len(self.usage) > 1:
            # find a "hole" between lowest and highest (assume a block was dealloc'ed)
            rec_only = list(self.usage.values())  # simply drop tensor names, not needed
            sorted_rec = sorted(rec_only, key=lambda rec: rec["addr"])
            for i in range(len(sorted_rec) - 1):
                frag_st = sorted_rec[i]["addr"] + sorted_rec[i]["size"]
                frag_end = sorted_rec[i + 1]["addr"]
                if frag_end - frag_st >= size_needed:
                    return frag_st
            return -1
        else:
            # cannot find any free blocks
            return -1

    def try_allocate(self, mem_usage: dict, idx: int, org_op_name: str):
        """
        Simple reuse rule:
        1. for an "input" tensor, found a matched tensor (name and size) on LX
        2. for an output tensor, if this op is on the "white list" => prep for pinning
            => alloc a new LX block for the "output" of the op
        If can_reuse => add lx info to corresponding buffer.layout
        NOTE: 1. deeptool LX partition issue was fixed on 260220 (commit #?).
              2. if an op, e.g. max, occurs multiple times on graph, output buffers will
                 have different names -> end-of-life analysis will take care of dealloc
        TODO: 1. prev Op's sdsc.out.out.out.json may have useful info, not needed yet
              2. may be able to generalize this decision in buf end-of-life analysis
        """
        graph_output_buf_name = V.graph.get_output_names()
        for tensor_name, needed in mem_usage.items():
            if tensor_name == graph_output_buf_name:
                continue  # graph output has to go back to HBM

            # Decide whether to reuse.
            addr = -1
            if needed["is_input"]:
                if (
                    tensor_name in self.usage
                    and self.usage[tensor_name]["size"] == needed["size"]
                ):
                    addr = self.usage[tensor_name]["addr"]
            else:
                if any(
                    op in org_op_name for op in OP_OUTPUT_GOOD_FOR_LX_REUSE
                ) or needed.get("clone_to_lx", False):
                    addr = self.find_free_block(needed["size"])

            # can reuse (i.e., input found on lx or output alloc succeeded), add lx info
            # into V.graph.buffers.layout for later codegen use.
            if addr != -1:
                self.usage[tensor_name] = {"addr": addr, "size": needed["size"]}

                buf = V.graph.get_buffer(tensor_name)
                layout = buf.get_layout()
                layout.allocation[f"lx:{idx}"] = addr  # node idx is for debugging
                # Record usage history for debugging
                self.lx_usage_hist.append(
                    {
                        "node_idx": idx,
                        "op_name": org_op_name,
                        "tensor_name": tensor_name,
                        "addr": addr,
                        "size": needed["size"],
                    }
                )

    def deallocate(self, bufs: list[str]):
        """Try to deallocate each of the buffers in a list, if exists."""
        if isinstance(bufs, str):
            bufs = [bufs]

        for buf in bufs:
            if buf in self.usage:
                del self.usage[buf]

    # TODO add dealloc and defrag mechanism to allocator later


def mem_usage_by_node(n: SchedulerNode):
    """Get a summary of memory usage of the input node"""
    mem_usage = {}
    for r_or_w, buf_memDeps in enumerate([n.read_writes.reads, n.read_writes.writes]):
        for buf_memDep in buf_memDeps:
            buf = V.graph.get_buffer(buf_memDep.name)
            dev_layout = buf.layout.device_layout  # this is device layout
            dev_size = (
                math.prod(dev_layout.device_size[:-1]) * 128
            )  # num_sticks * bytes_per_stick
            mem_usage[buf_memDep.name] = {
                "is_input": r_or_w == 0,
                "size": dev_size,
            }

    return mem_usage


def consider_for_scratchpad(
    n: SchedulerNode,
    alloc: ScratchPadAllocator,
    idx: int,
):
    # 1. summarize both inputs and output sizes used by this node.
    mem_usage = mem_usage_by_node(n)

    org_node = n.node.origin_node
    # [hack] proof of concept, clone tensor from hbm to lx for reuse
    if "mul" in org_node.name and org_node.args[1].meta["val"].shape == Size([1]):
        mem_usage[n.node.name]["clone_to_lx"] = True

    # 2. if alloc successful, lx info will be added to corresponding FixedTiledLayout,
    # which will be used in generate_sdsc() later.
    alloc.try_allocate(mem_usage, idx, org_node.name)


def buf_end_of_life_analysis(nodes: list[BaseSchedulerNode]):
    """
    First, find out the last time each buffer was used. {buf1: idx_last_used, ...}
    Turn it into {idx_last_used+1:[buf1, ], ...}, ie. buffers to be deleted at given idx
    """
    last_used: dict = {}
    for idx, n in enumerate(nodes):
        for buf in n.used_buffer_names():  # just buf names
            last_used[buf] = idx

    bufs_to_dealloc_at_idx: dict = {}
    for buf, idx in last_used.items():
        # if last used at idx => del at idx+1
        if idx + 1 in bufs_to_dealloc_at_idx:
            bufs_to_dealloc_at_idx[idx + 1].append(buf)
        else:
            bufs_to_dealloc_at_idx[idx + 1] = [buf]

    return bufs_to_dealloc_at_idx


def scratchpad_planning(
    nodes: list[BaseSchedulerNode],
) -> list[BaseSchedulerNode]:
    # Nodes are in topological order (guarenteed by caller).
    # Work division has already been done.
    # Stickification has already been done (therefore all ComputedBeffers have FixedTiledLayouts)

    alloc = ScratchPadAllocator()

    node_idx_to_dealloc_bufs = buf_end_of_life_analysis(nodes)

    # fx_graph = V.graph.graph
    # fx_nodes = list(fx_graph.nodes)
    # fx_graph.inserting_before(fx_nodes[1])  # node[0] is placeholder for inputs
    # new_fx_node = fx_graph.create_node("call_function", lambda x: x, fx_nodes[1].args )  # identity
    # fx_nodes[1].args = (new_fx_node,)
    # V.graph.orig_gm.recompile()
    # # at this point, fx graph is updated but GraphLowering has not.

    # tmp = ExternKernel.copy_input(nodes[0].node)  # create a PW TensorBox from a ComputedBuffer
    # com_buf = tmp.data.data
    # com_buf.origin_node = new_fx_node
    # com_buf.origins.add(new_fx_node)
    # curr_sch = nodes[0].scheduler
    # sch_node = SchedulerNode(curr_sch, com_buf)
    # propagate_spyre_tensor_layouts([sch_node])
    # # min_order will be used for nodes sorting later
    # sch_node.min_order = nodes[0].min_order - 1
    # nodes = [sch_node] + nodes
    # # update name_to_buf mapping
    # curr_sch.name_to_buf = {
    #     buf.get_name(): buf for node in nodes for buf in node.get_outputs()
    # }

    for idx, n in enumerate(nodes):
        # release unneeded LX allocations before actual planning
        alloc.deallocate(node_idx_to_dealloc_bufs.get(idx, []))

        if isinstance(n, SchedulerNode) and isinstance(n.node, ComputedBuffer):
            consider_for_scratchpad(n, alloc, idx)
    # print(alloc.lx_usage_hist)
    return nodes
