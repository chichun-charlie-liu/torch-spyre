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
    NodeUser,
)
from torch._inductor.virtualized import V
from torch import ops
from .stickify import propagate_spyre_tensor_layouts

OP_OUTPUT_GOOD_FOR_LX_REUSE = [
    "max",
    "sum",
    "clone",
]


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

    def get_available_total(self):
        total_avail = self.limit
        for rec in self.usage.values():
            total_avail -= rec["size"]
        return total_avail

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
        NOTE: 1. if an op, e.g. max, occurs multiple times on graph, output buffers will
                 have different names -> end-of-life analysis will take care of dealloc
              2. prev Op's sdsc.out.out.out.json may have useful info, not needed yet
              3. may be able to generalize this decision in buf end-of-life analysis
              4. greedy alloc may cause fragments, can further improve
        """
        graph_output_buf_name = V.graph.get_output_names()
        for tensor_name, needed in mem_usage.items():
            if tensor_name in graph_output_buf_name:
                continue  # graph output has to go back to HBM

            # Decide whether to reuse.
            addr = -1
            tensor_on_lx = self.usage.get(tensor_name, {})
            size_match = tensor_on_lx.get("size", 0) == needed["size"]
            allowed_output_op = any(op in org_op_name for op in OP_OUTPUT_GOOD_FOR_LX_REUSE)

            if needed["is_input"] and tensor_on_lx and size_match:
                addr = self.usage[tensor_name]["addr"]
            elif not needed["is_input"] and allowed_output_op:
                addr = self.find_free_block(needed["size"])

            # add lx info into V.graph.buffers.layout for later codegen use.
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

    # 2. if alloc successful, lx info will be added to corresponding FixedTiledLayout,
    # which will be used in generate_sdsc() later.
    org_op_name = n.node.origin_node.target._opname
    alloc.try_allocate(mem_usage, idx, org_op_name)


def buf_end_of_life_analysis(nodes: list[BaseSchedulerNode]):
    """
    First, find out the last time each buffer was used. {buf1: idx_last_used, ...}
    Turn it into {idx_last_used+1:[buf1, ], ...}, ie. buffers to be deleted at given idx
    """
    last_used: dict = {}
    occurence: dict = {}
    for idx, n in enumerate(nodes):
        for buf in n.used_buffer_names():  # just buf names
            last_used[buf] = idx
            occurence[buf] = occurence.get(buf, 0) + 1

    bufs_to_dealloc_at_idx: dict = {}
    for buf, idx in last_used.items():
        # if last used at idx => del at idx+1
        if idx + 1 in bufs_to_dealloc_at_idx:
            bufs_to_dealloc_at_idx[idx + 1].append(buf)
        else:
            bufs_to_dealloc_at_idx[idx + 1] = [buf]

    return bufs_to_dealloc_at_idx


def try_clone_input_to_lx(
    nodes: list[BaseSchedulerNode],
    lx_free_total: int,
) -> list[BaseSchedulerNode]:
    """
    Check if any input tensors can fit onto scratchpad and needed more than once =>
    add corresponding "clone" node.
    NOTE check Scheduler._replace_node() and fuse_nodes_once() for important items that
        need to be updated.
    """

    graph_lowering = V.graph
    scheduler = V.graph.scheduler
    fx_graph = V.graph.graph
    fx_non_arg_nodes = []
    fx_arg_nodes = {}
    for n in fx_graph.nodes:
        if n.op == "placeholder":
            fx_arg_nodes[n.name] = n
        else:
            fx_non_arg_nodes.append(n)

    for inp_name in V.graph.graph_input_names:

        buf = V.graph.get_buffer(inp_name)
        dev_layout = buf.layout.device_layout
        dev_size = math.prod(dev_layout.device_size[:-1]) * 128

        # Step 0: check how many times this buffer will be read, decide cloning or not
        nodes_to_be_updated = []
        num_read = 0
        for n in nodes:
            if inp_name in [r.name for r in n.read_writes.reads]:
                num_read += 1
                nodes_to_be_updated.append(n)

        if num_read == 1 or dev_size > lx_free_total:
            continue

        # step 1: create a new FX node on FX graph and then refresh dependencies
        fx_inp = fx_arg_nodes[inp_name]
        old_users = list(fx_inp.users.keys())    # get old users before insertion
        fx_graph.inserting_before(fx_non_arg_nodes[0])
        new_fx_node = fx_graph.create_node(
            "call_function", ops.aten.clone.default, (fx_inp,)
        )
        # update user nodes' .args attr
        for user in old_users:
            user.args = tuple(new_fx_node if ar is fx_inp else ar for ar in user.args)
        V.graph.orig_gm.recompile()

        # step 2: Use the new FX node -> new TensorBox -> new SchedulerNode
        # NOTE .run_node(n) needs a {fx nodes: TensorBox} mapping for each elem in n.args
        # e.g. new_fx_node.args=(fx_inp,), i.e. arg0_1 -> point to arg0_1's TensorBox
        for tb in graph_lowering.name_to_users[inp_name]:
            tb_fx_node = list(tb.data.origins)[0]
            graph_lowering.env[tb_fx_node] = tb
            # all TBs related to inp_name are added, except the new TB below
        # graph_lowering.args_iter = graph_lowering.example_inputs  # was needed for something?
        new_tb = graph_lowering.run_node(new_fx_node)
        com_buf = new_tb.data.data
        new_sch_node = scheduler.create_scheduler_node(com_buf)
        propagate_spyre_tensor_layouts([new_sch_node])
        new_buf_name = com_buf.name
        graph_lowering.env[new_fx_node] = new_tb

        # Update graph_lowering.name_to_users[inp_name] (arg0_1). Except the InpBuf and
        # the new_buf, the rest should become users of new_buf
        users_of_inp, users_of_new_buf = [], []
        for tb in graph_lowering.name_to_users[inp_name]:
            if tb.data.data.name in [inp_name, new_buf_name]:
                users_of_inp.append(tb)
            else:
                users_of_new_buf.append(tb)
        graph_lowering.name_to_users[inp_name] = users_of_inp
        graph_lowering.name_to_users[new_buf_name] = users_of_new_buf

        # 3 Update dependencies that accessed arg0_1 -> new_buf
        for n in nodes_to_be_updated:  # n is a SchedulerNode
            # 3-1 update LoopIR.inner_fn under CompBuf.
            # NOTE cannot be updated directly => create new CompBuf to get new LoopIR
            n_fx = n.node.data.origin_node  # args of this fx node are up-to-date
            old_com_buf = n.node
            new_tb_n = graph_lowering.run_node(n_fx)
            new_com_buf_n = new_tb_n.data.data            
            old_com_buf.data = new_com_buf_n.data
            # must clear cached body or it will not refresh _body and n.read_writes
            old_com_buf.get_default_sizes_body.clear_cache(old_com_buf)
            n.recompute_size_and_body()

            # clean up tables to remove the unwanted items due to new TB created above
            del graph_lowering.name_to_op[new_com_buf_n.operation_name]
            del graph_lowering.name_to_buffer[new_com_buf_n.name]
            graph_lowering.name_to_users[new_buf_name].pop(-1)

            new_sch_node.outputs[0].users.append(NodeUser(n, False, False))

        # other items to update
        new_sch_node.min_order = nodes[0].min_order - 1
        new_sch_node.max_order = nodes[0].max_order - 1
        nodes = [new_sch_node] + nodes
        scheduler.nodes = nodes
        # scheduler.nodes = scheduler.topological_sort_schedule(scheduler.nodes)
        # scheduler.prune_redundant_deps(scheduler.nodes)
        scheduler.name_to_node = {n.get_name():n for n in scheduler.nodes}
        scheduler.name_to_fused_node = scheduler.name_to_node
        scheduler.name_to_buf.update(new_sch_node.outputs_by_name)
        lx_free_total -= dev_size

    return nodes


def scratchpad_planning(
    nodes: list[BaseSchedulerNode],
) -> list[BaseSchedulerNode]:
    # Nodes are in topological order (guarenteed by caller).
    # Work division has already been done.
    # Stickification has already been done (therefore all ComputedBeffers have FixedTiledLayouts)

    alloc = ScratchPadAllocator()

    nodes = try_clone_input_to_lx(nodes, alloc.get_available_total())
    node_idx_to_dealloc_bufs = buf_end_of_life_analysis(nodes)

    for idx, n in enumerate(nodes):
        # release unneeded LX allocations before actual planning
        alloc.deallocate(node_idx_to_dealloc_bufs.get(idx, []))

        if isinstance(n, SchedulerNode) and isinstance(n.node, ComputedBuffer):
            consider_for_scratchpad(n, alloc, idx)
    # print(alloc.lx_usage_hist)
    return nodes
