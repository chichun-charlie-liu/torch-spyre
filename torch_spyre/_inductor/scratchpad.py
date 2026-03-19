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
from torch._inductor.ops_handler import WrapperHandler
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


class NameSwapHandler(WrapperHandler):
    def __init__(self, inner, name_map: dict[str, str]):
        super().__init__(inner)
        self._name_map = name_map

    def load(self, name, index):
        return super().load(self._name_map.get(name, name), index)


def create_Loop_hack_inner_fn(old_Loop, name_map):
    """ Use ops_handler to swap the name of buffers"""
    def new_inner_fn(*args):
        # Pointwise has 1 pos arg index while Reduction has 2, i.e. (index, rindex)
        with V.set_ops_handler(NameSwapHandler(V.ops, name_map)):
            return old_Loop.inner_fn(*args)

    # old_Loop could be a Pointwise or Reduction.
    kwargs = {k: getattr(old_Loop, k) for k in old_Loop.__dataclass_fields__.keys()}
    kwargs["inner_fn"] = new_inner_fn
    new_Loop = old_Loop.__class__(**kwargs) 
    # Additional attr that are not included in dataclass_fields. NOTE it relies on a
    # special method to force reset attrs of a frozen dataclass, see ir.Loops.create()
    new_Loop._post_init_setattr("origins", old_Loop.origins)
    new_Loop._post_init_setattr("origin_node", old_Loop.origin_node)
    new_Loop._post_init_setattr("traceback", old_Loop.traceback)
    # .get_stack_traces() get info from "origins", no need to manually set anything
    # LoopBody will be created later when we call CompBuf.recompute()

    return new_Loop


def try_clone_input_to_lx(
    nodes: list[BaseSchedulerNode],
    lx_free_total: int,
) -> list[BaseSchedulerNode]:
    """
    Check if any input tensors can fit onto scratchpad and needed more than once =>
    add corresponding "clone" node so that we can reuse it from scratchpad.

    During the lowering process, FX nodes are interpreted into SchedulerNodes, but info
    added on SchedulerNodes may not be entirely back-propogated to FX graph, e.g. a
    SchedulerNode could be created without a corresponding FX node. It would be safer
    not to directly rely on FX graph when possible. For example:
    1. if we need to know the users of a schedulerNode, better check node.read_writes
        instead of origin_node.args.
    2. if we need a new LoopIR for a given schedulerNode, we could make changes to its
        corresponding FX node then utilize GraphLowering.run_node(updated_fx_node) ->
        new CompBuf -> new LoopIR. But in case corresponding FX node has defect to begin
        with, we chose to hack the inner_fn then build a new LoopIR from there.
    NOTE:
    - ONCE WE correctly updated args in "node.data(i.e. a LoopIR).inner_fn", 
      node.read_writes and node._body can be refreshed by calling node.recompute_body().
      But need to make sure to clear cached body first.
    - check Scheduler._replace_node() and fuse_nodes_once() for hints of important items
      that need to be updated.
    """

    graph_lowering = V.graph
    scheduler = V.graph.scheduler
    fx_graph = V.graph.graph

    buf_read_counts = {}
    buf_users = {}
    for n in nodes:
        reads = n.get_read_write_buffer_accesses(
            include_reads=True, include_writes=False
        )
        for b in reads.keys():
            buf_read_counts[b] = buf_read_counts.get(b, 0) + 1
            buf_users[b] = buf_users.get(b, [])
            buf_users[b].append(n)  # TODO a node cannot read the same buf twice? no need to dedup?

    for inp_name in V.graph.graph_input_names:

        # Step 0: check how many times this buffer will be read, decide cloning or not
        buf = V.graph.get_buffer(inp_name)
        dev_layout = buf.layout.device_layout
        dev_size = math.prod(dev_layout.device_size[:-1]) * 128
        is_on_lx = buf.layout.allocation != {}
        if buf_read_counts[inp_name] == 1 or dev_size > lx_free_total or is_on_lx:
            continue

        # step 1: create a new FX node on FX graph and then refresh dependencies
        fx_inp = list(buf.origins)[0]
        old_users = list(fx_inp.users.keys())    # get old users before insertion
        fx_graph.inserting_after(fx_inp)
        new_fx_node = fx_graph.create_node(
            "call_function", ops.aten.clone.default, (fx_inp,)
        )
        # update user's input (nodes.args is a tuple of fx nodes)
        for user in old_users:
            user.args = tuple(new_fx_node if ar is fx_inp else ar for ar in user.args)
        V.graph.orig_gm.recompile()

        # step 2: Use the new FX node -> new TensorBox -> new SchedulerNode
        # NOTE .run_node(n) needs a {fx nodes: TensorBox} mapping for each elem in n.args
        # e.g. new_fx_node.args=(arg0, ), env[arg0_1] -> point to arg0_1's TensorBox
        env = {}
        for tbs in graph_lowering.name_to_users.values():
            for tb in tbs:
                tb_fx_node = list(tb.data.origins)[0]
                if tb_fx_node in env and env[tb_fx_node] is not tb:
                    raise ValueError("A TensorBox has more than 1 associated FX node.")
                env[tb_fx_node] = tb
        graph_lowering.env.update(env)
        # graph_lowering.args_iter = graph_lowering.example_inputs  # doesn't seem needed anymore?
        new_tb = graph_lowering.run_node(new_fx_node)
        com_buf = new_tb.data.data
        new_sch_node = scheduler.create_scheduler_node(com_buf)
        propagate_spyre_tensor_layouts([new_sch_node])
        new_buf_name = com_buf.name
        graph_lowering.env[new_fx_node] = new_tb

        # Update graph_lowering.name_to_users[inp_name] (a list of TensorBox). Existing
        # users of arg0, other than InpBuf and new_buf, should become users of new_buf.
        users_of_inp, users_of_new_buf = [], []
        for tb in graph_lowering.name_to_users[inp_name]:
            if tb.data.data.name in [inp_name, new_buf_name]:
                users_of_inp.append(tb)
            else:
                users_of_new_buf.append(tb)
        graph_lowering.name_to_users[inp_name] = users_of_inp
        graph_lowering.name_to_users[new_buf_name] = users_of_new_buf

        # Step 3: Update user nodes of arg0_1 -> change it to new_buf
        for n_user in buf_users[inp_name]:
            old_com_buf = n_user.node
            # hack inner_fn with a nameSwapper ops handler and make a new LoopIR
            new_Loop = create_Loop_hack_inner_fn(
                old_com_buf.data, name_map={inp_name: new_buf_name}
            )
            old_com_buf.data = new_Loop
            # must clear cached body or recompute() will not do anything
            old_com_buf.get_default_sizes_body.clear_cache(old_com_buf)
            n_user.recompute_size_and_body()

            new_sch_node.outputs[0].users.append(NodeUser(n_user, False, False))

        # other items to update
        first_user = buf_users[inp_name][0]
        new_sch_node.min_order = first_user.min_order - 0.5
        new_sch_node.max_order = first_user.max_order - 0.5
        idx_to_first_user = nodes.index(first_user)
        nodes.insert(idx_to_first_user, new_sch_node)
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
