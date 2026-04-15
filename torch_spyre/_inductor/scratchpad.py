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

from torch._inductor.ir import (
    ComputedBuffer,
    MutationLayoutSHOULDREMOVE,
    Operation,
)
from torch._inductor.lowering import clone as clone_lowering
from torch._inductor.ops_handler import WrapperHandler
from torch._inductor.scheduler import (
    BaseSchedulerNode,
    SchedulerNode,
    NodeUser,
)
from torch._inductor.virtualized import V
from torch import ops
from .logging_utils import get_inductor_logger
from .ir import FixedTiledLayout
from . import config

OP_OUTPUT_GOOD_FOR_LX_REUSE = [
    "max",
    "sum",
    "clone",
]

logger = get_inductor_logger("LX_PLANNING")


class ScratchPadAllocator:
    """LX manager simplified version"""

    def __init__(self, size: int = -1):
        # scratch pad is 2MB = 2<<20 bytes in total. preserve total * DXP_LX_FRAC_AVAIL
        # for backend usage unless specified otherwise
        if size == -1:
            size = int((2 << 20) * (1.0 - config.dxp_lx_frac_avail))
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
            is_graph_input = tensor_name not in V.graph.name_to_buffer
            is_graph_output = tensor_name in graph_output_buf_name
            core_div_mismatch = (not needed["is_input"]) and needed["core_div_mismatch"]
            if is_graph_input or is_graph_output or core_div_mismatch:
                # graph input itself cannot be pin, but we may be able to clone
                # graph output has to go back to HBM
                # if buf users have diff core-splits -> cause cross-core LX read/write
                continue

            # Decide whether to reuse.
            addr = -1
            tensor_on_lx = self.usage.get(tensor_name, {})
            size_match = tensor_on_lx.get("size", 0) == needed["size"]
            allowed_output_op = any(
                op in org_op_name for op in OP_OUTPUT_GOOD_FOR_LX_REUSE
            )

            if needed["is_input"] and tensor_on_lx and size_match:
                addr = self.usage[tensor_name]["addr"]
            elif not needed["is_input"] and allowed_output_op:
                addr = self.find_free_block(needed["size"])

            # add lx info into V.graph.buffers.layout for later codegen use.
            if addr != -1:
                self.usage[tensor_name] = {"addr": addr, "size": needed["size"]}

                buf = V.graph.get_buffer(tensor_name)
                layout = buf.get_layout()
                layout.allocation["lx"] = addr
                # NOTE assume same addr for same buf, no realloc needed/allowed
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


def mem_usage_by_op(op: ComputedBuffer):
    """Get a summary of memory usage of the input operation"""
    rw = op.get_read_writes()
    mem_usage = {}
    for is_input, deps in [(True, rw.reads), (False, rw.writes)]:
        for dep in deps:
            buf = V.graph.get_buffer(dep.name)
            dev_layout = buf.layout.device_layout  # this is device layout
            dev_size = (
                math.prod(dev_layout.device_size[:-1]) * 128
            )  # num_sticks * bytes_per_stick
            mem_usage[dep.name] = {
                "is_input": is_input,
                "size": dev_size,
            }

    return mem_usage

def consider_for_scratchpad(
    op: ComputedBuffer,
    alloc: ScratchPadAllocator,
    idx: int,
    core_div_mismatch: dict[str, bool] = {},
):
    """
    If core_div_mismatch is not provided, we will consider LX pinning without taking
    core division into account (previous behavior), may result in slices of a LX tensor
    scattered over different core's scratchpad, which may result in unusable tensor and
    incorrect results.
    """
    # 1. summarize both inputs and output sizes used by this node.
    mem_usage = mem_usage_by_op(op)
    for buf in mem_usage:
        mem_usage[buf]["core_div_mismatch"] = core_div_mismatch.get(buf, False)
        # if a buf is not in core_div_mismatch => it has no users => graph output

    # 2. if alloc successful, lx info will be added to corresponding FixedTiledLayout,
    # which will be used in generate_sdsc() later.
    org_op_name = op.origin_node.target._opname
    alloc.try_allocate(mem_usage, idx, org_op_name)


def buf_analysis(operations: list[Operation]):
    """
    First, find out the last time each buffer was used. {buf1: idx_last_used, ...}
    Turn it into {idx_last_used+1:[buf1, ], ...}, ie. buffers to be deleted at given idx
    """
    last_used: dict = {}
    buf_read_counts: dict[str, int] = {}
    buf_write_counts: dict[str, int] = {}
    buf_users: dict[str, SchedulerNode] = {}
    buf_users_read_and_write: dict[str, SchedulerNode] = {}
    core_div_mismatch: dict[str, bool] = {}

    for idx, op in enumerate(operations):
        rw = op.get_read_writes()
        buf_read_by_op = rw.reads
        for buf in op.used_buffer_names():  # just buf names
            last_used[buf] = idx
            if buf in buf_read_by_op:
                buf_read_counts[buf] = buf_read_counts.get(buf, 0) + 1
                buf_users[buf] = buf_users.get(buf, []) + [op]
            else:
                buf_write_counts[buf] = buf_write_counts.get(buf, 0) + 1
            buf_users_read_and_write[buf] = buf_users_read_and_write.get(buf, []) + [op]

    bufs_to_dealloc_at_idx: dict = {}
    for buf, idx in last_used.items():
        # if last used at idx => del at idx+1
        if idx + 1 in bufs_to_dealloc_at_idx:
            bufs_to_dealloc_at_idx[idx + 1].append(buf)
        else:
            bufs_to_dealloc_at_idx[idx + 1] = [buf]

    # Check core-division -> If the node generating the buffer and any of the nodes
    # consuming this buffer have different core division => do not pin this buffer to LX
    # NOTE Because each core can only write to its own scratchpad. For example, if a
    #       buffer is sliced 8 ways (stored on 8 LX) but next Op is 4-cores -> next op
    #       has to read from 2 different scratchpads...
    # TODO looking for options to broadcast to or all_reduce from multiple scratchpad
    using_multicore = config.sencores > 1
    for buf_name, users_rw in buf_users_read_and_write.items():
        # this dict includes graph input and output
        same_core_div = True
        if using_multicore and len(users_rw) > 1:
            # >1 check is for graph output
            u0_split = users_rw[0].op_it_space_splits
            same_core_div = all(u0_split == u.op_it_space_splits for u in users_rw[1:])
        core_div_mismatch[buf_name] = not same_core_div

    return bufs_to_dealloc_at_idx, buf_users, core_div_mismatch


class NameSwapHandler(WrapperHandler):
    def __init__(self, inner, name_map: dict[str, str]):
        super().__init__(inner)
        self._name_map = name_map

    def load(self, name, index):
        return super().load(self._name_map.get(name, name), index)


def create_Loop_hack_inner_fn(old_Loop, name_map):
    """Use ops_handler to swap the name of buffers"""

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


def try_insert_clone_nodes_for_inputs(
    nodes: list[BaseSchedulerNode],
    lx_free_total: int,
    buf_users: dict[str, SchedulerNode],
    core_div_mismatch: dict[str, bool],
) -> list[BaseSchedulerNode]:
    """
    Check if any input tensors can fit onto scratchpad and needed more than once =>
    Add corresponding "clone" node to copy it to scratchpad and reduce reading from HBM.

    Simplified flow to create a new SchedulerNode, no FX graph involved:
        new Pointwise (wrapped in a TensorBox) -> ComputedBuffer -> SchedulerNode

    NOTE:
    - To update existing users of the old buffer -> hack the inner_fn then refresh LoopIR
    - Once we correctly updated inner_fn (double check args in "node.data" i.e. a LoopIR),
      node.read_writes and node._body can be refreshed by calling node.recompute_body(),
      remember to clear cached body first.
    - If we need to know the users of a schedulerNode, better check node.read_writes
      instead of origin_node.args.
    - check Scheduler._replace_node() and fuse_nodes_once() for hints of important items
      that need to be updated.
    """

    graph_lowering = V.graph
    scheduler = V.graph.scheduler
    fx_graph = V.graph.graph

    for inp_name in V.graph.graph_input_names:
        # Step 0: check how many times this buffer will be read, decide cloning or not
        buf = V.graph.get_buffer(inp_name)
        dev_layout = buf.layout.device_layout
        dev_size = math.prod(dev_layout.device_size[:-1]) * 128
        is_on_lx = buf.layout.allocation != {}
        used_only_once = len(buf_users[inp_name]) == 1
        if (
            used_only_once
            or dev_size > lx_free_total
            or is_on_lx
            or core_div_mismatch[inp_name]
        ):
            continue

        # Step 1: Create a Pointwise IR -> a ComputedBuffer which has the same layout
        #         as input (already FixedTileLayout) -> SchedulerNode
        clone_IR_tb = clone_lowering(buf)  # a TensorBox wrapping a PointwiseIR
        com_buf = ComputedBuffer(
            name=None,
            layout=FixedTiledLayout(
                buf.layout.device,
                buf.layout.dtype,
                buf.layout.size,
                buf.layout.stride,
                buf.layout.device_layout,
            ),
            data=clone_IR_tb.data.data,
        )
        # create a "dangling" FX node, just to store meta data
        fx_inp = list(buf.origins)[0]
        com_buf.origin_node = fx_graph.create_node(
            "call_function", ops.aten.clone.default, (fx_inp,)
        )
        com_buf.name = V.graph.register_buffer(com_buf)
        V.graph.register_operation(com_buf)
        new_sch_node = scheduler.create_scheduler_node(com_buf)
        new_buf_name = com_buf.name

        # Step 2: Update graph_lowering.name_to_users (a list of TensorBox), eg, existing
        # users of arg0, other than InpBuf and new_buf, should become users of new_buf.
        users_of_inp, users_of_new_buf = [], []
        for tb in graph_lowering.name_to_users[inp_name]:
            if tb.data.data.name in [inp_name, new_buf_name]:
                users_of_inp.append(tb)
            else:
                users_of_new_buf.append(tb)
        graph_lowering.name_to_users[inp_name] = users_of_inp
        graph_lowering.name_to_users[new_buf_name] = users_of_new_buf

        # Step 3: Update user nodes's inner_fn, _body, read_writes of old_buf -> new_buf
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
        scheduler.name_to_node = {n.get_name(): n for n in scheduler.nodes}
        scheduler.name_to_fused_node = scheduler.name_to_node
        scheduler.name_to_buf.update(new_sch_node.outputs_by_name)
        lx_free_total -= dev_size

    return nodes


def scratchpad_planning(
    operations: list[Operation],
) -> None:
    # Operations are in topological order (guaranteed by GraphLowering).
    # Core division has already been done.
    # Stickification has already been done (therefore all ComputedBeffers have FixedTiledLayouts)

    alloc = ScratchPadAllocator()

    idx_to_dealloc_bufs, buf_users, core_div_mismatch = buf_analysis(operations)

    num_ops_before = len(operations)
    operations = try_insert_clone_nodes_for_inputs(
        operations,
        alloc.get_available_total(),
        buf_users,
        core_div_mismatch,
    )

    if len(operations) > num_ops_before:
        idx_to_dealloc_bufs, buf_users, core_div_mismatch = buf_analysis(operations)

    for idx, op in enumerate(operations):
        # release unneeded LX allocations before actual planning
        alloc.deallocate(idx_to_dealloc_bufs.get(idx, []))

        if isinstance(op, ComputedBuffer):
            if isinstance(op.layout, MutationLayoutSHOULDREMOVE):
                continue
            consider_for_scratchpad(op, alloc, idx, core_div_mismatch)
    # logger.info(alloc.lx_usage_hist)
