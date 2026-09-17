# Copyright 2026 The Torch-Spyre Authors.
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

"""Clone a gathered matmul operand into a shape/layout LX_PLANNING can place.

Gathering (e.g. ``torch.index_select``) is basically a memory copy, which
gets almost no benefit from multi-core execution. But the resulting tensor
is usually needed by a matmul -- e.g. paged attention's query-row gather --
which *does* benefit, provided the gathered tensor is (1) sliced to match
the matmul's own per-core division and (2) placed in LX scratchpad instead
of HBM. This pass builds on top of the existing single-core gather to target
exactly that case: it clones the gathered buffer with a shape, device
layout, and work division that mirror what the consuming matmul(s) already
committed to, so the scratchpad allocator can place it in LX like any other
ordinary intermediate.

Why a clone, not an in-place fix: the gather's own device layout is chosen
early (``propagate_spyre_tensor_layouts``), before the matmul's division is
known (``_distribute_work``, much later) -- so the two routinely disagree,
and the allocator reports a "core div mismatch" and leaves the buffer in
HBM. Separately, a buffer read by a ``spyre.restickify`` is unconditionally
barred from LX (``_restickify_barrier``); K/V page gathers usually hit that
(the squeeze/permute/unsqueeze need a real stick-dim swap), so this mostly
ends up targeting the query-row gather.

The general version of this problem -- reconciling a producer's and a
consumer's work divisions so more buffers qualify for LX -- is really a
work-division/LX-planning co-optimization problem, and could be solved that
way instead. This pass is a narrower, lower-cost native fix for one
specific, common shape of that problem (a gather feeding a matmul it
disagrees with), without having to invoke the co-optimizer at all.

Runs as a ``pre_optimization_pass`` (wired in ``scratchpad/allocator.py``'s
``select_allocator()``), positioned after ``_distribute_work`` (so the
matmul's division already exists to copy) and before ``_prepare_buffers``
(so the LX solver sees the clone as an ordinary intermediate, no extra
plumbing needed).

Mechanism, per gathered buffer with at least one qualifying consumer
(``_feeds_matmul_without_restickify``: a matmul reached directly, or through
plain pass-through ops, without ever crossing a restickify -- checked per
direct reader, since one gather commonly feeds several matmuls at once,
e.g. one query gather shared by every KV-block's matmul in an
online-softmax loop):

1. ``_detect_nested_split`` finds the one host axis the gather's flat layout
   leaves whole but every qualifying consumer needs split in two (e.g. paged
   attention's GQA head axis, kv_head x query-in-group) -- required to agree
   across all consumers.
2. ``_build_clone`` gives the clone that split as a real host shape and
   device layout, not just a device-layout relabeling of an unchanged shape,
   via a fresh ``Pointwise`` whose loader recombines the two new indices
   before reading the original buffer.
3. ``_commit_producer_division`` commits a work division on the clone by
   reusing each consumer's own per-symbol core assignment verbatim (never
   re-derived), so producer and consumer(s) agree on which core owns which
   physical slice.
"""

import dataclasses

import sympy
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import ComputedBuffer, Pointwise
from torch_spyre._C import SpyreTensorLayout

from .ir import FixedTiledLayout
from .logging_utils import get_inductor_logger
from .op_spec import TensorWorkDivision
from .pass_utils import (
    commit_tensor_work_division,
    copy_op_metadata,
    iteration_space_from_op,
    redirect_computed_buffer_reads,
)
from .scratchpad.allocator import ScratchpadOptimizationPass

logger = get_inductor_logger("insert_gather_clone")

_GATHER_TARGETS = frozenset({"aten.index.Tensor", "aten.index_select.default"})
_RESTICKIFY_TARGET = "spyre.restickify.default"
_MATMUL_TARGETS = frozenset(
    {"spyre.batched_matmul.default", "aten.bmm.default", "aten.mm.default"}
)
_MAX_CHAIN_HOPS = 4  # generous bound on plain pass-through ops before giving up


def _origin_targets(op: ComputedBuffer) -> set[str]:
    origins = getattr(op, "origins", None) or []
    return {str(getattr(n, "target", None)) for n in origins}


def _single_consumer(graph: GraphLowering, buf_name: str) -> "ComputedBuffer | None":
    """The one ComputedBuffer reading buf_name, or None if not exactly one."""
    consumers = _direct_consumers(graph, buf_name)
    return consumers[0] if len(consumers) == 1 else None


def _direct_consumers(graph: GraphLowering, buf_name: str) -> list[ComputedBuffer]:
    """Every ComputedBuffer with a direct read of buf_name."""
    return [
        op
        for op in graph.operations
        if isinstance(op, ComputedBuffer)
        and buf_name in {d.name for d in op.get_read_writes().reads}
    ]


def _feeds_matmul_without_restickify(
    graph: GraphLowering, consumer: ComputedBuffer, hops_left: int = _MAX_CHAIN_HOPS
) -> bool:
    """Whether consumer is itself a matmul/bmm, or (by single-consumer
    pass-through hops) reaches one, without ever passing through a
    spyre.restickify op.

    Called per direct reader, not once for the whole gathered buffer, so
    each fan-out edge (see module docstring) qualifies independently. Plain
    pass-through ops are neither disqualifying nor qualifying by themselves;
    the walk continues through them looking for the matmul.
    """
    targets = _origin_targets(consumer)
    if _RESTICKIFY_TARGET in targets:
        return False
    if targets & _MATMUL_TARGETS:
        return True
    if hops_left <= 0:
        return False
    nxt = _single_consumer(graph, consumer.get_name())
    if nxt is None:
        return False
    return _feeds_matmul_without_restickify(graph, nxt, hops_left - 1)


@dataclasses.dataclass(frozen=True)
class _NestedSplit:
    """How to split one flat host dim of `src` into (outer, inner) so a clone
    matches `consumer`'s already-committed division on both its host shape
    and its device layout."""

    host_dim: int
    outer_split: int
    inner_new: int
    device_layout: SpyreTensorLayout


def _detect_nested_split(
    src: ComputedBuffer, consumer: ComputedBuffer, read_dep
) -> "_NestedSplit | None":
    """Whether -- and how -- a clone of `src` needs a host dim genuinely split
    in two to match `consumer`'s already-committed division, or None if no
    (clean, single-axis) nested split is needed or possible.

    For each of the consumer's committed splits, check whether its host
    stride in `read_dep` is already a registered device dim on `src`'s own
    layout. Exactly one unregistered stride is handled: paged attention's
    GQA case, where the consumer treats one flat head axis as two nested
    levels (kv_head, query-in-group) while `src` only ever exposed it as one
    flat dim. More than one unregistered stride, or none, is left alone --
    not this pass's problem to solve.

    A device-layout-only fix (leaving `src`'s host shape flat) is not
    enough: the matmul wants kv_head and query-in-group as two genuinely
    separate loop symbols, not one flat symbol behind a clever device
    layout -- which also sidesteps `work_division_from_view`'s "fused
    ownership" canonical-mapping proof, which does not always succeed even
    for physically valid splits. So this locates the *host* dim to split
    too, not just the device dim.
    """
    ownership = getattr(consumer, "iteration_space_ownership", None)
    if ownership is None:
        return None
    old_layout = src.layout
    old_stl = old_layout.device_layout
    registered = {s for s in old_stl.stride_map if s > 0}

    missing = []
    for sym, split in ownership.work_slices.items():
        if split <= 1:
            continue
        coeff = read_dep.index.coeff(sym)
        if coeff == 0:
            continue
        coeff = int(coeff)
        if coeff not in registered:
            missing.append((coeff, split))
    if len(missing) != 1:
        return None
    outer_stride, outer_split = missing[0]

    # Find an existing device dim this outer stride coarsens: same axis
    # (stride divides evenly), real (device_size > 1, skip padding/unused
    # dims), and the regrouping covers it exactly.
    host_strides = list(old_layout.stride)
    host_sizes = list(old_layout.size)
    for dim, stride in enumerate(old_stl.stride_map):
        if stride <= 0 or old_stl.device_size[dim] <= 1:
            continue
        if outer_stride % stride != 0:
            continue
        inner_new = outer_stride // stride
        if inner_new <= 1:
            continue
        axis_extent = old_stl.device_size[dim]
        if axis_extent % inner_new != 0:
            continue
        outer_new = axis_extent // inner_new
        if outer_new != outer_split:
            continue  # must match what the consumer actually needs
        if stride not in host_strides:
            continue  # no host axis (e.g. stick padding) at this exact stride
        host_dim = host_strides.index(stride)
        if host_sizes[host_dim] != axis_extent:
            continue  # host/device extents disagree -- likely padded, skip

        # Other axes may be padded beyond their logical host extent for a
        # reason specific to `src` (e.g. a gather's indirect write needs its
        # entry dim grown to a whole stick -- see `padded_entry_output_stl`).
        # The clone has no such requirement, so shrink every other
        # host-mapped axis back to its logical extent, keeping
        # `_commit_producer_division` unblocked by padding unrelated to the
        # split made here.
        #
        # Except the last device dim: that's the physical stick-element axis
        # (always exactly `elems_per_stick`), not a host-mapped one -- even
        # though its stride (1) often numerically matches the host's own
        # innermost stride too, whenever a host axis (e.g. head_size) spans
        # more than one stick. Resizing it to the host extent silently
        # claims an oversized stick, which the low-level scheduler rejects
        # with an opaque "Could not find any suitable dimension mapping"
        # instead of a clean, catchable error.
        stick_dim = len(old_stl.stride_map) - 1
        new_device_size = [
            host_sizes[host_strides.index(s)]
            if d != dim and d != stick_dim and s > 0 and size > 1 and s in host_strides
            else size
            for d, (s, size) in enumerate(zip(old_stl.stride_map, old_stl.device_size))
        ]
        new_stride_map = list(old_stl.stride_map)
        new_device_size[dim] = inner_new
        new_device_size.insert(dim, outer_new)
        new_stride_map.insert(dim, outer_stride)
        return _NestedSplit(
            host_dim=host_dim,
            outer_split=outer_split,
            inner_new=inner_new,
            device_layout=SpyreTensorLayout(
                new_device_size, new_stride_map, old_stl.device_dtype
            ),
        )
    return None


def _read_dep(consumer: ComputedBuffer, src_name: str):
    return next(
        (d for d in consumer.get_read_writes().reads if d.name == src_name),
        None,
    )


def _detect_nested_split_for_all(
    src: ComputedBuffer, consumers: list[ComputedBuffer]
) -> "tuple[_NestedSplit, list] | None":
    """`_detect_nested_split`, required to agree across every consumer.

    Consumers sharing one gathered operand all read the same memory; if they
    don't independently derive the identical split, it's safer to leave the
    buffer alone than to guess which one (if any) is right.
    """
    src_name = src.get_name()
    read_deps = []
    nested = None
    for consumer in consumers:
        read_dep = _read_dep(consumer, src_name)
        if read_dep is None:
            return None
        this_nested = _detect_nested_split(src, consumer, read_dep)
        if this_nested is None:
            return None
        if nested is None:
            nested = this_nested
        elif this_nested != nested:
            return None
        read_deps.append(read_dep)
    if nested is None:
        return None
    return nested, read_deps


def _build_clone(src: ComputedBuffer, consumers: list[ComputedBuffer]):
    """Returns (clone_buf, read_deps), or None if no clone is needed/possible.

    read_deps (parallel to consumers) is returned alongside the clone because
    `_commit_producer_division` needs the same consumer-side read dependencies
    again afterward, once the clone has a real name -- no reason to recompute
    them.
    """
    detected = _detect_nested_split_for_all(src, consumers)
    if detected is None:
        return None
    nested, read_deps = detected

    old_layout = src.layout
    host_dim, inner_new = nested.host_dim, nested.inner_new
    loader = src.make_loader()

    def inner_fn(index, _loader=loader, _dim=host_dim, _inner=inner_new):
        index = list(index)
        outer_idx, inner_idx = index[_dim], index[_dim + 1]
        old_index = index[:_dim] + [outer_idx * _inner + inner_idx] + index[_dim + 2 :]
        return _loader(old_index)

    old_size = list(old_layout.size)
    old_stride = list(old_layout.stride)
    new_size = (
        old_size[:host_dim] + [nested.outer_split, inner_new] + old_size[host_dim + 1 :]
    )
    new_stride = (
        old_stride[:host_dim]
        + [old_stride[host_dim] * inner_new, old_stride[host_dim]]
        + old_stride[host_dim + 1 :]
    )

    clone_tb = Pointwise.create(
        device=src.get_device(),
        dtype=src.get_dtype(),
        inner_fn=inner_fn,
        ranges=new_size,
    )
    clone_layout = FixedTiledLayout(
        old_layout.device,
        old_layout.dtype,
        new_size,
        new_stride,
        nested.device_layout,
        offset=old_layout.offset,
    )
    clone_buf = ComputedBuffer(
        name=None,
        layout=clone_layout,
        data=clone_tb.data.data,
    )
    clone_buf.origins = set(src.origins)
    clone_buf.origin_node = getattr(src, "origin_node", None)
    copy_op_metadata(src, clone_buf)
    return clone_buf, read_deps


def _commit_producer_division(
    clone_buf: ComputedBuffer,
    consumers: list[ComputedBuffer],
    read_deps: list,
) -> None:
    """Commit a work division on `clone_buf` mirroring every consumer's own
    already-committed division, so they all agree on which core owns which
    physical slice of the clone's memory -- not just on its device layout.

    Independently re-deriving a division (e.g. via
    `commit_iteration_space_ownership`) risks a different core-to-slice
    convention than the consumers already committed to -- the "fused
    ownership" mismatch this whole approach exists to avoid. So instead,
    reuse each consumer's own per-symbol core assignment directly: a
    `read_dep.index` (a consumer's access to the clone) and the clone's own
    write dep assign coefficients to loop symbols using the same physical
    offsets, so a matching coefficient means the same physical axis. When
    several consumers share the clone, each must map onto the same clone
    axis the same way -- any disagreement means nothing is committed, since
    committing one consumer's view could silently misdescribe another's
    read. Clone symbols nothing maps onto (e.g. the stick/elem axis) are
    left for `commit_tensor_work_division` to fill in as unsplit.
    """
    write_dep = next(
        (
            d
            for d in clone_buf.get_read_writes().writes
            if d.name == clone_buf.get_name()
        ),
        None,
    )
    if write_dep is None:
        return
    producer_syms: dict[int, sympy.Symbol] = {}
    for sym in iteration_space_from_op(clone_buf):
        coeff = write_dep.index.coeff(sym)
        if coeff != 0:
            producer_syms[int(coeff)] = sym

    work_slices: dict[sympy.Symbol, int] = {}
    owners: dict[sympy.Symbol, sympy.Expr] = {}
    num_cores = None
    for consumer, read_dep in zip(consumers, read_deps):
        ownership = getattr(consumer, "iteration_space_ownership", None)
        if ownership is None:
            continue
        if num_cores is None:
            num_cores = ownership.physical_core_count
        elif num_cores != ownership.physical_core_count:
            return  # consumers disagree on the physical core domain
        for sym, split in ownership.work_slices.items():
            if split <= 1:
                continue
            coeff = read_dep.index.coeff(sym)
            if coeff == 0:
                continue
            producer_sym = producer_syms.get(int(coeff))
            if producer_sym is None:
                return  # a committed consumer axis has no matching clone axis
            owner_expr = ownership.core_id_to_work_slice[sym]
            if producer_sym in work_slices:
                if (
                    work_slices[producer_sym] != split
                    or owners[producer_sym] != owner_expr
                ):
                    return  # consumers disagree on this clone axis
            else:
                work_slices[producer_sym] = split
                owners[producer_sym] = owner_expr

    if not work_slices:
        return
    commit_tensor_work_division(
        clone_buf, TensorWorkDivision(work_slices, owners, num_cores=num_cores)
    )


class InsertGatherClonePass(ScratchpadOptimizationPass):
    """Clone each gathered matmul operand with a consumer-matching device
    layout. Registered as a ``pre_optimization_pass`` in ``select_allocator()``
    -- see the module docstring for why this position, not an earlier pass,
    is load-bearing.
    """

    def apply_pass(self, graph: GraphLowering) -> None:
        operations = graph.operations
        for src in list(operations):
            if not isinstance(src, ComputedBuffer):
                continue
            if not (_origin_targets(src) & _GATHER_TARGETS):
                continue

            src_name = src.get_name()
            consumers = [
                c
                for c in _direct_consumers(graph, src_name)
                if _feeds_matmul_without_restickify(graph, c)
            ]
            if not consumers:
                continue

            built = _build_clone(src, consumers)
            if built is None:
                continue
            clone_buf, read_deps = built
            clone_buf.name = graph.register_buffer(clone_buf)
            graph.register_operation(clone_buf)
            operations.remove(clone_buf)
            operations.insert(operations.index(src) + 1, clone_buf)
            clone_name = clone_buf.get_name()
            _commit_producer_division(clone_buf, consumers, read_deps)

            for consumer in consumers:
                redirect_computed_buffer_reads(
                    consumer,
                    {src_name: clone_name},
                    operations,
                    pass_name="insert_gather_clone",
                    reason=(
                        "clone gathered matmul operand with a device layout "
                        "matching the consumer's already-committed division"
                    ),
                )
            logger.info(
                "insert_gather_clone: cloned %s -> %s, redirected %s",
                src_name,
                clone_name,
                [c.get_name() for c in consumers],
            )
