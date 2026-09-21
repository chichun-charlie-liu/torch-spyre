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

"""Re-commit a gathered matmul operand's own work division, in place.

Gathering (e.g. ``torch.index_select``) is basically a memory copy, which
gets almost no benefit from multi-core execution. But the resulting tensor
is usually needed by a matmul -- e.g. paged attention's query-row gather --
which *does* benefit, provided the gathered tensor is (1) sliced to match
the matmul's own per-core division and (2) placed in LX scratchpad instead
of HBM. In practice it routinely disagrees with the matmul instead: the
gather's own device layout is chosen early (``propagate_spyre_tensor_
layouts``), before the matmul's division is known (``_distribute_work``,
much later), and the allocator reports a "core div mismatch" and leaves the
buffer in HBM. An intermediate op between the gather and the matmul (most
often a restickify, when a real stick-dim swap is needed) routinely has the
identical disagreement, for the identical reason: its own division was
chosen independently of what the matmul would end up needing.

This pass fixes that by re-committing a work division directly on an
*existing* buffer -- never by building a new one. Which buffer: whichever
one, walking forward from the gather through any single-consumer
pass-through hops, is *itself* read directly by the matmul
(``_pre_matmul_buffer``) -- the gather itself if it feeds the matmul with no
real intermediate, or an intermediate op otherwise. If that target wasn't
the gather itself, the gather's own division is also re-committed to match
the (now-corrected) intermediate's, for the same reason: left alone, it
would still disagree with what the intermediate now needs from it, which
can by itself force the intermediate's kernel into an unfavorably small
core count to reconcile the gap.

The one hazard, found by testing this against paged attention's actual K/V
chain: a division that looks like a clean match can still be *wrong* if the
matmul's own division includes an axis the target reads with a *zero*
coefficient -- a genuine broadcast (e.g. K/V is read identically regardless
of which output row is being computed), not a mislabeled split.
`_missing_strides` treats that axis as "no gap" (correctly -- there's no
missing physical axis to add), but that axis's split then never reaches
`work_slices` either (nothing maps onto it). An earlier version of this pass
still set `num_cores` to the *consumer's* full physical core count in that
case, claiming a core count no combination of the committed splits actually
accounted for -- valid only if the buffer's kernel replicates the dropped
axis across the shortfall, which nothing arranges for it to do. `producer`'s
own kernel then ran that axis unsplit, one real copy where several were
needed, and both the direct-residency check and the LX-relayout planner's
own eligibility gate correctly refused to place it in LX
(`scratchpad/utils.py`'s ``get_ncores_for_buffers``: "There is no
single-base LX broadcast, so treat it as a core-division mismatch and keep
the buffer in HBM (correct, just unpinned)") -- happening to both K's
restickify and V's broadcast-expand intermediate, which this pass had
otherwise already correctly stopped disagreeing with the matmul on. Fewer
hops, but each lost LX placement it already had via the *existing*
LX-relayout planner, which already handles this precise broadcast (a real,
physical copy fanning distinct data out across more cores) correctly on its
own, unprompted, whenever a buffer's own (untouched) division is internally
consistent (see ``collect_lx_relayout_plans``, ``lx_relayout.py``).

So `num_cores` is never copied from a consumer: `_commit_matching_division`
sets it to the product of `producer`'s *own* committed splits, always
internally consistent by construction. This doesn't change where a
broadcast-bearing buffer like K's restickify or V's broadcast-expand
intermediate lands -- the residency and relayout-eligibility checks above
key off the *consumer's* read pattern, not what the producer claims about
itself, so a buffer genuinely read by fewer cores than its matmul runs stays
correctly barred from direct LX placement either way. What an honest count
changes is candidacy for a *relayout*: `collect_lx_relayout_plans`'s own
gate (`source_view.num_cores != source_num_cores`) reads the producer's own
declared count, so an inflated one made every such buffer invisible to that
planner too, not just to direct placement -- silently, since that specific
check has no logged rejection reason.

A version before that cloned the gather into a reshaped buffer whenever its
consumer needed a genuinely nested split (e.g. paged attention's GQA head
axis, kv_head x query-in-group, out of one flat head dim), unconditionally;
that clone routinely won LX placement for itself while competing with --
and displacing -- buffers the matmul needed more, a net loss. The reason it
was needed at all: the matmul's own division was the cost model's default,
chosen with no regard for what K/V's own layout could offer for free.

That default is itself avoidable. `_cost_model_matmul_planner`
(`work_division.py`) prices a batch-dim split on a true BMM with a
`log2(b) * 10us` penalty on top of a term that rewards splitting the M/token
axis instead -- enough that it prefers *zero* batch split even when the
batch axis alone exactly covers every core (confirmed for paged attention's
K/V-consuming matmul: `b=(1, 1) m=16 n=2 k=1` even though the batch size
alone is 32). That model is strictly per-op cost; it has no term for the
mismatch this creates with K/V's own batch-shaped layout (kv_head x
query-in-group), which is exactly what forces the reconciliation the rest
of this pass exists to avoid. So `_force_batch_division` overrides it:
before reading a matmul's committed division, this pass first tries
re-committing that matmul's own division to split purely along its batch
dims (`apply_splits`, bypassing the cost model's search entirely) --
whenever that's legal, the buffer being reconciled against it can then
usually match with no reconciliation at all.

Forcing the matmul this way trades one asymmetry for another: it is a
clean win for whichever operand's own layout is already batch-shaped (K/V),
but the *other* operand (Q, whose own gather output is still one flat
"heads" dim, not yet split into kv_head x query-in-group) now needs exactly
the nested split described above, which it didn't need against the cost
model's original (non-batch) division. So the nested-split clone is not
gone -- `_fix_gather_via_clone` -- it is just narrowed to only the case that
makes it safe: a gather feeding a matmul *directly* (no intermediate), and
only after `_force_batch_division` has already been tried on that matmul.
The clone's own division is then a clean match to a batch-aligned matmul
-- not a mismatched one it has to compete with other buffers to fix -- which
is the difference between this and the version that was reverted.

Runs as a ``pre_optimization_pass`` (wired in ``scratchpad/allocator.py``'s
``select_allocator()``), positioned after ``_distribute_work`` (so the
matmul's division already exists to copy, before `_force_batch_division`
overrides it) and before ``_prepare_buffers`` (so the LX-relayout planner
sees the corrected division, with no extra plumbing needed).

Mechanism, per gathered buffer with at least one qualifying direct-consumer
chain (checked per direct reader, since one gather commonly feeds several
matmuls at once, e.g. one query gather shared by every KV-block's matmul in
an online-softmax loop):

1. ``_pre_matmul_buffer`` walks forward from the gather's direct consumer,
   through single-consumer hops, to the buffer immediately read by a
   matmul -- the gather itself (zero hops) or some intermediate.
2. ``_fix_target_division`` tries `_force_batch_division` on each matmul
   consumer first (best-effort, win or lose), then to re-commit *that
   buffer's own* division to match, reusing the matmul's own
   already-committed per-symbol core assignment verbatim (never re-derived)
   but its *own* honestly-computed core count (see above), so producer and
   matmul agree on which core owns which physical slice. Only commits when
   every one of the target's readers -- not just this edge's -- is itself
   directly a matmul, and when none of them need a nested split or
   broadcast (``_missing_strides`` empty against the target's own,
   already-adequate device layout); mutating a shared buffer's division
   based on only some of its readers would silently misdescribe the others.
3. If step 2 failed and the target *was* the gather itself (no
   intermediate), ``_fix_gather_via_clone`` tries the nested-split clone
   described above instead.
4. If step 2 succeeded and the target *wasn't* the gather itself (a real
   intermediate sat in between), ``_fix_broadcast_via_widen`` first checks
   whether that intermediate is a *pure broadcast* of the gather -- present
   in the intermediate's own shape, absent from the gather's, read with a
   zero coefficient (``_detect_broadcast_axis``). If so, the gather is
   widened *in place*, under its own name, to the intermediate's shape (its
   own body wrapped to drop the broadcast axis's index -- no new buffer, and
   no recombination needed the way a genuine nested split needs, since
   every value of a broadcast axis reads the identical source coordinate),
   the matmul consumer(s) are redirected to read the widened gather
   directly, and the now-dead intermediate is dropped. This is the
   broadcast-shaped mirror of step 3's split-shaped clone: same "gather
   feeds a matmul through exactly one qualifying hop" precondition, same
   reason it waits until *after* step 2 has already tried to give the
   matmul a batch-aligned division (so what the gather widens to match is a
   clean, honest division, not one it would still have to compete with
   other buffers to reconcile) -- except here the gather absorbs the
   intermediate instead of the intermediate surviving alongside a clone.
5. Otherwise (step 2 succeeded, an intermediate sat in between, but it
   wasn't a pure broadcast of the gather -- e.g. K's restickify, a genuine
   stride/stick-dim change, not just a broadcast), ``_propagate_to_gather``
   tries to re-commit the *gather's own* division to match that
   intermediate's -- now corrected by step 2, so it's as trustworthy a
   reference as the matmul itself. Only commits when the intermediate is
   the gather's *only* reader, for the same reason as step 2.
"""

import dataclasses
import math
import os

import sympy
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import ComputedBuffer, Layout, Pointwise, Reduction
from torch_spyre._C import SpyreTensorLayout

from . import config
from .constants import BATCH_MATMUL_FP8_OP, BATCH_MATMUL_OP
from .ir import FixedTiledLayout
from .logging_utils import get_inductor_logger
from .op_spec import TensorWorkDivision
from .pass_utils import (
    commit_tensor_work_division,
    concretize_expr,
    copy_op_metadata,
    iteration_space_from_op,
    redirect_computed_buffer_reads,
    replace_computed_buffer_body,
)
from .scratchpad.allocator import ScratchpadOptimizationPass
from .work_division import (
    _pick_innermost_output_dim,
    _single_input_row_dims,
    apply_splits,
    work_division_context_for_op,
    work_division_splits_are_legal,
)

logger = get_inductor_logger("insert_gather_clone")


def _cache_key(cached_method: object) -> str:
    """Return the cache attribute name a ``cache_on_self``/``cache_on_self_
    and_args`` method stores its result under, by inspecting its own
    ``.clear_cache`` closure -- resolved once at import time so an upstream
    rename fails loudly here rather than silently no-oping later. Mirrors
    ``wsr/coarse_tile.py``'s own helper of the same name (not imported
    cross-module to avoid coupling this file to that one's internals).
    """
    clear_fn = cached_method.clear_cache  # type: ignore[attr-defined]
    for i, name in enumerate(clear_fn.__code__.co_freevars):
        if name == "key":
            return clear_fn.__closure__[i].cell_contents
    raise AttributeError(
        f"Cannot find 'key' in clear_cache closure of {cached_method!r}"
    )


# Cleared on `src` after `_fix_broadcast_via_widen` mutates its layout in
# place: both are derived from `layout`/`data`, so a stale cached result
# from before the widen would silently describe the old, unwidened buffer.
_LAYOUT_FREE_SYMS_KEY = _cache_key(Layout.get_free_symbol_uses)
_COMPUTED_BUF_FREE_SYMS_KEY = _cache_key(ComputedBuffer.get_free_symbol_uses)


def _clear_cache(obj: object, key: str) -> None:
    # cache_on_self/cache_on_self_and_args store results via
    # object.__setattr__ to bypass frozen-dataclass guards (Loops, Layout),
    # so clearing must also use object.__delattr__ -- plain delattr() raises
    # FrozenInstanceError.
    if hasattr(obj, key):
        object.__delattr__(obj, key)


_GATHER_TARGETS = frozenset({"aten.index.Tensor", "aten.index_select.default"})
_MATMUL_TARGETS = frozenset(
    {"spyre.batched_matmul.default", "aten.bmm.default", "aten.mm.default"}
)
_MAX_CHAIN_HOPS = 4  # generous bound on plain pass-through ops before giving up


def _origin_targets(op: ComputedBuffer) -> set[str]:
    origins = getattr(op, "origins", None) or []
    return {str(getattr(n, "target", None)) for n in origins}


def _is_matmul(op: ComputedBuffer) -> bool:
    return bool(_origin_targets(op) & _MATMUL_TARGETS)


def _matmul_batch_split(op: ComputedBuffer) -> "dict[sympy.Symbol, int] | None":
    """The (symbol -> full extent) split that divides `op` -- a batched
    matmul -- purely along its batch dims, leaving M/N/K unsplit, or None if
    `op` isn't a batched matmul, or its axes can't be classified this way.

    Mirrors `_cost_model_matmul_planner`'s own row-dim classification
    (`work_division.py`), so this agrees with the cost model on which axes
    are "batch" for this exact op -- it's the same classification, just
    asked for a specific candidate instead of the cheapest one. See
    `_force_batch_division` for why that candidate is worth committing
    despite the cost model not choosing it on its own.
    """
    if not isinstance(op.data, Reduction):
        return None
    if op.data.reduction_type not in (BATCH_MATMUL_OP, BATCH_MATMUL_FP8_OP):
        return None

    ctx = work_division_context_for_op(op)
    input_tds, output_td = ctx.tensor_deps[:-1], ctx.tensor_deps[-1]

    output_coord_vars = {
        v
        for e in output_td.device_coords[:-1]
        for v in e.free_symbols
        if isinstance(v, sympy.Symbol)
    }
    ordered_output_coord_vars = [
        d for d in ctx.it_space_adjusted if d in output_coord_vars
    ]
    n_dims = [d for d in ordered_output_coord_vars if d in ctx.stick_vars]
    row_dims = [d for d in ordered_output_coord_vars if d not in ctx.stick_vars]
    if len(n_dims) != 1 or not row_dims:
        return None

    m_candidates = _single_input_row_dims(row_dims, input_tds)
    if len(m_candidates) == 1:
        m_dim = m_candidates[0]
    elif len(m_candidates) > 1:
        m_dim = _pick_innermost_output_dim(m_candidates, output_td.dep.index)
        if m_dim is None:
            return None
    else:
        return None
    batch_dims = [d for d in row_dims if d != m_dim]
    if not batch_dims:
        return None

    return {d: int(concretize_expr(ctx.it_space_adjusted[d])) for d in batch_dims}


def _force_batch_division(op: ComputedBuffer) -> bool:
    """Re-commit `op`'s (a batched matmul's) own division to split purely
    along its batch dims, leaving M/N/K unsplit -- overriding whatever
    `_distribute_work`'s cost-model search chose, best-effort (returns False
    rather than raising if not applicable or not legal).

    `_cost_model_matmul_planner` (`work_division.py`) prices a batch split on
    a true BMM with a `log2(b) * 10us` penalty (`_BMM_BATCH_SPLIT_PENALTY_US`)
    on top of an `m_lane_underuse_us` term that rewards splitting the M/token
    axis instead -- together enough to make the model prefer *zero* batch
    split even when the batch axis alone exactly covers every core (confirmed
    for paged attention's K/V-consuming matmul: `b=(1, 1) m=16 n=2 k=1` even
    though `B=32` fills the full core budget on its own). That model is
    strictly per-op cost; it has no term for the mismatch this creates with
    K/V's own batch-shaped layout (kv_head x query-in-group), which is
    exactly what forces the shuffle/broadcast reconciliation the rest of this
    pass exists to avoid. Splitting purely by batch instead means K/V's own
    already-batch-divided layout can agree with the matmul directly, with no
    intervening reconciliation at all -- see module docstring for whether
    that trade (a costlier-by-this-model matmul, no reconciliation) is worth
    it in practice; this is deliberately unconditional so it can be measured.
    """
    splits = _matmul_batch_split(op)
    if splits is None:
        return False
    if math.prod(splits.values()) > config.sencores:
        return False  # more batch than cores available -- not attempted
    if not work_division_splits_are_legal(op, splits):
        return False
    apply_splits(op, splits)
    return True


def _matmul_axes(
    op: ComputedBuffer,
) -> "dict[str, tuple[sympy.Symbol, int]] | None":
    """Classify `op` -- a batched matmul -- into four named axes: `T` (the
    token/M row dim), `St` (the one stick/N dim), `Hnum` (the larger-extent
    batch dim, e.g. kv_head), `Hgrp` (the smaller-extent batch dim, e.g.
    query-in-group) -- or None if `op` isn't a batched matmul, its axes
    can't be classified this way, or it doesn't have exactly two batch dims.
    Returns `{name: (symbol, full_extent)}`.

    Same classification `_matmul_batch_split` uses (mirroring
    `_cost_model_matmul_planner`'s own row-dim split), just naming every
    axis individually instead of only the batch ones, for
    `_force_custom_matmul_division`'s experiments with matmul divisions the
    cost model wouldn't choose and `_force_batch_division` doesn't produce
    either (mixed T/batch splits). `Hnum`/`Hgrp` are disambiguated by
    extent (larger first) -- true for this benchmark's shapes
    (`num_kv_heads=8` > `num_queries_per_kv=4`), but not a general
    guarantee; verify against the actual op if reusing this for a different
    shape.
    """
    if not isinstance(op.data, Reduction):
        return None
    if op.data.reduction_type not in (BATCH_MATMUL_OP, BATCH_MATMUL_FP8_OP):
        return None

    ctx = work_division_context_for_op(op)
    input_tds, output_td = ctx.tensor_deps[:-1], ctx.tensor_deps[-1]

    output_coord_vars = {
        v
        for e in output_td.device_coords[:-1]
        for v in e.free_symbols
        if isinstance(v, sympy.Symbol)
    }
    ordered_output_coord_vars = [
        d for d in ctx.it_space_adjusted if d in output_coord_vars
    ]
    n_dims = [d for d in ordered_output_coord_vars if d in ctx.stick_vars]
    row_dims = [d for d in ordered_output_coord_vars if d not in ctx.stick_vars]
    if len(n_dims) != 1 or not row_dims:
        return None

    m_candidates = _single_input_row_dims(row_dims, input_tds)
    if len(m_candidates) == 1:
        m_dim = m_candidates[0]
    elif len(m_candidates) > 1:
        m_dim = _pick_innermost_output_dim(m_candidates, output_td.dep.index)
        if m_dim is None:
            return None
    else:
        return None
    batch_dims = [d for d in row_dims if d != m_dim]
    if len(batch_dims) != 2:
        return None

    def extent(d: sympy.Symbol) -> int:
        return int(concretize_expr(ctx.it_space_adjusted[d]))

    st_dim = n_dims[0]
    hnum_dim, hgrp_dim = sorted(batch_dims, key=extent, reverse=True)
    return {
        "T": (m_dim, extent(m_dim)),
        "St": (st_dim, extent(st_dim)),
        "Hnum": (hnum_dim, extent(hnum_dim)),
        "Hgrp": (hgrp_dim, extent(hgrp_dim)),
    }


def _force_custom_matmul_division(op: ComputedBuffer, spec: "dict[str, int]") -> bool:
    """Re-commit `op`'s (a batched matmul's) own division to an explicit,
    named combination of splits (see `_matmul_axes` for the axis names) --
    e.g. `{"T": 16, "Hnum": 2}` -- bypassing the cost model and
    `_force_batch_division`'s pure-batch choice alike, best-effort (returns
    False rather than raising if not applicable or not legal). Used only
    from the `TORCH_SPYRE_MATMUL_SPLIT`-driven experiment below; see there.
    """
    axes = _matmul_axes(op)
    if axes is None:
        return False
    splits: dict[sympy.Symbol, int] = {}
    for name, factor in spec.items():
        axis = axes.get(name)
        if axis is None:
            return False
        sym, extent = axis
        if factor <= 0 or extent % factor != 0:
            return False
        splits[sym] = factor
    if math.prod(splits.values()) > config.sencores:
        return False
    if not work_division_splits_are_legal(op, splits):
        return False
    apply_splits(op, splits)
    return True


# TEMP: experiment scaffold for testing matmul work divisions the cost model
# and `_force_batch_division` wouldn't choose (e.g. mixed T/batch splits).
# `TORCH_SPYRE_MATMUL_SPLIT="T:16,St:2"` (comma-separated "Axis:factor"
# pairs, axis names from `_matmul_axes`) forces that exact division on every
# matmul this pass touches, in place of `_force_batch_division`, and (in
# `apply_pass`) skips `_fix_gather_via_clone` entirely so Q falls back to
# its natural gather+shuffle path regardless of which division is forced --
# remove this whole scaffold (this block, the wiring in `_fix_target_
# division` and `apply_pass`, and the `os` import) once this round of
# experiments is done.
def _parse_matmul_split_spec(raw: "str | None") -> "dict[str, int] | None":
    if not raw:
        return None
    spec: dict[str, int] = {}
    for term in raw.split(","):
        name, _, factor = term.partition(":")
        spec[name.strip()] = int(factor)
    return spec


_MATMUL_SPLIT_SPEC = _parse_matmul_split_spec(
    os.environ.get("TORCH_SPYRE_MATMUL_SPLIT")
)

# TEMP: gates the re-enabled `_fix_broadcast_via_widen` call in `apply_pass`
# (see the plan file this fix came from,
# `/home/cliu/.claude/plans/generic-percolating-backus.md`, for the full
# root-cause writeup). Previously shipped as part of "v18" and reverted for
# a reproducible wrong-value bug (max_diff=0.00065); `_fix_broadcast_via_
# widen`'s device-layout construction now reinstates a real slot for the
# gather's own indexed coordinate instead of borrowing the broadcast
# consumer's layout wholesale. Still isolated behind this flag -- not
# proven correct beyond this investigation's one V-shaped repro -- so a bad
# result can be reverted with a one-line env-var change. Remove this flag
# (and fold the call into the unconditional path) once confirmed safe more
# broadly.
_ENABLE_BROADCAST_WIDEN = os.environ.get("TORCH_SPYRE_ENABLE_BROADCAST_WIDEN") == "1"

# TEMP: control-group scaffold. TORCH_SPYRE_MATMUL_SPLIT_ONLY=1 makes
# `apply_pass` force the matmul's own division (via `_force_custom_matmul_
# division`, same as the normal `_MATMUL_SPLIT_SPEC` path) and do
# *nothing* else -- no re-committing a pre-matmul buffer's division to
# match (`_fix_target_division`'s second half), no propagation onto the
# raw gather (`_propagate_to_gather`), no clone/widen. Answers "how much
# of the division-propagation mechanism's effect is just forcing the
# matmul, versus the propagation onto Q/K/V itself" -- i.e. isolates
# whatever `_distribute_work`/the scratchpad allocator's own existing
# machinery does on its own once the matmul's division is fixed, before
# justifying a dedicated propagation mechanism at all. Remove once that
# comparison is done.
_MATMUL_SPLIT_ONLY = os.environ.get("TORCH_SPYRE_MATMUL_SPLIT_ONLY") == "1"


def _force_matmul_splits_only(graph: GraphLowering) -> None:
    if _MATMUL_SPLIT_SPEC is None:
        return
    for op in graph.operations:
        if isinstance(op, ComputedBuffer) and _is_matmul(op):
            _force_custom_matmul_division(op, _MATMUL_SPLIT_SPEC)


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


def _pre_matmul_buffer(
    graph: GraphLowering, buf: ComputedBuffer, hops_left: int = _MAX_CHAIN_HOPS
) -> "ComputedBuffer | None":
    """`buf` itself, if every one of its direct consumers is itself a matmul,
    else the result of walking forward through `buf`'s single consumer (if
    there's exactly one) looking for such a buffer -- or None if none is
    found within `hops_left` hops, or the walk dead-ends (no consumer, or
    more than one).

    This is the buffer whose division should be fixed: whichever one, along
    the path from the original gather, is actually read directly by a
    matmul. See module docstring for why fixing an earlier buffer in the
    chain instead (e.g. the gather itself, when a real intermediate sits
    between it and the matmul) would only mirror that intermediate's own,
    equally unreliable division rather than the matmul's.
    """
    consumers = _direct_consumers(graph, buf.get_name())
    if consumers and all(_is_matmul(c) for c in consumers):
        return buf
    if hops_left <= 0:
        return None
    nxt = _single_consumer(graph, buf.get_name())
    if nxt is None:
        return None
    return _pre_matmul_buffer(graph, nxt, hops_left - 1)


def _missing_strides(
    src: ComputedBuffer, consumer: ComputedBuffer, read_dep
) -> list[tuple[int, int]]:
    """Every (coeff, extent) in `consumer`'s own read of `src` whose host
    stride isn't already a registered device dim on `src`'s own layout.

    Empty means `src`'s existing device layout already has a physical axis
    for every symbol `consumer` addresses -- no structural change needed,
    only `src`'s *work division* (which core owns which slice) might still
    disagree with what `consumer` committed to. Any nonempty result -- a
    genuine nested split needed (e.g. one flat head axis addressed as two
    nested levels) or an outright broadcast (a consumer axis with no
    corresponding source axis at all) -- is left alone; neither is a
    division-only fix, and this pass does not build a clone for the former
    (see module docstring).
    """
    old_stl = src.layout.device_layout
    registered = {s for s in old_stl.stride_map if s > 0}
    missing = []
    for sym, extent in iteration_space_from_op(consumer).items():
        coeff = read_dep.index.coeff(sym)
        if coeff == 0:
            continue
        coeff = int(coeff)
        if coeff not in registered:
            missing.append((coeff, int(extent)))
    return missing


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
    in two to match how `consumer` actually addresses it, or None if `_missing_
    strides` doesn't report exactly one gap, or that one gap isn't a clean,
    single-axis split of an existing device dim.

    Reserved for the case `_commit_matching_division` (a division-only fix)
    can't handle: a consumer axis with a *nonzero* coefficient into `src`
    that has no registered device stride at all -- e.g. paged attention's
    GQA head axis, addressed by the matmul as two nested levels (kv_head,
    query-in-group) while `src`'s gather only ever produced one flat dim.
    Distinguished from a broadcast (a consumer axis with a *zero*
    coefficient, which `_missing_strides` never reports at all) by that
    nonzero coefficient: a broadcast needs no new host axis, just
    replication `_commit_matching_division` alone cannot arrange either,
    which is why this pass otherwise leaves it alone (see module docstring).

    A device-layout-only fix (leaving `src`'s host shape flat) is not
    enough: the matmul wants kv_head and query-in-group as two genuinely
    separate loop symbols, not one flat symbol behind a clever device
    layout -- which would also require proving a "fused ownership"
    canonical-mapping that does not always succeed even for physically valid
    splits. So this locates the *host* dim to split too, not just the
    device dim.
    """
    old_layout = src.layout
    old_stl = old_layout.device_layout

    missing = _missing_strides(src, consumer, read_dep)
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
        # entry dim grown to a whole stick). The clone has no such
        # requirement, so shrink every other host-mapped axis back to its
        # logical extent, keeping `_commit_matching_division` unblocked by
        # padding unrelated to the split made here.
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


@dataclasses.dataclass(frozen=True)
class _BroadcastAxis:
    """How `target`'s own `data.ranges` correspond to `src`'s: `dim` is the
    one extra (broadcast) position in `target` that `src` doesn't have at
    all, and `src_positions[j]` is, for `target`'s j-th *other* position,
    the corresponding position in `src`'s own (full, unsqueezed) ranges --
    any of `src`'s own positions absent from `src_positions` are its size-1
    dims that a downstream ``squeeze`` dropped before `target` ever reads it
    (index_select's own output keeps a size-1 "which page" dim `target`'s
    chain squeezes away while also unsqueezing in the broadcast dim -- see
    module docstring), each standing for a fixed index of 0.
    """

    dim: int
    src_positions: tuple[int, ...]


def _detect_broadcast_axis(
    src: ComputedBuffer, target: ComputedBuffer, read_dep
) -> "_BroadcastAxis | None":
    """Whether -- and how -- `target` is a pure broadcast of `src`: `target`'s
    own `data.ranges`, once its one broadcast axis (read with a *zero*
    coefficient) is set aside, correspond one-for-one, *in the same physical
    order*, to `src`'s own `data.ranges` with any size-1 positions removed.
    None if the shapes don't correspond this way.

    Matching sizes alone is not enough: two same-sized host dims (e.g. paged
    attention's ``block_size``/``head_size``, both 128) reordered by a real
    restickify-shaped permute look identical under a size-only comparison,
    which would silently widen `src` with the wrong dim swapped in -- wrong
    values, not a caught error (this is exactly how K's restickify was, once,
    briefly, misdetected as a pure broadcast; caught via a nonzero max_diff
    even against this benchmark's vacuous all-zero reference). So each
    remaining position's read-index coefficient must match `src`'s own
    *stride* at the position it's claimed to correspond to, not just its
    size -- the same physical-axis-identity test `_commit_matching_division`
    already relies on elsewhere in this file, applied to `src`'s host layout
    instead of a device one.

    Mirrors `_detect_nested_split`'s nonzero-coefficient case for the
    zero-coefficient one: no missing device stride to fill (there's no
    corresponding source axis at all to register a stride for), so this
    doesn't reuse `_missing_strides` -- it works directly off both buffers'
    own host `ranges`/`stride`, which is what a genuine broadcast actually
    differs on.
    """
    if not isinstance(src.data, Pointwise) or not isinstance(target.data, Pointwise):
        return None
    src_ranges = list(src.data.ranges)
    target_ranges = list(target.data.ranges)
    src_real_positions = [i for i, s in enumerate(src_ranges) if s != 1]
    src_real_ranges = [src_ranges[i] for i in src_real_positions]
    if len(target_ranges) != len(src_real_ranges) + 1:
        return None
    target_syms = list(iteration_space_from_op(target).keys())
    if len(target_syms) != len(target_ranges):
        return None
    src_strides = list(src.layout.stride)
    for dim in range(len(target_ranges)):
        remaining_ranges = target_ranges[:dim] + target_ranges[dim + 1 :]
        if remaining_ranges != src_real_ranges:
            continue
        if read_dep.index.coeff(target_syms[dim]) != 0:
            continue
        remaining_syms = target_syms[:dim] + target_syms[dim + 1 :]
        if not all(
            read_dep.index.coeff(sym) == src_strides[pos]
            for sym, pos in zip(remaining_syms, src_real_positions)
        ):
            continue
        return _BroadcastAxis(dim=dim, src_positions=tuple(src_real_positions))
    return None


def _reinstated_squeezed_device_dims(
    src: ComputedBuffer, broadcast: "_BroadcastAxis", old_stl: SpyreTensorLayout
) -> "tuple[list[int], list[int]] | None":
    """Device dims of `src`'s own pre-widen `device_layout` whose host
    position was squeezed away by `_detect_broadcast_axis` (i.e. not one of
    `broadcast.src_positions`) -- in every case observed so far, exactly
    one: the gather's own indexed ("which page") coordinate.

    `target`'s own device layout has no slot at all for this coordinate --
    it was squeezed away upstream, before `target`'s own `ranges` were ever
    computed -- so copying `target`'s layout wholesale onto the widened
    `src` (the v18 bug) leaves nothing for `src`'s own `inner_fn` to still
    address indirectly through. This reinstates it by finding `src`'s own
    matching device dim(s) and returning their `(sizes, strides)`, in
    `old_stl`'s existing relative order, ready to prepend verbatim onto a
    widened device layout. Returns None if none are found -- callers must
    treat that as "cannot safely widen", not silently fall back to the
    wholesale copy.
    """
    squeezed_host_strides = {
        int(src.layout.stride[pos])
        for pos in range(len(src.data.ranges))
        if pos not in broadcast.src_positions
    }
    sizes: list[int] = []
    strides: list[int] = []
    # old_stl's own last entry is its stick dim (device_size is the elem
    # count per stick, never 1), so this loop naturally excludes it without
    # a separate stick check.
    for d in range(len(old_stl.stride_map) - 1):
        if (
            old_stl.device_size[d] == 1
            and int(old_stl.stride_map[d]) in squeezed_host_strides
        ):
            sizes.append(old_stl.device_size[d])
            strides.append(old_stl.stride_map[d])
    return (sizes, strides) if sizes else None


def _build_broadcast_widened_body(
    src: ComputedBuffer, target: ComputedBuffer, broadcast: "_BroadcastAxis"
):
    """A `Pointwise` body for `src`, widened to `target`'s own `ranges`,
    whose `inner_fn` drops the broadcast axis's index component, reinserts a
    fixed 0 for each of `src`'s own size-1 positions `target`'s chain
    squeezed away, and delegates to `src`'s own existing body -- no
    recombination needed (unlike a genuine nested split): every value of a
    broadcast axis maps to the exact same source coordinate, so its index
    component is simply discarded rather than folded into another one.
    """
    old_inner_fn = src.data.inner_fn
    old_rank = len(src.data.ranges)
    dim, src_positions = broadcast.dim, broadcast.src_positions

    def inner_fn(
        index, _old=old_inner_fn, _dim=dim, _positions=src_positions, _rank=old_rank
    ):
        index = list(index)
        del index[_dim]
        old_index = [0] * _rank
        for pos, value in zip(_positions, index):
            old_index[pos] = value
        return _old(old_index)

    new_tb = Pointwise.create(
        device=src.get_device(),
        dtype=src.get_dtype(),
        inner_fn=inner_fn,
        ranges=list(target.data.ranges),
    )
    return new_tb.data.data


def _build_nested_clone(src: ComputedBuffer, nested: "_NestedSplit") -> ComputedBuffer:
    """Build (but don't register) a clone of `src` with `nested`'s host dim
    genuinely split in two, via a fresh `Pointwise` whose loader recombines
    the two new indices before reading `src`.
    """
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
    return clone_buf


def _read_dep(consumer: ComputedBuffer, src_name: str):
    return next(
        (d for d in consumer.get_read_writes().reads if d.name == src_name),
        None,
    )


def _commit_matching_division(
    producer: ComputedBuffer,
    consumers: list[ComputedBuffer],
    read_deps: list,
) -> bool:
    """Re-commit `producer`'s own work division to match every one of
    `consumers`' *already-committed* division, in place -- no clone, no new
    buffer. Returns whether a division was actually committed.

    `consumers` must each already carry a trustworthy `iteration_space_
    ownership` -- either because it's a matmul (whose division was set by
    `_distribute_work` from the matmul's own needs), or because it's an
    intermediate this pass just corrected to match a matmul (see
    `_fix_target_division`, `_propagate_to_gather`). Requires `_missing_
    strides(producer, consumer, read_dep)` empty for every one -- no nested
    split or broadcast; this is a division-only fix.

    Independently re-deriving a division (e.g. via
    `commit_iteration_space_ownership`) risks a different core-to-slice
    convention than `consumers` already committed to. So instead, reuse
    each consumer's own per-symbol core assignment directly: a
    `read_dep.index` (a consumer's access to `producer`) and `producer`'s
    own write dep assign coefficients to loop symbols using the same
    physical offsets, so a matching coefficient means the same physical
    axis. When several consumers share `producer`, each must map onto the
    same axis the same way -- any disagreement means nothing is committed,
    since committing one consumer's view could silently misdescribe
    another's read. Symbols nothing maps onto (e.g. the stick/elem axis)
    are left for `commit_tensor_work_division` to fill in as unsplit.

    `producer`'s own `num_cores` is set to the product of *its own*
    committed splits, never copied from a consumer's full physical core
    count: a consumer axis with a zero read-index coefficient into
    `producer` -- a genuine broadcast -- never reaches `work_slices` at all
    (see `_missing_strides`'s docstring), so blindly inheriting the
    consumer's count would claim a core count no combination of `producer`'s
    own splits actually accounts for. That claim is only valid if
    `producer`'s kernel actually replicates the dropped axis across the
    shortfall, which nothing arranges for it to do -- see module docstring.
    """
    name = producer.get_name()
    for consumer, read_dep in zip(consumers, read_deps):
        if _missing_strides(producer, consumer, read_dep):
            return False

    write_dep = next(
        (d for d in producer.get_read_writes().writes if d.name == name),
        None,
    )
    if write_dep is None:
        return False
    producer_syms: dict[int, sympy.Symbol] = {}
    for sym in iteration_space_from_op(producer):
        coeff = write_dep.index.coeff(sym)
        if coeff != 0:
            producer_syms[int(coeff)] = sym

    work_slices: dict[sympy.Symbol, int] = {}
    owners: dict[sympy.Symbol, sympy.Expr] = {}
    consumer_num_cores = None
    for consumer, read_dep in zip(consumers, read_deps):
        ownership = getattr(consumer, "iteration_space_ownership", None)
        if ownership is None:
            continue
        if consumer_num_cores is None:
            consumer_num_cores = ownership.physical_core_count
        elif consumer_num_cores != ownership.physical_core_count:
            return False  # consumers disagree on the physical core domain
        for sym, split in ownership.work_slices.items():
            if split <= 1:
                continue
            coeff = read_dep.index.coeff(sym)
            if coeff == 0:
                continue
            producer_sym = producer_syms.get(int(coeff))
            if producer_sym is None:
                return False  # committed consumer axis has no matching axis
            owner_expr = ownership.core_id_to_work_slice[sym]
            if producer_sym in work_slices:
                if (
                    work_slices[producer_sym] != split
                    or owners[producer_sym] != owner_expr
                ):
                    return False  # consumers disagree on this axis
            else:
                work_slices[producer_sym] = split
                owners[producer_sym] = owner_expr

    if not work_slices:
        return False
    num_cores = math.prod(int(s) for s in work_slices.values())
    commit_tensor_work_division(
        producer, TensorWorkDivision(work_slices, owners, num_cores=num_cores)
    )
    return True


def _fix_target_division(graph: GraphLowering, target: ComputedBuffer) -> bool:
    """Re-commit `target`'s own work division to match its matmul
    consumer(s), in place. `target` is the buffer found by
    `_pre_matmul_buffer` -- the gather itself, or an intermediate (most
    often a restickify) sitting between it and the matmul.

    Only commits when every one of `target`'s readers -- not just the edge
    that found it -- is itself directly a matmul: mutating a shared
    buffer's division based on only some of its readers would silently
    misdescribe the others.

    Before reading each matmul consumer's division, tries `_force_batch_
    division` on it first (best-effort, ignored on failure): overriding the
    matmul to a pure-batch split, when legal, gives `target` a much better
    division to mirror -- one already shaped like K/V's own batch axes,
    needing no reconciliation at all -- than whatever the cost model's
    default (non-batch) split would have produced. TEMP: when
    `_MATMUL_SPLIT_SPEC` is set (see the experiment scaffold above),
    `_force_custom_matmul_division` replaces `_force_batch_division` here
    entirely, forcing that exact division instead.
    """
    consumers = _direct_consumers(graph, target.get_name())
    if not consumers or not all(_is_matmul(c) for c in consumers):
        return False
    for consumer in consumers:
        if _MATMUL_SPLIT_SPEC is not None:
            _force_custom_matmul_division(consumer, _MATMUL_SPLIT_SPEC)
        else:
            _force_batch_division(consumer)
    read_deps = [_read_dep(c, target.get_name()) for c in consumers]
    if any(d is None for d in read_deps):
        return False
    return _commit_matching_division(target, consumers, read_deps)


def _fix_gather_via_clone(
    graph: GraphLowering, operations: list, src: ComputedBuffer
) -> "str | None":
    """When `src` (a gather feeding a matmul directly) can't match its
    matmul consumer(s) via `_fix_target_division` alone because of a
    genuinely nested split (see `_detect_nested_split`), build a clone with
    that nested host shape/device layout, commit its division to match, and
    redirect the consumers to read it instead of `src`. Returns the clone's
    name on success, else None.

    Reserved for exactly the gather-feeds-matmul-directly case (`target is
    src` in `apply_pass`) -- a previous version of this pass tried a
    nested-split clone unconditionally and reverted it: the clone routinely
    won LX placement for itself while competing with, and displacing,
    buffers the matmul needed more, a net loss (see module docstring). This
    is only attempted now, *after* `_fix_target_division` has already tried
    `_force_batch_division` on the matmul (win or lose), because giving the
    matmul a batch-aligned division is what makes the clone's own division
    a clean, honest match -- one that does not itself need reconciling --
    rather than reproducing the same competing-for-LX problem under a
    division that still disagreed with the matmul.
    """
    consumers = _direct_consumers(graph, src.get_name())
    if not consumers or not all(_is_matmul(c) for c in consumers):
        return None
    detected = _detect_nested_split_for_all(src, consumers)
    if detected is None:
        return None
    nested, read_deps = detected

    clone_buf = _build_nested_clone(src, nested)
    clone_buf.name = graph.register_buffer(clone_buf)
    graph.register_operation(clone_buf)
    operations.remove(clone_buf)
    operations.insert(operations.index(src) + 1, clone_buf)
    clone_name = clone_buf.get_name()

    _commit_matching_division(clone_buf, consumers, read_deps)

    for consumer in consumers:
        redirect_computed_buffer_reads(
            consumer,
            {src.get_name(): clone_name},
            operations,
            pass_name="insert_gather_clone",
            reason=(
                "clone gathered matmul operand with a nested host shape "
                "matching the consumer's already-committed (batch-forced) "
                "division"
            ),
        )
    return clone_name


def _fix_broadcast_via_widen(
    graph: GraphLowering, operations: list, src: ComputedBuffer, target: ComputedBuffer
) -> bool:
    """When `target` -- `src`'s sole consumer, already fixed by `_fix_target_
    division` to match its own matmul consumer(s) -- is a pure broadcast of
    `src` (see `_detect_broadcast_axis`), widen `src`'s own body in place to
    `target`'s shape (dropping the broadcast axis's index; no recombination
    needed, unlike a genuine nested split) and redirect `target`'s matmul
    consumer(s) to read `src` directly. No new buffer: `src` keeps its own
    name, so the matmul's read index needs no changes, only the load name.
    `target` becomes dead and is dropped.

    Reserved for exactly this shape: `target` must be `src`'s *only* reader
    (mirrors `_propagate_to_gather`'s own restriction) and every one of
    `target`'s own readers must be a matmul (mirrors `_fix_target_
    division`'s) -- widening `src` in place based on only one of several
    readers of either buffer would silently misdescribe the others. Returns
    whether the widen+redirect actually happened; a failed division commit
    inside it (best-effort, like `_fix_gather_via_clone`'s) doesn't undo it.
    """
    if _direct_consumers(graph, src.get_name()) != [target]:
        return False
    consumers = _direct_consumers(graph, target.get_name())
    if not consumers or not all(_is_matmul(c) for c in consumers):
        return False
    target_read_dep = _read_dep(target, src.get_name())
    if target_read_dep is None:
        return False
    broadcast = _detect_broadcast_axis(src, target, target_read_dep)
    if broadcast is None:
        return False
    consumer_read_deps = [_read_dep(c, target.get_name()) for c in consumers]
    if any(d is None for d in consumer_read_deps):
        return False

    # `target`'s own device layout has no slot for `src`'s indexed ("which
    # page") coordinate -- it was squeezed away upstream, before `target`'s
    # own `ranges` were computed. Capture `src`'s pre-widen layout (still
    # `new_buf.layout.device_layout` after `replace_computed_buffer_body`,
    # which builds the new ComputedBuffer with `layout=op.layout`, the same
    # object -- not a copy) so the reinstated slot below is read before
    # anything overwrites it.
    new_data = _build_broadcast_widened_body(src, target, broadcast)
    new_buf = replace_computed_buffer_body(
        src,
        new_data,
        operations,
        pass_name="insert_gather_clone",
        reason="widen gather in place to absorb its sole broadcast consumer",
    )
    old_stl = new_buf.layout.device_layout
    reinstated = _reinstated_squeezed_device_dims(src, broadcast, old_stl)
    if reinstated is None:
        # Can't prove a safe slot exists for the indexed coordinate --
        # bail out rather than reproduce the v18 wrong-value bug (see
        # module docstring / plan file for the full root-cause writeup).
        return False
    reinstated_sizes, reinstated_strides = reinstated

    target_layout = target.layout
    target_stl = target_layout.device_layout
    new_buf.layout.size = list(target_layout.size)
    new_buf.layout.stride = list(target_layout.stride)
    # Reinstated slot(s) first (device position 0 -- nothing downstream
    # re-rotates this for us, see plan file), then `target`'s own dims,
    # keeping its stick dim last.
    new_buf.layout.device_layout = SpyreTensorLayout(
        device_size=reinstated_sizes + list(target_stl.device_size),
        stride_map=reinstated_strides + list(target_stl.stride_map),
        device_dtype=target_stl.device_dtype,
    )
    new_buf.layout.offset = target_layout.offset
    # `get_free_symbol_uses` is cache_on_self on both Layout and
    # ComputedBuffer; a result cached from before this mutation would
    # silently keep describing the pre-widen shape (see `wsr/coarse_tile.py`'s
    # own in-place device-layout mutation for the same precaution).
    _clear_cache(new_buf.layout, _LAYOUT_FREE_SYMS_KEY)
    _clear_cache(new_buf, _COMPUTED_BUF_FREE_SYMS_KEY)

    _commit_matching_division(new_buf, consumers, consumer_read_deps)

    for consumer in consumers:
        redirect_computed_buffer_reads(
            consumer,
            {target.get_name(): new_buf.get_name()},
            operations,
            pass_name="insert_gather_clone",
            reason="read the widened gather directly now that it absorbs the broadcast",
        )
    if target in operations:
        operations.remove(target)
    return True


def _propagate_to_gather(
    graph: GraphLowering, src: ComputedBuffer, target: ComputedBuffer
) -> bool:
    """Re-commit `src`'s (the gather's) own work division to match
    `target`'s -- already corrected by `_fix_target_division` to match the
    matmul -- in place.

    Fixing only `target` (e.g. a restickify) leaves `src` itself divided
    however `_distribute_work` originally, independently chose -- possibly
    still disagreeing with what `target` now needs from it, which can by
    itself force `target`'s own kernel into an unfavorably small core count
    to reconcile the gap (see module docstring). Mirroring `target`'s own,
    now-trustworthy ownership onto `src` closes that gap too.

    Only commits when `target` is `src`'s *only* reader: with more than
    one, mutating `src`'s division to match just this one would silently
    misdescribe the others (mirrors `_fix_target_division`'s own
    restriction, one hop earlier).
    """
    if _direct_consumers(graph, src.get_name()) != [target]:
        return False
    read_dep = _read_dep(target, src.get_name())
    if read_dep is None:
        return False
    return _commit_matching_division(src, [target], [read_dep])


class InsertGatherClonePass(ScratchpadOptimizationPass):
    """Re-commit a gathered matmul operand's (and, where one sits in
    between, an intermediate's) own work division to match its matmul
    consumer, in place. Registered as a ``pre_optimization_pass`` in
    ``select_allocator()`` -- see the module docstring for why this
    position, not an earlier pass, is load-bearing.
    """

    def apply_pass(self, graph: GraphLowering) -> None:
        # TEMP: TORCH_SPYRE_MATMUL_SPLIT_ONLY=1 (see _force_matmul_splits_
        # only above) -- force the matmul's division and stop, skipping
        # every other mechanism in this pass entirely.
        if _MATMUL_SPLIT_ONLY:
            _force_matmul_splits_only(graph)
            return
        operations = graph.operations
        for src in list(operations):
            if not isinstance(src, ComputedBuffer):
                continue
            if not (_origin_targets(src) & _GATHER_TARGETS):
                continue

            src_name = src.get_name()
            seen: set[str] = set()
            for direct in _direct_consumers(graph, src_name):
                # If this edge's direct consumer is already the matmul, the
                # buffer adjacent to it is `src` itself (zero hops) --
                # `_pre_matmul_buffer` starting *at* an already-matmul buffer
                # would instead ask whether that matmul's own output feeds
                # another matmul, which is a different question.
                target = (
                    src if _is_matmul(direct) else _pre_matmul_buffer(graph, direct)
                )
                if target is None or target.get_name() in seen:
                    continue
                seen.add(target.get_name())
                if not _fix_target_division(graph, target):
                    # _fix_target_division already tried to force a
                    # batch-aligned division onto the matmul consumer(s)
                    # (win or lose) -- if src itself is the target (no
                    # intermediate) and it still didn't match, the gap may
                    # be a genuine nested split that only a clone can give
                    # it, now that a batch-aligned division is worth
                    # cloning towards. See `_fix_gather_via_clone`. TEMP:
                    # skipped entirely during a `_MATMUL_SPLIT_SPEC`
                    # experiment, so Q falls back to its natural
                    # gather+shuffle path no matter what division got
                    # forced onto the matmul -- see the experiment
                    # scaffold above.
                    if target is src and _MATMUL_SPLIT_SPEC is None:
                        clone_name = _fix_gather_via_clone(graph, operations, src)
                        if clone_name is not None:
                            logger.info(
                                "insert_gather_clone: cloned gather %s -> %s "
                                "with a nested split matching its matmul "
                                "consumer(s)",
                                src_name,
                                clone_name,
                            )
                    continue
                logger.info(
                    "insert_gather_clone: re-committed %s's own division to "
                    "match its matmul consumer(s) directly (source gather "
                    "%s)",
                    target.get_name(),
                    src_name,
                )
                if target is src:
                    continue
                # TEMP: TORCH_SPYRE_ENABLE_BROADCAST_WIDEN=1 (see
                # `_ENABLE_BROADCAST_WIDEN` above). Was disabled here
                # pending a correctness fix to `_fix_broadcast_via_widen`'s
                # device-layout construction (v18's wrong-value bug); now
                # tried first, before `_propagate_to_gather`, since a
                # successful widen makes `target` dead (removed) and there
                # is nothing left to propagate onto.
                if _ENABLE_BROADCAST_WIDEN and _fix_broadcast_via_widen(
                    graph, operations, src, target
                ):
                    logger.info(
                        "insert_gather_clone: widened gather %s in place to "
                        "absorb its sole broadcast consumer %s",
                        src_name,
                        target.get_name(),
                    )
                    continue
                if _propagate_to_gather(graph, src, target):
                    logger.info(
                        "insert_gather_clone: propagated %s's division back "
                        "onto gather %s",
                        target.get_name(),
                        src_name,
                    )
