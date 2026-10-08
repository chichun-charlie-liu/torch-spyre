# The co-optimizer

:::{admonition} Draft
:class: warning

This is a first draft assembled from `sa_co_optimization.md`, `scratchpad_planning.md`,
`work_division_planning.md`, and the current state of
`torch_spyre/_inductor/scratchpad/`. It is meant as a jumping-off point for review, not a
finished page — see the open questions in [Status of coarse tiling](#status-of-coarse-tiling)
before relying on any claim there.
:::

This page is the map of **why** torch-spyre co-optimizes several compiler decisions together
instead of solving each one separately, **how** that joint search works in general, and
**which** decisions it currently covers. For the full detail on any one piece, follow the links
into [Scratchpad Planning](scratchpad_planning.md), [Work Division Planning](work_division_planning.md),
and [Joint core-division + LX placement](sa_co_optimization.md).

**Quick navigation:**

- [Why a co-optimizer](#why-a-co-optimizer)
- [How it works in general](#how-it-works-in-general)
- [What gets co-optimized](#what-gets-co-optimized)
  - [Scratchpad placement](#a-scratchpad-placement)
  - [Work division](#b-work-division)
  - [Coarse tiling](#c-coarse-tiling)
- [Cost model](#cost-model)
- [Status of coarse tiling](#status-of-coarse-tiling)
- [Related documents](#related-documents)

## Why a co-optimizer

In the `torch-spyre` compiler pipeline, work division runs before LX planning, and its
objective is simply to pick the best parallelization scheme for each op *individually*. But
consider a simplified program with two memory-bound pointwise ops in a chain: the key
performance optimization is making sure op1's output stays on each core's own scratchpad for
op2 to read directly, rather than round-tripping through HBM. Because scratchpad is private per
core, the data core 0 holds in its scratchpad after op1 must be exactly what core 0 needs for
op2. If work division splits op1 along rows and op2 along columns, the data op2 needs on a
given core was never produced on that core — so op1's output has to spill to HBM, and every
core processing op2 pays the slower HBM fetch to get its share.

Solving work division and LX placement one at a time, freezing the earlier choice before the
later pass runs, leaves exactly this interaction on the table: either compute utilization is
sacrificed for memory locality, or memory locality is sacrificed and HBM traffic balloons. In
either case, joint assignments that *reconcile* several ops onto one shared division — one
that keeps everyone's compute shape good **and** keeps the shared buffer resident — are missed
simply because the earlier pass already committed. The co-optimizer exists to search that
combined space instead of two disjoint ones.

## How it works in general

In a nutshell, the co-optimizer uses a cost model to estimate two things per candidate: the
runtime of matmul compute (the only op type with its own compute-roofline term; a dependent
fused reduction riding in the same bundle as a matmul gets a smaller additive correction, and
every other op is treated as pure memory traffic) and the data-transfer time implied by where
each buffer lives (HBM or LX), using known or measured bandwidth for the target hardware.
Rather than committing to one work division per op up front, it evaluates several combinations
together and prices the communication cost that each one implies, then commits only the
winning combination at the end of the search. A few different solver backends can drive this
search; the sections below cover each in detail, and [Cost model](#cost-model) covers the
model itself — how it plugs into the solvers, how to inspect it, and how to recalibrate it.

The co-optimizer is reached through `CoOptimizingAllocator`
(`torch_spyre/_inductor/scratchpad/allocator.py`), gated by `config.co_optimizing_lx_planning`
(env `CO_OPTIMIZING_LX_PLANNING`, on by default) and selected via `config.layout_solver`. It
runs as a **pre-scheduling pass** — `V.graph` is live but `V.graph.scheduler` is still `None`,
so fusion hasn't happened yet and anything the engine needs to know about the kernels its
decisions will land in has to be *estimated* from the flat, ordered operation list rather than
read off the real fused grouping.

### The simplest version: exhaustive search over greedy placement

Before getting into the production solvers, it's worth seeing the simplest possible instance
of "co-optimize divisions and placement jointly," because it makes the shape of the problem
concrete without any annealing or constraint-solving machinery to explain first.

`ExhaustiveSearchSolver` (`torch_spyre/_inductor/scratchpad/exhaustive_search.py`) wraps a
plain, placement-only solver — most simply `GreedyLayoutSolver`, which otherwise only ever
places buffers against divisions some earlier pass already fixed — and brute-forces the other
half of the problem around it. For every buffer that has more than one candidate division, it
tries every combination (an exhaustive DFS, bounded by `K^N` leaves over the `N` buffers that
actually have a choice), and at each leaf:

1. Builds a fresh set of per-core buffer sizes implied by that combination of chosen divisions.
2. Hands them to a brand-new instance of the wrapped placement solver (a solver is single-use)
   and calls `plan_layout()`.
3. Scores the leaf by summing the total size of every buffer the inner solver could *not* pin
   to LX — the same differential-HBM-traffic idea the production solvers' memory-only
   fallback uses.

The combination with the lowest score wins; its divisions are committed to every buffer, and
the wrapped solver runs once more to produce the final addresses.
`TestExhaustiveSearchResidency.test_mismatched_consumer_spills_the_producer`
(`tests/inductor/test_scratchpad_solver.py`) wires this up end to end — a three-buffer
producer→consumer→sink chain, run through `ExhaustiveSearchSolver([producer, consumer, sink],
inner_factory=GreedyLayoutSolver, ...)` — and shows the piece this page cares about: the
`core_div_mismatch` check this solver runs is exactly the same `cd_parent_matches` relation
the production solvers use to decide which combinations even make a buffer eligible for LX,
not a check bolted on separately per solver. (In this particular fixture every buffer carries
only one division candidate, so the exhaustive search has nothing left to search over — the
test isolates the residency check alone — but any buffer with more than one candidate is
exactly what the DFS loops over.) The real version of the search this page is building toward
is the one in [Why a co-optimizer](#why-a-co-optimizer): several buffers each with a few
divisions to choose from, where only some combinations avoid a mismatch.

This also shows why it's not what ships by default: scoring a leaf means re-running the real
placement solver from scratch, so the cost is exponential in the number of buffers with a
real choice. `select_allocator()` only reaches for `ExhaustiveSearchSolver` as a fallback —
when `co_optimizing_lx_planning` is set but the configured `layout_solver` has no
purpose-built core-division-capable solver to co-optimize with (i.e. anything other than
`"cpsat"` with `ortools` installed, or `"simulated_annealing"`) — and only when the caller has
opted in via `config.allow_exhaustive_search` (env `ALLOW_EXHAUSTIVE_SEARCH`); otherwise it
raises rather than silently paying that cost. The two solvers below exist to make the general
case affordable.

### Pruning the exhaustive search: `_enum_split_options`

`select_allocator()` always pairs the `ExhaustiveSearchSolver` fallback above with
`CoOptimizingAllocator(prune=True)`, which swaps the division menu it DFS-searches: instead of
`enumerate_work_division_candidates()`'s full, standalone-work-division-planner cross product,
`_enum_split_options` (`allocator.py`) builds a much smaller, heuristic candidate list per op,
dispatching on op type. CP-SAT and SA never set `prune=True` — this path exists specifically to
keep the exhaustive DFS's exponential cost affordable, not as a third solving strategy
alongside them.

:::{figure} ../_static/images/lx/co-optimization.svg
:alt: Co-optimization searches over alternative split assignments, scoring each by HBM bytes left unpinned
:width: 700px
:align: center

The pruned search enumerates split variants per op, scores each combination by counting HBM
bytes the solver could not pin, and commits the winning assignment back before the standard
allocator flow.
:::

Generated alternatives are deduped by canonical key and filtered through `_split_fits_sticks`,
which rejects factors that overflow a stickified dim's stick count (those would abort the
SuperDSC bundler) or that land on a collapsed/broadcast dim. The upstream seed is always kept
and is already stick-valid from work division; every other candidate must still satisfy hard
work-division constraints (blocked axes stay unsplit, split domains restrict legal factors).

- **Pointwise ops** get their seed, dim-flip variants (move the seed's single output-dim
  factor onto each compatible alternative output dim, bounded by `DEFAULT_VARIANT_CAP = 6`),
  and the matmul tilings from the shared pool (below). Adopting a neighbouring matmul's tiling
  makes the op's per-core view match the matmul's, so a shared buffer pins to LX *and* the op
  runs at the matmul's high-utilization shape.
- **Matmul splits are not overridden onto a single dim, but neighbours' tilings and a
  batch-major split are offered.** Concentrating a balanced `M/4×N/8` split onto one dim
  (`M/32`) pins the matmul output and the surrounding chain to LX but is a poor matmul shape:
  on `mlp-linear-kn.t` (`SENCORES=32`) it regressed kernel time ~2.5× as process-engine
  utilization fell from 66% to 33%. So the rule remains **prioritize compute utilization for
  compute-bound ops**: the seed split is never flipped onto one dim. Instead,
  `_check_and_add_matmul_option` offers each matmul its seed plus (a) every *other* matmul's
  split transferred into this op's coordinates by axis role (so two matmuls whose
  work-division splits disagree can find a consistent assignment), and (b) a factored
  batch-major `B/M` split. All of these are full-core splits, so compute utilization is
  preserved.
- **Batch-major `B/M` tiling reconciles attention.** Two attention matmuls (`Q·Kᵀ` and
  `scores·V`) contract different axes, so neither can adopt the other's `N`/`K` tiling, but
  both keep the batch (`B`) and `M` output axes. `_factored_bm_splits` emits a single full-core
  `B/b · M/m` split (largest batch factor that fits, from `(8, 4, 2)` with `m = ncores / b`),
  valid for both matmuls and divisible into both stick-count extents. This shared tiling is
  also offered to the **softmax reductions** (`max`/`sum`) in their own output coordinates via
  `_reduction_bm_axes`. Reductions are otherwise left on their seed, but offering them the
  `B/M` split lets the whole softmax chain between the two matmuls reconcile to one tiling. On
  `mha_4h` (`SENCORES=32`) this converges both matmuls and the entire softmax chain on
  `B/4·M/8`, pinning the scores matrix and the chain to LX. Reductions are not given dim-flip
  variants (their reduced axis is fixed), and any candidate that fails to reconcile a shared
  buffer's per-core view self-eliminates during scoring.

The shared matmul-tiling pool is collected once by `_find_distinct_matmul_splits`: each
distinct matmul seed split plus each matmul's factored `B/M` split, deduped. This pool seeds
both the pointwise candidate lists and the cross-matmul transfer.

On `mlp-linear-kn.t` (`SENCORES=32`) the pointwise-seeding path lifted process-engine
utilization from ~66% to ~79% and cut fused kernel time by ~17% (about 2× faster than the
sendnn reference).

The leaf-scoring function itself is the same one described above: it runs the full
`_generate_buffers + plan_layout` pass on the candidate splits and counts the HBM bytes of
every buffer the solver could not pin. Repeated `_per_core_view_on_buf` work is memoized
across leaves, and the split-invariant liveness / filtered-op-view / mem-usage computations
are hoisted out of the per-leaf path.

The factored-`B/M` and cross-matmul transfer options are marked TEMP/TODO: the intent is for
work division to assign consistent splits directly, at which point these compensating options
can be removed. See [Co-optimization is still limited](scratchpad_planning.md#co-optimization-is-still-limited)
for the remaining gaps in this path.

### The production solvers

Two solver backends implement the joint search at scale, differing in paradigm but sharing
the same inputs and the same cost model:

- **`CpSatLayoutSolver`** (`config.layout_solver = "cpsat"`, the default) models the problem as
  one global constraint-satisfaction model via Google OR-Tools CP-SAT. Core-division candidates
  (and, where landed, tiling candidates) are decision variables, placement is a 2D no-overlap
  packing — each resident buffer an optional `[lifetime] × [address, address+size)` rectangle —
  and producer/consumer slicing-match constraints tie divisions together across edges. CP-SAT
  searches exactly, minimizing predicted HBM traffic.
- **`SaCoOptimizingSolver`** (`config.layout_solver = "simulated_annealing"` with
  `co_optimizing_lx_planning`) anneals a single joint state heuristically. The state is the pair
  `(pi, W)`: the LX layout permutation `pi` (held in a composed `PermutationBasedLayoutSolver`)
  and the division vector `W`, one candidate-menu index per buffer. It seeds every buffer at
  menu index 0 with a FirstFit placement, then runs one geometric cool for
  `clamp(40n, 200, 15000)` steps using three move types — **reorder** (re-place a buffer in `pi`,
  weight 0.5), **flip** (change one buffer's division, weight 0.3), and **recolor** (flood a
  coordinated division change across a connected region via the `cd_parent_matches`
  compatibility relation, weight 0.2) — and returns the best state seen, so the result is never
  worse than the seed. See [Joint core-division + LX placement](sa_co_optimization.md) for the
  full mechanics.

**What "co-optimizing" means concretely**: both solvers are driven by a *single* cost
expression (`cost_expr`) built once, up front, from the whole graph's predicted runtime —
`predict_by_bundle(graph.operations, op_features, ...)`, a symbolic expression over every
buffer's `sym_is_lx` (is it LX-resident) and `sym_core_divs` (which division was chosen).
Because both kinds of decision are symbols in the *same* expression, a move that changes one
automatically re-prices the other — there's no separate objective per variable to reconcile
after the fact. That's what distinguishes this from running two independent passes in
sequence. When no cost expression is usable (symbolic shapes, serialized captures, or a
construction error), both solvers fall back to a cheaper **memory-only** objective: pure
differential HBM traffic, where a resident buffer costs zero and only spills are summed. This
fallback is weaker — on a graph where everything already fits, every division scores
identically and the search has nothing left to optimize on.

Without `co_optimizing_lx_planning`, each solver can still be selected on its own, but then it
only *places* buffers against divisions some earlier pass already fixed — it never searches
divisions itself. The joint behavior is what `CoOptimizingAllocator` adds on top.

## What gets co-optimized

Three variables are, or are becoming, part of the joint search:

### a) Scratchpad placement

**What it is**: where each buffer's resident copy lives in LX — represented as
`address: Optional[int]` on `LifetimeBoundBuffer`; `None` after solving means the buffer
spilled to HBM. Eligibility is decided once, declaratively, in
`ScratchpadAllocator._residency_reasons`, and carried to every solver as a single
`residency_reason` field (`None` = may be pinned, any string = may not) — see
[Declarative exclusion](scratchpad_planning.md#declarative-exclusion). The actual placement
search (which address, in what order) is `pi` in the SA solver and the 2D no-overlap packing
in CP-SAT.

**Status**: implemented, and the variable the whole scratchpad-planning pass exists to solve.
See [Scratchpad Planning](scratchpad_planning.md) for the full allocator architecture, and note
that the standalone, placement-only `SimulatedAnnealingLayoutSolver`
(see [Simulated Annealing Layout Planner](simulated_annealing_layout.md)) is a *different*
class from the joint `SaCoOptimizingSolver` — it only ever moves `pi`, never the division
vector `W`.

### b) Work division

**What it is**: how many cores an op's output or reduction uses, and which slice of the
iteration space each core owns — one split factor per iteration-space symbol. Splitting an
**output** dim gives each core a disjoint output slice; splitting a **reduction** dim gives
each core a partial result that must be combined. See
[Work Division Planning](work_division_planning.md) for the standalone 3-pass planner
(span-reduction, cost-model matmul split, default distribution) that picks this when the
co-optimizer isn't driving it.

**As a co-optimized variable**: the joint solvers don't re-derive divisions; they search over
the *same* candidate space the standalone planner would offer, pre-enumerated per op by
`enumerate_work_division_candidates()` (`work_division.py`) and attached to each buffer as a
menu of `CoreDivision` candidates (`CoreDivisionBuffer.core_divisions`,
`plan_solver.py`). In the SA solver this menu index is exactly the `W` half of the `(pi, W)`
state; the **flip** move changes one buffer's chosen index and resizes its per-core footprint,
and **recolor** propagates a compatible change across a connected region. In CP-SAT, the menu
becomes a decision variable alongside placement in the same constraint model.

**Status**: implemented and tested — e.g.
`test_a_flip_that_shrinks_a_footprint_into_capacity_raises_the_count` exercises exactly the
scenario the joint search exists for: a buffer that doesn't fit undivided becomes eligible
once a flip halves its footprint. The cost expression also prices division choice directly —
`test_core_division_symbol_drives_the_score` shows the predicted score scaling with the number
of cores a division uses, confirming it feeds the runtime term, not just a memory-fit
heuristic.

### c) Coarse tiling

**What it is**: cutting an op's iteration space into sequential tile runs — a `TileSpec` of
`TileAxis` entries (output and/or reduction axes, each with a count) — so a working set that
doesn't fit LX at full size can fit one tile at a time. It's a different lever from work
division: division splits the iteration space *across cores at the same time*; coarse tiling
splits it *across time, on the same core(s)*. The underlying loop-IR mechanism — how a tiled
run is represented and survives scheduling and codegen — is implemented and documented in
[Coarse-Tiling Loop IR for the Spyre Backend](coarse_tiling_loops.md); that document also
points to the design RFC. What's described here is specifically the *solver's* ability to
choose a coarse tiling as part of the joint search, which is a separate, newer effort.

**Status: in progress, in both solvers, not yet on `main`.** The apply-side plumbing is
landed — `scratchpad/coarse_tiling.py` takes a `TileSpec` and lowers it to `DimHint`s, and
`CoreDivision.tiling` / `min_footprint` in `plan_solver.py` already account for a tiling if one
is present. But automatic, solver-chosen coarse tiling (as opposed to tiling driven by explicit
`for_each_tile`/hint-based code) is still landing:

- **CP-SAT**: a stacked pair of PRs (`pr-4767`, prediction; `pr-4768`, integration) adds a
  `predict_frame` dry-run (`wsr/tile_prediction.py`) that computes what a candidate tiling
  would produce without mutating IR, and extends `ilp_solver_ortools.py` with a 5-stage
  lexicographic objective (residency, then cut count, then parallelism, then division shape,
  then tile count) once tiling candidates are in play. Neither file exists in this form on
  `main` yet.
- **Simulated annealing**: a 7-PR stack (`pr-4456` through `pr-4894`) extends
  `SaCoOptimizingSolver`'s division representation to carry a tiling choice and extends the
  **recolor** move to propose tiling changes, with companion-buffer copy-out costs (for data
  that escapes a tile run) priced into the cost model along the way. `main`'s
  `sa_cooptimizer.py` does not yet carry this.

Both tracks gate the same join: a buffer's `min_footprint` already has a `/ tile_count` term
waiting to be driven by a real search instead of staying at the untiled default, and both
tracks share a real file-level conflict surface (`plan_solver.py`, `allocator.py`,
`config.py`, the `wsr/` tiling helpers) that whichever lands second will need to rebase
through non-mechanically.

This matches what the existing docs say about the gap:
`scratchpad_planning.md`'s "Current limitations" lists "No coarse-tiling integration" and its
"Future work" lists "Joint operation with the `coarse_tiling` pass" — both still accurate as
of this draft.

## Cost model

Both production solvers share one cost model (`torch_spyre/_inductor/cost_model.py`) — the
same `predict_by_bundle` / `cost_expr` referenced throughout the sections above. This section
covers the model as its own piece: how it is wired into each solver, how to see what it
predicted for a real compile, and how its constants should be kept honest against real
hardware.

### How it's integrated: symbolic (CP-SAT) vs. compiled-and-evaluated (SA)

The two production solvers consume the *same* `cost_expr` — one sympy expression built once,
up front, over every buffer's `sym_is_lx` and `sym_core_divs` symbols — but bind it to a chosen
candidate in two different ways:

- **CP-SAT (symbolic all the way through)**: `_minimize_cost_expr`
  (`scratchpad/ilp_solver_ortools.py`) maps each sympy symbol straight onto a native OR-Tools
  decision variable —

  ```python
  sym_map[t.buffer.sym_is_lx.name] = t.in_buffer
  for key, symbol in t.buffer.sym_core_divs.items():
      sym_map[symbol.name] = t.cp_core_divs[key]
  ...
  cp_cost = _SympyExprToCpSat(model, sym_map, buffer_map).convert(cost_expr)
  ```

  — then hands the *whole expression*, still unevaluated, to the constraint model as the
  objective to minimize. CP-SAT never evaluates `cost_expr` in Python; it searches the
  symbolic constraint space directly.

- **Simulated annealing (compiled, then evaluated numerically per move)**: `_build_score_fn`
  (`scratchpad/sa_cooptimizer.py`) instead resolves every symbol to a plain Python callable of
  `(chosen, resident)` —

  ```python
  value_of[buf.sym_is_lx] = lambda chosen, resident, name=buf.name: (
      1 if name in resident else 0
  )
  ...
  fn = sympy.lambdify(free, cost_expr, modules="math")
  ```

  — compiles `cost_expr` once via `sympy.lambdify` into a fast numeric function, and calls that
  function fresh for every candidate state the annealer visits. Each move gets a real number
  back, not a symbolic rewrite.

Same model, same expression, two different consumption strategies: one treats it as a
constraint to satisfy exactly, the other as a function to sample cheaply, many times, during a
heuristic search.

### Inspecting a prediction: `SPYRE_DUMP_COST`

`SPYRE_DUMP_COST=1` (see `dump_cost_model.py`) prints a human-readable breakdown of the cost
model's prediction for every bundle in a real compile — the same `explain()` function used
throughout this page's examples. For a single `128×512 @ 512×256` matmul on 32 cores:

```
  mm           read=393216B write=65536B lx=0B
      output op0       torch [128, 256] -> device [4, 128, 64] in HBM | 32768 elems x 2B = 65536 B (hbm counted: 65536 B) graph boundary (charged despite LX)
      input  arg0_1     torch [128, 512] -> device [8, 128, 64] in HBM | 65536 elems x 2B = 131072 B (hbm counted: 131072 B) x4 consumers broadcast (loaded once) graph boundary (clone-in charged here if LX)
      input  arg1_1     torch [512, 256] -> device [4, 512, 64] in HBM | 131072 elems x 2B = 262144 B (hbm counted: 262144 B) x8 consumers broadcast (loaded once) graph boundary (clone-in charged here if LX)
  -- prediction (turnaround, bundled matmul model): T = compute + (R+W)/BW_PEAK + a*min(R,W) --
     R=393216B (read)   W=65536B (write)
     compute = MACs/cores/(mac_peak*pt_eff) = 16777216/32/(1140*0.616) = 0.75 us  (M/m=16, pt_eff=0.616)
     base = R/150 + W/150 = 3.06 us
     turn = a*min(R,W) = 0.00574*65536 = 0.38 us
     => T_model = 3.43 us
```

`SPYRE_DUMP_COST_EXPR_FILE=<path>` is the companion machine-readable dump: one JSON line per
solve (`cost_expr_record` in `scratchpad/plan_solver.py`), recording the objective's per-bundle
terms as lossless `sympy.srepr` strings, the symbol bindings the solver actually chose, every
term evaluated under those bindings, and the alternative divisions each buffer could have taken
instead — a self-describing record of one solve, meant to be read without needing to know which
flags produced it.

### Keeping the model honest: recalibrating against real hardware

*(placeholder — to fill in)*

## Status of coarse tiling

Open items from drafting this page that are worth resolving before calling it done:

- Confirm with whoever is driving the CP-SAT and SA coarse-tiling branches which PR stack is
  expected to land first, and whether this page should wait for that merge or describe the
  in-flight design now (as it does today) and get updated after.
- The existing docs *and the current code* use "tiling" for at least two different things —
  `CoreDivision.tiling` (coarse tiling proper, a `TileSpec`), and, separately, (a)
  `scratchpad_planning.md`'s co-optimization section calling core-division splits for matmuls
  "the matmuls' tilings," and (b) `sa_cooptimizer.py`'s own `_flood_region(anchor, tiling)` and
  `_rng.choice(self._nontrivial_menu[anchor])`, where `tiling` names a **division-menu index**,
  not a `TileSpec` — confirmed on `upstream/main` as of this draft (`sa_cooptimizer.py:519-581`).
  So the naming collision isn't just a doc-wording slip; it's in a parameter name in the
  pre-coarse-tiling code the SA stack is extending. Worth a terminology pass across the
  documents *and* that code once coarse tiling lands, so "division"/"split" and "tiling" stay
  consistently distinct.

## Related documents

- [Scratchpad Planning](scratchpad_planning.md) — the allocator architecture, the solvers, LX
  eligibility rules, and the memory-hierarchy background
- [Work Division Planning](work_division_planning.md) — the standalone work-division planner
  and the candidate space the co-optimizer searches
- [Joint core-division + LX placement](sa_co_optimization.md) — the SA co-optimizer's search,
  objective, and test fixtures in full detail
- [Simulated Annealing Layout Planner](simulated_annealing_layout.md) — the placement-only
  annealer, a different class from the joint SA co-optimizer
- [Coarse-Tiling Loop IR for the Spyre Backend](coarse_tiling_loops.md) — how a tiled run is
  represented and survives scheduling and codegen, independent of who chooses the tiling
