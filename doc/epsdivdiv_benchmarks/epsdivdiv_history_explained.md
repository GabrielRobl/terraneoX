# EpsDivDiv Optimization History — Explained

The story has a clear arc: **three distinct eras**, each hitting a different bottleneck ceiling,
with one massive structural breakthrough in the middle.

---

## Era 1: CPU-style code on a GPU (v00a–v02) — 0.012 → 0.107 Gdof/s

The first three versions treat the GPU essentially like a fast CPU. Each thread independently
computes one hex cell with no cooperation between threads.

**v00a** is the textbook implementation: assemble the full 18×18 local stiffness matrix `A`,
then compute `A·src`. Storing two such matrices in registers (648 doubles = 5184 bytes per
thread) is absurd on a GPU — the compiler spills everything to slow local memory. 255
registers/thread confirms this is maxed out.

**v00b** eliminates the matrix entirely by fusing assembly and application in one pass — a
clean algorithmic insight that drops the register footprint dramatically in principle, though
the GPU still spills at 255 regs.

**v01** switches to code-generated scalar arithmetic (no `dense::Mat<3,3>` abstractions),
giving the compiler direct register visibility. Still 255 regs — the O(3×3) nested loops over
velocity components are keeping everything live simultaneously.

**v02** splits the nested `dimi×dimj` loops into a gather pass (over `dimj`) then a scatter
pass (over `dimi`), reducing complexity from O(9) to O(6). A genuine FLOP reduction — but
still 255 regs and MDRangePolicy. The kernel remains DRAM-bound.

The hardware utilization chart makes the problem clear: 53% DRAM utilization but only 6% SM
compute in v02. The kernel is spending most of its time waiting for memory.

---

## Era 2: The Architectural Breakthrough (v03–v07) — 1.7 → 4.9 Gdof/s

**V03 is the single biggest jump** — 0.1 → 1.7 Gdof/s, a 16× improvement in one commit.
Two things happen simultaneously:

1. **TeamPolicy replaces MDRangePolicy**: threads are grouped into cooperative blocks that share
   on-chip scratch memory. Surface coordinates (which are identical for all cells in a radial
   column) are loaded once by one thread and reused by all others in the team. This eliminates
   redundant global memory reads.

2. **6-point quadrature collapses to 1 point**: the Gauss points at the centroid have
   compile-time constant shape function gradients — a 6× reduction in per-element arithmetic.
   The trade-off (reduced integration accuracy) is acceptable for iterative solvers that only
   need spectral equivalence.

The DRAM utilization drops from 53% to 14% — shared memory is doing its job.

**V04–V05** extend the shared memory pattern: first to the source vector, then to the
viscosity coefficient. Neighboring cells share nodes, so loading the full column into shmem
cuts global reads per unique value from ~36N to ~4N. Throughput reaches 5.4 Gdof/s.

**V06 (xy tiling)** is where it gets interesting — it is a **regression** to 4.9 Gdof/s. The
idea is sound (tiling laterally shares edge nodes too), but the thread mapping `tx = tid%lat_tile`
makes consecutive threads access different x-positions, which are not contiguous in memory. The
tiling creates shared memory opportunity but breaks memory coalescing. This sets up V08.

**V07** is organizational — host-side dispatch separates the fast (Dirichlet/Neumann) and slow
(free-slip, stored matrices) code paths so the fast path has zero runtime branches. No real
throughput change, but essential architectural cleanup.

---

## Era 3: Squeezing the GPU (v08–current) — 6.4 → ~7.8 Gdof/s

By V07 the kernel is L2-bound and SM compute is only at 39%. The remaining gains come from
register pressure reduction and better occupancy — getting more blocks onto each SM so the GPU
can hide latency.

**V08** fixes the coalescing problem from V06: `r` (radial index) becomes the fastest-varying
thread dimension. Consecutive threads now access consecutive radial cells, which are contiguous
in memory. Throughput jumps to 6.4 Gdof/s. L2 utilization hits 83% and stays there from now on.

**V09** introduces scoped gather/scatter phases: the Jacobian inverse is computed in Phase 1,
used for the gather, then the scope closes (freeing those registers), and J is recomputed cheaply
in Phase 2 for the scatter. Trading ~30 FLOPs for ~30 registers turns out to be a good deal.
With `LaunchBounds<128,5>` telling the compiler to cap registers and target 5 blocks/SM,
occupancy improves. Throughput: 7.7 Gdof/s. SM compute hits 63%.

**V10** introduces `r_passes=2`: each thread processes 2 radial cells sequentially instead of 1.
This halves the team size from 256 to 128 threads, which gives the compiler more register budget
per thread (or equivalently, more blocks per SM). The shared memory slab covers both passes, so
the load cost is amortized. Throughput: 7.8 Gdof/s.

**Current** (hybrid invJ, cross-product J, 96 regs) refines the Jacobian formulation to avoid
redundant computation, but lands at essentially the same ~7.8 Gdof/s. The roofline position has
not changed — it is still sitting on the L2 bandwidth line.

---

## The Big Picture

Three bottlenecks, attacked in sequence:

```
Era 1: DRAM-bound (register spills -> massive global traffic)
         Fix: shared memory, 1-qp collapse
Era 2: L2-bound (enough shared data, but poor access patterns)
         Fix: coalesced thread mapping, cooperative loads
Era 3: L2-saturated (85-90% L2 BW), low SM compute
         Fix: register pressure -> more blocks -> better occupancy -> SM 63%
```

The trajectory on the hardware utilization chart tells the whole story:

| Version | DRAM BW | L2 BW | SM Compute |
|---------|---------|-------|-----------|
| v00a    | 12%     | 82%   | 9%        |
| v00b    | 43%     | 82%   | 7%        |
| v01     | 22%     | 83%   | 9%        |
| v02     | 53%     | 32%   | 6%        |
| v03     | 14%     | 59%   | 17%       |
| v04     | 10%     | 47%   | 28%       |
| v05     | 11%     | 46%   | 36%       |
| v06     | 29%     | 83%   | 39%       |
| v07     | 29%     | 84%   | 39%       |
| v08     | 14%     | 90%   | 43%       |
| v09     | 19%     | 81%   | 63%       |
| v10     | 20%     | 85%   | 62%       |

SM compute climbs from 9% to 63% over the optimization campaign, while DRAM BW drops from ~40%
to 20%, and L2 BW saturates at 85–90%. The operator is now fundamentally limited by L2
bandwidth — and that bottleneck is structural, driven by the uncoalesced atomic scatter pattern
that is inherent to the wedge element topology.
