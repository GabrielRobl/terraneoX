# EpsDivDiv Operator — Optimization History & Technical Documentation

Production file: `src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp`
Predecessor files: `epsilon_divdiv_simple.hpp` (v00a), `epsilon_divdiv.hpp` (v00b)
GitHub base: `https://github.com/mantleconvection/terraneo`
Links use `blob/<commit>/...#L<line>` to pin each optimization to the exact commit where it was introduced.

---

## 1. Physical & Mathematical Background

### The Stokes Equation for Mantle Convection

TerraNeo solves the incompressible Stokes equation in a spherical shell geometry, modeling the viscous creep flow inside a planetary mantle:

```
-div( 2η ε(u) ) + ∇p = f    (momentum balance)
 div(u) = 0                   (incompressibility)
```

where **u** is the velocity vector field (3 components in Cartesian), **p** is pressure, **η** is the dynamic viscosity, **ε(u) = ½(∇u + ∇uᵀ)** is the symmetric strain rate tensor, and **f** is the body force (buoyancy).

### The Epsilon-DivDiv Bilinear Form

The **EpsDivDiv** operator implements the viscous part of the discrete Stokes operator. In weak (variational) form, it evaluates for a source velocity field **u** and a test function **v**:

```
a(u, v) = ∫_Ω  k(x) [ 2 ε(v) : ε(u)  -  (2/3) div(v) · div(u) ] dx
```

- **`ε(v) : ε(u)`** — the double contraction (Frobenius inner product) of the symmetric gradient tensors. This is the viscous dissipation term.
- **`div(v) · div(u)`** — the divergence-divergence coupling. The `-2/3` prefactor comes from the traceless (deviatoric) stress formulation: `σ = 2η(ε(u) - ⅓ tr(ε(u)) I)`. Since `tr(ε(u)) = div(u)`, this yields the `2ε - (2/3)div·I` form.
- **`k(x)`** — a spatially varying scalar coefficient, representing viscosity (or a viscosity ratio in preconditioners).

The operator computes `dst = A · src`, i.e. the matrix-free action of the stiffness matrix on a velocity vector, without ever assembling the global matrix.

### Discretization: Wedge Elements on a Spherical Shell

The computational domain is a thick spherical shell (e.g., core-mantle boundary to Earth's surface) discretized into:

1. **Icosahedral grid on the sphere** — the surface is tiled by triangular patches from a refined icosahedron, giving quasi-uniform coverage.
2. **Hexahedral cells** — each surface quad (from 2 triangles) is extruded radially, forming hex cells indexed by `(subdomain, x, y, r)`.
3. **Wedge elements** — each hex cell is split diagonally into **2 wedges** (triangular prisms). The wedge is the actual finite element. A wedge has **6 nodes** (3 on each radial face), and with 3 velocity components (VecDim=3) there are **18 local DoFs** per wedge.
4. **Q1 (trilinear) basis** — 6 shape functions per wedge, evaluated on a reference wedge with coordinates (ξ, η, ζ) ∈ [0,1]² × [-1,1].

### Jacobian and Reference-to-Physical Mapping

Each wedge is mapped from reference to physical coordinates via:

```
x(ξ,η,ζ) = Σ_i N_i(ξ,η,ζ) · x_i
```

where `x_i` are the physical node positions (3 surface points × 2 radial levels). The **Jacobian** `J = ∂x/∂(ξ,η,ζ)` is a 3×3 matrix. Its inverse transpose `J⁻ᵀ` maps reference gradients to physical gradients:

```
∇_x N_i = J⁻ᵀ · ∇_ξ N_i
```

For wedge elements in a radial shell, the Jacobian has a special structure: two lateral columns (edge vectors L1, L2 on the triangular face) and one radial column (midpoint direction Rm scaled by half-thickness). This structure is exploited in later versions (V-Current) via cross-product formulas that avoid forming the full 9-entry J matrix.

### Quadrature

Integration over each element uses numerical quadrature:

```
∫_Ω_e f dx ≈ Σ_q w_q · f(x_q) · |det(J(x_q))|
```

- **v00a through V01**: 6-point Felippa 3×2 rule (3 triangle × 2 line points) — exact for the bilinear form.
- **V03 onwards**: Collapsed to **1-point** Felippa 1×1 rule at the centroid (ξ=⅓, η=⅓, ζ=0, weight=1). This is a reduced-integration approximation that trades exactness for 6× fewer evaluations per element. The resulting operator is spectrally close enough for iterative solver convergence.

---

## 2. GPU Architecture Context

### Target Hardware

The operator is optimized for **NVIDIA H100 SXM GPUs** ([Hopper architecture](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/), compute capability 9.0). Key specs from the [H100 datasheet](https://resources.nvidia.com/en-us-gpu-resources/h100-datasheet-24306) and [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html):

- **132 Streaming Multiprocessors (SMs)**, each with 65536 32-bit registers and up to 228 KB shared memory (configurable L1/shared split).
- **Max 255 registers per thread**, max 32 thread blocks per SM. High register counts reduce **occupancy** — the ratio of active warps to the SM's capacity. Low occupancy means the GPU cannot hide memory latency.
- **80 GB HBM3** with **3.35 TB/s** memory bandwidth, but high latency (~300+ cycles). Shared memory is on-chip, acting as a programmer-managed cache.
- **34 TFLOPS FP64** peak (non-tensor). 16896 CUDA cores.
- **Coalesced memory access** is critical: consecutive threads should access consecutive addresses.

### Kokkos Execution Model

[Kokkos](https://github.com/kokkos/kokkos) is a performance-portable C++ library abstracting GPU/CPU parallelism. Key concepts used in this operator:

| Concept | Role in this operator |
|---------|----------------------|
| `MDRangePolicy` | Maps each (subdomain, x, y, r) cell to one thread — used in v00a through V01. Simple but no data sharing between threads. |
| `TeamPolicy` | Groups threads into **teams** (= CUDA thread blocks). Each team processes a tile of cells and shares data via scratch memory. Used from V03 onwards. |
| `team.team_rank()` | Thread's index within its team (0..team_size-1). |
| `TeamThreadRange` | Distributes a loop across team members — used for cooperative shared memory loads. |
| `Kokkos::single(PerTeam)` | Executes a lambda on exactly one thread per team (e.g., to load shared data). |
| `team.team_barrier()` | Synchronizes all threads in a team (ensures shared memory is visible). |
| `team_shmem` / scratch memory | Per-team on-chip memory (= CUDA shared memory). Declared via `set_scratch_size()` in launch config. |
| `LaunchBounds<BlockSize, MinBlocks>` | Hints to the compiler: max threads per block and min blocks per SM. Controls register allocation and occupancy. |
| `Kokkos::atomic_add` | Thread-safe accumulation into global memory — necessary because neighboring elements share nodes. |

---

## 3. Glossary

| Term | Meaning |
|------|---------|
| **hex cell** | A hexahedral (8-node) cell in the `(x, y, r)` grid, split into 2 wedges |
| **wedge** | Triangular prism (6 nodes). The actual finite element. Two wedges per hex cell. |
| **dimi, dimj** | Velocity component indices (0=x, 1=y, 2=z). dimi for test, dimj for trial. |
| **scatter** | Accumulating element contributions into the global destination vector via atomics |
| **gather** | Reading source DoFs from global or shared memory into registers |
| **invJ** / J⁻ᵀ | Inverse transpose of the Jacobian, used to transform reference gradients |
| **sym_grad / ε(u)** | Symmetric part of the velocity gradient tensor: ½(∇u + ∇uᵀ) |
| **double_contract** | Frobenius inner product A:B = Σᵢⱼ Aᵢⱼ Bᵢⱼ |
| **σ (sigma)** | Deviatoric stress tensor: 2ε(u) − ⅔ div(u)·I |
| **kwJ** | Combined weight: k(x) × |det(J)| × quadrature_weight |
| **GCA** | Galerkin Coarse Approximation — uses stored 18×18 local matrices on coarser multigrid levels |
| **free-slip BC** | Tangential-only flow at boundary (normal component = 0), requires local coordinate transform |
| **Dirichlet BC** | No-flow boundary (all components zero on boundary face) |
| **Neumann BC** | Natural (stress-free) boundary — no special treatment needed in weak form |
| **r_tile, lat_tile** | Tiling sizes along radial and lateral grid dimensions |
| **r_passes** | Number of radial cells each thread processes sequentially (amortizes shared memory load cost) |
| **occupancy** | Fraction of GPU SM capacity actually used; limited by register count, shared memory, and block size |
| **LaunchBounds** | Compiler hint controlling register allocation: `<max_threads, min_blocks_per_SM>` |

---

## 4. Version History

### v00a: EpsilonDivDivSimple (aba88f1 — "Add viscosity weighted mass, block triangular preconditioners")

File: `src/terra/fe/wedge/operators/shell/epsilon_divdiv_simple.hpp`

**Approach:** The original, textbook-style implementation. **Assembles the full 18×18 local element matrix A, then multiplies `dst = A · src`.** This is the clearest expression of the math but the worst for GPU performance.

- `MDRangePolicy` — 1 thread per hex cell, no data sharing
  [L196](https://github.com/mantleconvection/terraneo/blob/aba88f1/src/terra/fe/wedge/operators/shell/epsilon_divdiv_simple.hpp#L196) `parallel_for("matvec", local_domain_md_range_policy_cells(...))`
- **Assembles full `dense::Mat<18,18> A[2]`** per hex cell (both wedges), then multiplies `dst = A * src`
  [L214](https://github.com/mantleconvection/terraneo/blob/aba88f1/src/terra/fe/wedge/operators/shell/epsilon_divdiv_simple.hpp#L214) `dense::Mat< ScalarT, 18, 18 > A[num_wedges_per_hex_cell] = {};`
  [L365-377](https://github.com/mantleconvection/terraneo/blob/aba88f1/src/terra/fe/wedge/operators/shell/epsilon_divdiv_simple.hpp#L365-L377) `dst[0] = A[0] * src[0]; dst[1] = A[1] * src[1];`
- **6-point Felippa 3×2 quadrature** (3 triangle × 2 line points)
  [L227](https://github.com/mantleconvection/terraneo/blob/aba88f1/src/terra/fe/wedge/operators/shell/epsilon_divdiv_simple.hpp#L227) `quad_felippa_3x2_num_quad_points`
  [L247](https://github.com/mantleconvection/terraneo/blob/aba88f1/src/terra/fe/wedge/operators/shell/epsilon_divdiv_simple.hpp#L247) `for ( int q = 0; q < num_quad_points; q++ )`
- Nested `dimi × dimj` loops (O(3×3)) assembling A via `double_contract`
  [L239](https://github.com/mantleconvection/terraneo/blob/aba88f1/src/terra/fe/wedge/operators/shell/epsilon_divdiv_simple.hpp#L239) `for ( int dimi = 0; dimi < 3; ++dimi )`
  [L241](https://github.com/mantleconvection/terraneo/blob/aba88f1/src/terra/fe/wedge/operators/shell/epsilon_divdiv_simple.hpp#L241) `for ( int dimj = 0; dimj < 3; ++dimj )`
  [L282-284](https://github.com/mantleconvection/terraneo/blob/aba88f1/src/terra/fe/wedge/operators/shell/epsilon_divdiv_simple.hpp#L282-L284) `A[wedge](...) += w * k_eval * abs_det * (0.5 * sym_grad_i.double_contract(sym_grad_j) - 2/3 * div_i * div_j)`
- Uses **dense `Mat<3,3>` objects** for Jacobian and symmetric gradients — clean abstraction but heavy register use
  [L255-256](https://github.com/mantleconvection/terraneo/blob/aba88f1/src/terra/fe/wedge/operators/shell/epsilon_divdiv_simple.hpp#L255-L256) `J`, `J.inv_transposed(det)`
  [L268](https://github.com/mantleconvection/terraneo/blob/aba88f1/src/terra/fe/wedge/operators/shell/epsilon_divdiv_simple.hpp#L268) `sym_grad_i = (grad_i + grad_i.transposed())`
- **Boundary handling via 18×18 Hadamard-product mask**: fills a mask with 0/1, then zeros out constrained coupling entries
  [L300-301](https://github.com/mantleconvection/terraneo/blob/aba88f1/src/terra/fe/wedge/operators/shell/epsilon_divdiv_simple.hpp#L300-L301) `boundary_mask.fill(1.0);`
  [L337](https://github.com/mantleconvection/terraneo/blob/aba88f1/src/terra/fe/wedge/operators/shell/epsilon_divdiv_simple.hpp#L337) `A[wedge].hadamard_product( boundary_mask );`
- Scatter via `atomically_add_local_wedge_vector_coefficients` helper
  [L387-392](https://github.com/mantleconvection/terraneo/blob/aba88f1/src/terra/fe/wedge/operators/shell/epsilon_divdiv_simple.hpp#L387-L392)

**Why this is slow on GPUs:** Storing two 18×18 matrices (648 doubles = 5184 bytes) in registers per thread is far beyond what GPUs can handle; the compiler spills to local memory. The 6-point quadrature means 6× more Jacobian evaluations. The O(3×3)=9 dimi/dimj passes means heavy redundant computation. The `dense::Mat<3,3>` temporaries add further register pressure from padding and abstraction overhead.

---

### v00b: EpsilonDivDiv — refactored (bdb954c — "Use refactored eps + divdiv in Stokes test")

File: `src/terra/fe/wedge/operators/shell/epsilon_divdiv.hpp`

**Approach:** Evolved through several commits (10eb34b → bec4d13 → 63c8f46 → 0b3f724 → bdb954c). The key breakthrough is **fused local matvec** — instead of first assembling 18×18 A and then multiplying, the assembly and application are fused into a single pass. This eliminates the 648-entry matrix from registers entirely.

- **Fused local matvec** — no full 18×18 matrix
  [L382-396](https://github.com/mantleconvection/terraneo/blob/bdb954c/src/terra/fe/wedge/operators/shell/epsilon_divdiv.hpp#L382-L396) `assemble_trial_test_vecs(...)` then `fused_local_mv(...)`
- **`assemble_trial_test_vecs()`**: precomputes symmetric gradient vectors for trial and test functions
  [L227](https://github.com/mantleconvection/terraneo/blob/bdb954c/src/terra/fe/wedge/operators/shell/epsilon_divdiv.hpp#L227)
- Gather/scatter via hex-node indexing (`hex_offset_x/y/r[8]`) with `atomic_add` per node
  [L348-350](https://github.com/mantleconvection/terraneo/blob/bdb954c/src/terra/fe/wedge/operators/shell/epsilon_divdiv.hpp#L348-L350) `hex_offset_x/y/r` arrays
  [L411](https://github.com/mantleconvection/terraneo/blob/bdb954c/src/terra/fe/wedge/operators/shell/epsilon_divdiv.hpp#L411) `Kokkos::atomic_add(&dst_(...))`
- Still uses **6-point Felippa 3×2 quadrature**, `MDRangePolicy`, O(3×3) dimi/dimj loops
  [L198](https://github.com/mantleconvection/terraneo/blob/bdb954c/src/terra/fe/wedge/operators/shell/epsilon_divdiv.hpp#L198) `parallel_for("matvec", local_domain_md_range_policy_cells(...))`
  [L354-356](https://github.com/mantleconvection/terraneo/blob/bdb954c/src/terra/fe/wedge/operators/shell/epsilon_divdiv.hpp#L354-L356) `for dimi × for dimj`
  [L375](https://github.com/mantleconvection/terraneo/blob/bdb954c/src/terra/fe/wedge/operators/shell/epsilon_divdiv.hpp#L375) `for ( int q = 0; q < num_quad_points; q++ )`
- **Supports GCA** (Galerkin Coarse Approximation) via `assemble_local_matrix()` for stored matrices
  [L269](https://github.com/mantleconvection/terraneo/blob/bdb954c/src/terra/fe/wedge/operators/shell/epsilon_divdiv.hpp#L269) stored-matrix path in `operator()`

#### Key intermediate refactors (all in `epsilon_divdiv.hpp`):
- (bec4d13) Fuse separate `dirichlet_cmb`, `dirichlet_surface`, `neumann`, `diagonal` functions into one
- (63c8f46) Fuse diagonal kernel into single kernel — no duplicate code paths
- (0b3f724) Add on-the-fly single-element assembly + trial/test gradient vector assembly

**Why this is better:** Eliminating the full 18×18 matrix frees ~648 register slots per thread. The fused approach computes `dst += (test_grad · trial_grad) * src` inline, keeping only the current test/trial gradient vectors alive at a time (~36 doubles instead of 648).

---

### V01: Initial Code-Generated Kernel (e7ae1b3 — "Generate epsdivdiv kernel + benchmark")

**Approach:** The first code-generated version. A Python script generates explicit scalar arithmetic (no `dense::Mat` abstractions), giving the compiler full visibility into the computation. Still uses MDRangePolicy and 6 quadrature points.

- `MDRangePolicy` — 1 thread per hex cell
  [L196](https://github.com/mantleconvection/terraneo/blob/e7ae1b3/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L196) `parallel_for("matvec", local_domain_md_range_policy_cells(...))`
- Nested `for dimi × for dimj` — O(3×3)=9 dim-pair passes
  [L403](https://github.com/mantleconvection/terraneo/blob/e7ae1b3/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L403) `for ( dimi = 0; dimi < 3 ...)`
  [L405](https://github.com/mantleconvection/terraneo/blob/e7ae1b3/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L405) `for ( dimj = 0; dimj < 3 ...)`
- Full 6-point quadrature per wedge
  [L316](https://github.com/mantleconvection/terraneo/blob/e7ae1b3/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L316) `for ( q = 0; q < 6; q += 1 )`
- `dst_array[3][2][6]` in registers, 24 `atomic_add`s
  [L311](https://github.com/mantleconvection/terraneo/blob/e7ae1b3/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L311) `double dst_array[3][2][6] = { 0 };`
  [L513](https://github.com/mantleconvection/terraneo/blob/e7ae1b3/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L513) atomic_add scatter block

**Why this helps:** Replacing `dense::Mat<3,3>` template objects with raw `double` scalars gives the compiler better control over register allocation. No virtual function overhead, no unnecessary copies through matrix constructors. This is the baseline for all subsequent GPU-focused optimizations.

---

### V02: Split dimi/dimj (c9c1e21 — "Split dimi/dimj loop → 2×3 complexity instead of 3×3")

**Approach:** The key algebraic insight is that the bilinear form `a(u,v)` can be decomposed: first accumulate the trial contribution (gather over dimj=0,1,2 into a symmetric "stress" tensor), then scatter the test contribution (over dimi=0,1,2). This reduces the loop complexity from O(3×3)=9 to O(3+3)=6 passes.

- **Split nested loops**: gather over dimj first, then scatter over dimi
  [L396](https://github.com/mantleconvection/terraneo/blob/c9c1e21/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L396) `for (dimj = 0; dimj < 3 ...)` — trial/gather pass
  [L421](https://github.com/mantleconvection/terraneo/blob/c9c1e21/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L421) `for (dimi = 0; dimi < 3 ...)` — test/scatter pass
- Separate diagonal/boundary loop
  [L440](https://github.com/mantleconvection/terraneo/blob/c9c1e21/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L440) `for (dim_diagBC = 0; dim_diagBC < 3 ...)`

**Why this helps:** The original O(3×3) approach recomputes the gradient transform for each (dimi,dimj) pair. By splitting into gather + scatter, the intermediate "stress tensor" `σᵢⱼ = Σ_nodes (grad_j · src_j)` is accumulated once (3 passes for dimj) and then scattered (3 passes for dimi). This cuts ~33% of the floating-point work and reduces register pressure from shorter-lived intermediates.

---

### V03: Teams + Precomputation (b875f4c — "use teams + precomputation in epsdivdiv")

**Approach:** The first major architectural change. Switches from `MDRangePolicy` (independent threads) to `TeamPolicy` (cooperating thread blocks). One team handles an entire radial column of cells, allowing **shared memory** for data that is common to all cells in the column (surface coordinates). Also collapses from 6-point to 1-point quadrature.

- **Switch from MDRangePolicy to TeamPolicy** (1 team per xy-column, threads along r)
  [L240-241](https://github.com/mantleconvection/terraneo/blob/b875f4c/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L240-L241) `TeamPolicy<>( blocks_, block_size_ ).set_scratch_size(...)`
- `block_size_ = min(128, hex_rad_)`, blocks = subdomains × hex_lat² × blocks_per_column
  [L110](https://github.com/mantleconvection/terraneo/blob/b875f4c/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L110) `block_size_ = std::min( 128, threads_per_column );`
  [L112](https://github.com/mantleconvection/terraneo/blob/b875f4c/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L112) `blocks_ = local_subdomains_ * hex_lat_ * hex_lat_ * blocks_per_column_;`
- **Surface coords into shared memory** via team_rank==0 guard + barrier
  [L490](https://github.com/mantleconvection/terraneo/blob/b875f4c/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L490) shmem allocation
  [L493](https://github.com/mantleconvection/terraneo/blob/b875f4c/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L493) `if ( team.team_rank() == 0 )` loads coords
  [L511](https://github.com/mantleconvection/terraneo/blob/b875f4c/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L511) `team.team_barrier()`
- **Collapsed to single quadrature point** (qp=⅓,⅓,0; qw=1)
  [L471-474](https://github.com/mantleconvection/terraneo/blob/b875f4c/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L471-L474)
- Mask-based `has_flag()` replaces raw index comparisons for boundary detection
  [L142-149](https://github.com/mantleconvection/terraneo/blob/b875f4c/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L142-L149)

**Why this helps:**
- **Shared memory for coords:** All cells in a radial column share the same 3 surface triangle vertices (2 wedges × 3 nodes × 3 coords = 18 doubles). Without shared memory, each thread independently loads these from global memory — 18 loads × N_rad threads. With shared memory, they are loaded once by one thread and broadcast to all.
- **1-point quadrature:** Reduces per-element work by 6×. The shape function gradients at the centroid become compile-time constants (`dN_ref[6][3]`), enabling constant propagation. The accuracy trade-off is acceptable for iterative Krylov solvers where the operator only needs spectral equivalence, not exact integration.
- **TeamPolicy:** Enables the shared-memory pattern and gives the programmer control over block size, which affects occupancy.

---

### V04: Shared Mem Coords via `Kokkos::single` (fe1c12e — "improved 1thread=1cell: more scopes, qp collapsed, shared mem for coords")

**Approach:** Refinement of V03's shared memory loading. Uses `Kokkos::single(PerTeam)` instead of an `if(team_rank==0)` guard, which is more idiomatic and portable. Introduces the `column_grad_to_sym()` helper to precompute the symmetric gradient for a given dimension via a `switch(dim)` statement.

- **Full shmem for `wedge_surf_phy_coords[2][3][3]`** loaded via `Kokkos::single(PerTeam)`
  [L734](https://github.com/mantleconvection/terraneo/blob/fe1c12e/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L734) `Scratch3D wedge_surf_phy_coords( shmem, 2, 3, 3 );`
  [L736](https://github.com/mantleconvection/terraneo/blob/fe1c12e/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L736) `Kokkos::single( Kokkos::PerTeam( team ), [&]() { ... });`
- Introduces `column_grad_to_sym()` helper
  [L399](https://github.com/mantleconvection/terraneo/blob/fe1c12e/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L399) `void column_grad_to_sym(...)`

**Why this helps:** The `Kokkos::single(PerTeam)` pattern is semantically clearer and generates better code on some backends. The `column_grad_to_sym` helper reduces redundant code in the gather/scatter loop bodies.

---

### V05: Shared Memory for src + k (70bacff — "Update with shared mem for src and k dofs per team")

**Approach:** Extends the shared-memory pattern from just coordinates to the **source vector** and **viscosity coefficient**. In a radial column, neighboring cells share nodes (the top face of cell `r` is the bottom face of cell `r+1`), so loading the full column into shared memory avoids redundant global reads.

- **Source vector and coefficient loaded into team scratch**
  [L316-319](https://github.com/mantleconvection/terraneo/blob/70bacff/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L316-L319) `team_shmem_size()` — coords+src+k
- `src_sh(nlev, 4, 3)` and `k_sh(nlev, 4)` scratch views
  [L398](https://github.com/mantleconvection/terraneo/blob/70bacff/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L398) `Scratch3DLevels src_sh( shmem, nlev, 4, 3 );`
  [L402](https://github.com/mantleconvection/terraneo/blob/70bacff/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L402) `Scratch2DLevels k_sh( shmem, nlev, 4 );`
- Cooperative load: each thread loads its radial level, last thread loads extra (for r+1 boundary)
  [L460-481](https://github.com/mantleconvection/terraneo/blob/70bacff/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L460-L481) `auto load_level = [&](int level) { ... }`
  [L492](https://github.com/mantleconvection/terraneo/blob/70bacff/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L492) `team.team_barrier();`
- `WEDGE_TO_UNIQUE[2][6]` mapping → `dst8[3][8]` (8 unique hex nodes, deduplicating shared wedge nodes)
  [L355-358](https://github.com/mantleconvection/terraneo/blob/70bacff/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L355-L358) `WEDGE_TO_UNIQUE` array
  [L513](https://github.com/mantleconvection/terraneo/blob/70bacff/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L513) `double dst8[3][8] = { 0.0 };`

**Why this helps:**
- **Shared memory for src:** Each wedge reads 6 source nodes × 3 components = 18 doubles from global memory. Neighboring wedges share nodes, so in a column of N cells, the naive approach reads ~36N values but only ~4N+2 are unique. Shared memory reduces this to one read per unique value.
- **Shared memory for k:** Same argument — 6 k-values per wedge with sharing across neighbors.
- **`WEDGE_TO_UNIQUE` + `dst8[3][8]`:** The two wedges in a hex cell share 4 of their 6 nodes. By accumulating results into 8 unique hex-node slots (instead of 2×6=12 wedge-node slots), the scatter writes 8 atomics instead of 12, and avoids double-counting shared nodes.

---

### V06: XY Tiling (7f053dd — "Add xy tiling to eps + divdiv")

**Approach:** Extends the 1D radial tiling from V03-V05 to **3D tiling** over both lateral (x,y) and radial (r) dimensions. A team now handles a `lat_tile × lat_tile × r_tile = 4×4×8 = 128` cell block. All surface coordinates and source data for the tile slab are loaded cooperatively into shared memory.

- **3D (x,y,r) tiling**: `lat_tile=4, r_tile=8` → `team_size=128`
  [L76-81](https://github.com/mantleconvection/terraneo/blob/7f053dd/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L76-L81) `lat_tile_, r_tile_, team_size_`
  [L125-126](https://github.com/mantleconvection/terraneo/blob/7f053dd/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L125-L126) `lat_tile_ = 4; r_tile_ = 8;`
- Thread mapping: `tx = tid%lat_tile, ty = (tid/lat_tile)%lat_tile, tr = tid/(lat_tile*lat_tile)`
  [L364-366](https://github.com/mantleconvection/terraneo/blob/7f053dd/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L364-L366)
- Full tile slab loaded cooperatively via `TeamThreadRange`
  [L402](https://github.com/mantleconvection/terraneo/blob/7f053dd/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L402) coords load
  [L418](https://github.com/mantleconvection/terraneo/blob/7f053dd/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L418) src/k radial load

**Why this helps:** In the column-only tiling (V03-V05), surface coordinates for each (x,y) column are loaded by a separate team — but neighboring columns share edge/corner nodes. With lateral tiling, a 4×4 tile block has `5×5=25` unique surface nodes instead of `4×4×4=64` (4 nodes per column × 16 columns). This is a 2.56× reduction in coordinate loads. The same sharing benefit applies to source and k values along the lateral edges. It also increases the team size (128 threads), which improves GPU occupancy and warp scheduling.

---

### V07: Split Paths (95fbf31 — "split freeslip/store path and generated path") — ~47× speedup

**Approach:** The first version to recognize that boundary conditions create fundamentally different code paths. The **slow path** (stored matrices, free-slip BCs) is rarely needed in typical Stokes solves; the **fast path** (Dirichlet/Neumann matrix-free) handles the common case. By deciding the path on the **host** before kernel launch, the fast path is completely free of runtime branches that would cause warp divergence.

- **Host-side kernel dispatch**: `use_slow_path_` set at construction, no runtime branch in device kernel
  [L118](https://github.com/mantleconvection/terraneo/blob/95fbf31/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L118) `bool use_slow_path_ = false;`
  [L138](https://github.com/mantleconvection/terraneo/blob/95fbf31/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L138) `use_slow_path_ = has_freeslip_bc || has_stored_matrices;`
- Slow path: stored local matrices or freeslip; Fast path: Dirichlet/Neumann matrix-free
  [L325-342](https://github.com/mantleconvection/terraneo/blob/95fbf31/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L325) `if (use_slow_path_) → operator_slow_kernel; else → operator_fast_kernel`

**Why this helps:** On GPUs, all threads in a warp (32 threads) must execute the same instruction. If some threads take the free-slip path (requiring coordinate transforms, normal/tangential projection) and others don't, the warp executes both paths serially — **warp divergence**. By dispatching separate kernels, the fast path has zero branch overhead. The ~47× speedup is not entirely from branch elimination (it also reflects the difference between stored-matrix and matrix-free approaches), but the principle of host-side dispatch becomes foundational for all later versions.

---

### V08: Scalar Coalesced (03f228d — "separate vec funcs into 3 scalar funcs for coalesced accesses")

**Approach:** Reorders the thread-to-cell mapping so that the **radial index `r` is the fastest-varying** (innermost) dimension. Since consecutive radial cells are stored contiguously in memory, this ensures that consecutive threads within a warp access consecutive memory addresses — **memory coalescing**.

- **Thread index reordering**: r is now fastest-varying
  [L485-487](https://github.com/mantleconvection/terraneo/blob/03f228d/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L485-L487) `tr = tid % r_tile_; tx = (tid/r_tile_) % lat_tile_; ty = ...`
- `r_tile=16` (doubled from 8), `team_size = 4×4×16 = 256`
  [L192](https://github.com/mantleconvection/terraneo/blob/03f228d/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L192) `r_tile_ = 16;`
- **Three-way `KernelPath` enum**: `Slow / FastDirichletNeumann / FastFreeslip`
  [L121-127](https://github.com/mantleconvection/terraneo/blob/03f228d/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L121-L127) `enum class KernelPath { ... }`
- Per-wedge scatter `dst_w[3][6]` (shorter register lifetime than `dst8[3][8]`)
  [L999](https://github.com/mantleconvection/terraneo/blob/03f228d/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L999) `double dst_w[3][6] = { 0.0 };`

**Why this helps:**
- **Coalescing:** With the old mapping `tx = tid%4`, consecutive threads address different (x) positions separated by a stride of N_y×N_r in memory. With `tr = tid%16`, consecutive threads access r, r+1, r+2, ... — which are contiguous. A coalesced 128-byte transaction serves 16 doubles (128B/8B) vs 16 scattered cache lines. This can yield 4-16× improvement in effective memory bandwidth.
- **Larger r_tile:** Doubling from 8 to 16 means each team covers more radial cells, amortizing the shared memory load cost. But team_size also doubles to 256, which affects occupancy.
- **Per-wedge scatter:** Instead of accumulating into 8 unique hex-node slots (carrying `dst8[3][8]=24` doubles across both wedges), the result is scattered per-wedge into `dst_w[3][6]=18` doubles. The shorter register lifetime (only alive during one wedge's scatter, then freed) reduces peak register pressure.

---

### V09: Separate Scatter (d208988 — "separate scatter to 7.6 gdofs") — **7.6 Gdof/s**

**Approach:** Splits the kernel body into two phases with a hard scope boundary between them. **Phase 1** (gather) computes the Jacobian and accumulates the stress tensor `σ` from source data. **Phase 2** (scatter) recomputes the Jacobian and applies `kwJ · σ · g` to destination. The key trade-off: recomputing J costs ~30 FLOPs, but freeing all gather-phase temporaries (invJ rows, source values) reclaims ~30 registers.

- **Split gather/scatter into separate Jacobian scopes**: invJ freed between phases
  [L1005](https://github.com/mantleconvection/terraneo/blob/d208988/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L1005) `// Phase 1: Jacobian + Gather (gu tensor)`
  [L1082](https://github.com/mantleconvection/terraneo/blob/d208988/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L1082) `// Phase 2: Recompute Jacobian + Scatter`
- **No register buffer for dst** — inline atomics during scatter
  [L1134-1145](https://github.com/mantleconvection/terraneo/blob/d208988/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L1134-L1145) 3 inline `atomic_add` per node
- **`LaunchBounds<128, 5>`** on fast Dirichlet/Neumann path
  [L377](https://github.com/mantleconvection/terraneo/blob/d208988/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L377) `TeamPolicy< LaunchBounds< 128, 5 > >`
- **`template <bool Diagonal>`** specialization eliminates runtime branch
  [L847](https://github.com/mantleconvection/terraneo/blob/d208988/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L847) `template < bool Diagonal > void operator_fast_dirichlet_neumann_path(...)`

**Why this helps:**
- **Register pressure:** On H100, each SM has 65536 registers shared across all threads. At 128 threads/block and 96 regs/thread, only 5 blocks fit per SM (128×96=12288 regs/block, 5×12288=61440). By reducing peak register usage through scoping (recompute J instead of keeping it live), the compiler can allocate fewer registers per thread, allowing more blocks per SM → higher occupancy → better latency hiding.
- **`LaunchBounds<128, 5>`:** Tells the compiler: "compile for 128 threads per block, targeting ≥5 blocks per SM." This forces the compiler to cap register usage at ⌊65536/(128×5)⌋ = 102 registers/thread. Without this hint, the compiler might use more registers for ILP, reducing occupancy.
- **`template<bool Diagonal>`:** The diagonal action (needed for Jacobi preconditioning) is structurally simpler than the full operator. Templating eliminates the `if(diagonal_)` branch inside the hot loop, letting the compiler optimize each variant independently.

---

### V10: Sequential r_passes (c20ae75 — "use sequential r_passes to reduce register pressure → 7.8 Gdofs") — **7.8 Gdof/s**

**Approach:** Each thread processes **2 radial cells sequentially** (r_passes=2) instead of 1. The shared memory loads cover a `r_tile_block = r_tile × r_passes = 8×2 = 16` radial slab, but each thread only computes one cell at a time, reusing registers between passes.

- **`r_passes=2`**: each thread processes 2 radial cells sequentially
  [L107](https://github.com/mantleconvection/terraneo/blob/c20ae75/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L107) `int r_passes_;`
  [L195](https://github.com/mantleconvection/terraneo/blob/c20ae75/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L195) `r_passes_ = 2;`
- `r_tile_block = r_tile × r_passes = 16` — amortizes shmem loads
  [L196](https://github.com/mantleconvection/terraneo/blob/c20ae75/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L196) `r_tile_block_ = r_tile_ * r_passes_;`
- Sequential pass loop
  [L967](https://github.com/mantleconvection/terraneo/blob/c20ae75/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L967) `for ( int pass = 0; pass < r_passes_; ++pass )`

**Why this helps:**
- **Register reduction via r_tile halving:** With `r_tile=8` (instead of 16) and `r_passes=2`, the team_size drops from `4×4×16=256` to `4×4×8=128`. Fewer threads per block means the compiler has more registers per thread available, or conversely, the same register budget serves more blocks per SM.
- **Amortized shared memory cost:** The shared memory slab covers 16 radial levels (same as V08), loaded once by 128 threads. But each thread processes 2 cells sequentially, so the cost of the cooperative load + barrier is amortized over 2× the work.
- **Better occupancy:** `LaunchBounds<128, N>` with a smaller team size means more blocks can run concurrently. The ~3% throughput gain (7.6 → 7.8 Gdof/s) comes from this improved scheduling.

---

### Current Version (f6ae663 — "restructure jacobian comp to save registers → down to 80 per thread") — ~7.8 Gdof/s

**Approach:** Restructures the Jacobian computation to use a **cross-product formulation** that avoids storing the full 9-entry Jacobian matrix. Instead, the inverse Jacobian rows are computed directly from edge vectors and their cross products. Also introduces a **stress-tensor formulation** in the gather phase and **FMA-friendly prefactors** in the scatter phase.

- **Cross-product Jacobian inverse**: L1, L2, Rm → A=L2×Rm, B=Rm×L1, C=L1×L2 (avoids full J matrix)
  [L1020-1028](https://github.com/mantleconvection/terraneo/blob/f6ae663/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L1020-L1028) L1, L2, Rm vectors
  [L1030-1038](https://github.com/mantleconvection/terraneo/blob/f6ae663/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L1030-L1038) cross products A, B, C
  [L1039](https://github.com/mantleconvection/terraneo/blob/f6ae663/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L1039) `lat_det = L1 · A`
  [L1045-1047](https://github.com/mantleconvection/terraneo/blob/f6ae663/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L1045-L1047) invJ rows from scaled A, B, C
  [L1048](https://github.com/mantleconvection/terraneo/blob/f6ae663/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L1048) `// L1,L2,Rm,A,B,C,...freed`
- **No J recomputation in scatter** (unlike V09): invJ stays live, σ is scattered directly
  [L1081](https://github.com/mantleconvection/terraneo/blob/f6ae663/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L1081) `// Scatter (reuses i00..i22 — no J recomputation)`
- **Stress-tensor gather**: accumulates `σ = 2ε(u) - ⅔ div(u)·I` directly
  [L1056-1084](https://github.com/mantleconvection/terraneo/blob/f6ae663/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L1056-L1084) `sig00, sig11, sig22, sig10, sig20, sig21`
- **FMA-friendly scatter**: `dst += kwJ · (g · σ)` per test node, no intermediate buffers
  [L1090-1091](https://github.com/mantleconvection/terraneo/blob/f6ae663/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L1090-L1091) `kwJ` prefactor
  [L1104-1112](https://github.com/mantleconvection/terraneo/blob/f6ae663/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L1104-L1112) scatter: `kwJ * (g0*sig00 + g1*sig10 + g2*sig20)` per component
- **`LaunchBounds<128, 6>`** (up from 5 → higher occupancy target)
  [L377](https://github.com/mantleconvection/terraneo/blob/f6ae663/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L377) `TeamPolicy< LaunchBounds< 128, 6 > >`
- 80 registers/thread (down from ~96+), but no additional throughput gain over V10

**Why this helps:**
- **Cross-product J:** The standard approach stores J as 9 doubles, computes det, then invJ as 9 more doubles = 18 registers just for the Jacobian. The cross-product approach computes invJ rows directly: A, B, C are temporary (9 doubles, freed after scaling), and only the 9 invJ entries survive — at peak ~18 doubles live, but the temporaries are scoped so the compiler can reuse those registers immediately. The net saving is ~16 registers vs the standard path.
- **No J recomputation:** V09 traded registers for recomputation (compute J twice — once for gather, once for scatter). The current version avoids this by keeping invJ alive as `i00..i22` (9 doubles) across both gather and scatter phases. This is affordable now because the cross-product formulation freed enough registers.
- **Stress tensor accumulation:** Instead of separate `gu` tensor components + post-hoc scaling, the gather directly computes `σ = 2ε(u) - ⅔ div(u)·I`. This is algebraically equivalent but uses 6 symmetric tensor entries (`sig00..sig21`) instead of 9 full tensor entries — 33% fewer registers for the intermediate.
- **FMA-friendly:** The scatter expressions `kwJ * (g0*sig00 + g1*sig10 + g2*sig20)` map directly to fused multiply-add (FMA) instructions. FMA computes `a*b + c` in one cycle with one rounding, and the A100 has a 1:1 FMA-to-add ratio, so this formulation extracts maximum throughput from the functional units.
- **`LaunchBounds<128, 6>`:** With 80 regs/thread, 128×80=10240 regs/block, and 65536/10240=6.4 → 6 blocks/SM. Bumping the target from 5 to 6 is now feasible and increases theoretical occupancy from 5×128/2048=31.3% to 6×128/2048=37.5%.

---

## 5. Summary Table

| Version | Date | Commit | File | Key Innovation | Gdof/s | Speedup |
|---------|------|--------|------|----------------|--------|---------|
| v00a | 2025-11-09 | [aba88f1](https://github.com/mantleconvection/terraneo/commit/aba88f1) | [`epsilon_divdiv_simple.hpp`](../../src/terra/fe/wedge/operators/shell/epsilon_divdiv_simple.hpp) | Textbook: assemble 18×18 A, then A·src; Hadamard BC mask | 0.012 | 1× |
| v00b | 2025-11-17 | [bdb954c](https://github.com/mantleconvection/terraneo/commit/bdb954c) | [`epsilon_divdiv.hpp`](../../src/terra/fe/wedge/operators/shell/epsilon_divdiv.hpp) | Fused local matvec (no full A), trial/test vecs, GCA support | 0.019 | 1.6× |
| V01 | 2025-12-09 | [e7ae1b3](https://github.com/mantleconvection/terraneo/commit/e7ae1b3) | [`epsilon_divdiv_kerngen_v01_initial.hpp`](../../src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen_v01_initial.hpp) | Code-gen: MDRange, 6-qp, O(3×3) dimi/dimj, scalar arith | 0.044 | 3.8× |
| V02 | 2025-12-09 | [c9c1e21](https://github.com/mantleconvection/terraneo/commit/c9c1e21) | [`epsilon_divdiv_kerngen_v02_split_dimij.hpp`](../../src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen_v02_split_dimij.hpp) | Split dimi/dimj → O(3+3), fewer FLOPs | 0.107 | 9.2× |
| V03 | 2025-12-18 | [b875f4c](https://github.com/mantleconvection/terraneo/commit/b875f4c) | [`epsilon_divdiv_kerngen_v03_teams_precomp.hpp`](../../src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen_v03_teams_precomp.hpp) | TeamPolicy, 1-qp collapse, shmem coords | 1.71 | 146× |
| V04 | 2026-02-09 | [fe1c12e](https://github.com/mantleconvection/terraneo/commit/fe1c12e) | [`epsilon_divdiv_kerngen_v04_shmem_coords.hpp`](../../src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen_v04_shmem_coords.hpp) | `Kokkos::single(PerTeam)` coord load, `column_grad_to_sym` | 4.30 | 368× |
| V05 | 2026-02-10 | [70bacff](https://github.com/mantleconvection/terraneo/commit/70bacff) | [`epsilon_divdiv_kerngen_v05_shmem_src_k.hpp`](../../src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen_v05_shmem_src_k.hpp) | Shmem for src+k, `WEDGE_TO_UNIQUE` dedup | 5.41 | 463× |
| V06 | 2026-02-11 | [7f053dd](https://github.com/mantleconvection/terraneo/commit/7f053dd) | [`epsilon_divdiv_kerngen_v06_xy_tiling.hpp`](../../src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen_v06_xy_tiling.hpp) | 3D xy+r tiling (4×4×8), cooperative loads | 4.87 | 416× |
| V07 | 2026-02-22 | [95fbf31](https://github.com/mantleconvection/terraneo/commit/95fbf31) | [`epsilon_divdiv_kerngen_v07_split_paths.hpp`](../../src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen_v07_split_paths.hpp) | Host-side fast/slow path dispatch | 4.85 | 415× |
| V08 | 2026-02-25 | [03f228d](https://github.com/mantleconvection/terraneo/commit/03f228d) | [`epsilon_divdiv_kerngen_v08_scalar_coalesced.hpp`](../../src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen_v08_scalar_coalesced.hpp) | Coalesced r-first mapping, 3-way path, per-wedge scatter | 6.44 | 551× |
| V09 | 2026-02-27 | [d208988](https://github.com/mantleconvection/terraneo/commit/d208988) | [`epsilon_divdiv_kerngen_v09_separate_scatter.hpp`](../../src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen_v09_separate_scatter.hpp) | Separate gather/scatter (2× J), LB<128,5>, `template<Diagonal>` | 7.69 | 658× |
| V10 | 2026-02-28 | [c20ae75](https://github.com/mantleconvection/terraneo/commit/c20ae75) | [`epsilon_divdiv_kerngen_v10_seq_rpasses.hpp`](../../src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen_v10_seq_rpasses.hpp) | Sequential r_passes=2, amortized shmem | 7.84 | 670× |
| Cur | 2026-03-02 | [f6ae663](https://github.com/mantleconvection/terraneo/commit/f6ae663) | [`epsilon_divdiv_kerngen.hpp`](../../src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp) | Cross-product J (80 regs), stress-tensor gather, FMA scatter, LB<128,6> | ~7.8 | ~670× |

Benchmarked on NVIDIA H100 SXM, level 8 (505M dofs), 10 executions, single GPU. See `throughput_data.csv` for raw data.

## 6. Optimization Themes

Looking across the full history, the optimizations fall into recurring themes:

1. **Reduce register pressure** — the dominant constraint on GPU occupancy.
   - v00b: Fused matvec eliminates 18×18 matrix (648 → ~36 regs)
   - V02: Split dimi/dimj reduces live intermediates
   - V08: Per-wedge scatter (`dst_w[3][6]`) vs. per-hex (`dst8[3][8]`)
   - V09: Scoped gather/scatter phases, recompute J to free registers
   - V10: Sequential r_passes halves team_size, frees register budget
   - Current: Cross-product J avoids full 9-entry matrix

2. **Exploit shared memory** for data reuse across threads.
   - V03: Shared coords (same surface for entire column)
   - V04: Cleaner loading via `Kokkos::single`
   - V05: Shared src + k (neighbors share radial nodes)
   - V06: 3D tiling shares lateral edges too

3. **Eliminate runtime branches** in the GPU kernel.
   - V07: Host-side fast/slow path dispatch
   - V08: Three-way `KernelPath` enum
   - V09: `template<bool Diagonal>` compile-time specialization
   - Current: `cmb_shift`/`surface_shift` turn boundary skipping into loop-range adjustments

4. **Improve memory access patterns** for bandwidth efficiency.
   - V08: r-first thread mapping for coalesced global reads/writes
   - V10: r_passes amortizes shared memory loads over more work per thread

5. **Reduce floating-point work** per element.
   - V02: O(3+3) instead of O(3×3) dimi/dimj
   - V03: 1 quadrature point instead of 6
   - Current: FMA-friendly formulation maximizes hardware FMA throughput
