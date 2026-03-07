<img src="https://raw.githubusercontent.com/ZongYongyue/MeanFieldTheories.jl/main/docs/src/logo.png" width="300">

# MeanFieldTheories.jl

**MeanFieldTheories.jl** is a Julia package for studying quantum many-body systems using mean-field theory and related methods. It provides a complete workflow from constructing many-body Hamiltonians to obtaining self-consistent ground states and calculating collective excitation spectra, covering methods such as Hartree-Fock (HF), Single-Mode Approximation (SMA) and Random Phase Approximation (RPA).

See documents: https://ZongYongyue.github.io/MeanFieldTheories.jl

## Features

- **Fully customizable quantum system** Degrees of freedom (site, sublattice, spin, orbital, valley, …) are freely defined by the user via `SystemDofs`, with user-specified constraints.

-  **High flexibility for generating operator representations** DOF index constraints can be applied directly to `generate_onebody` and `generate_twobody` to select only the desired terms on each bond. 

-  **Highly free forms of interaction** Two-body interaction allows four different site index $(i,j,k,l)$. The creation-annihilation ordering of the operator string is also arbitrary and handled automatically.

- **Unrestricted Hartree-Fock in both real and momentum space.** All four Wick contraction channels (Hartree and Fock, both pairs) are kept open with no preset symmetry breaking.

- **Complete post-HF excitation spectrum.** On top of the mean-field ground state, collective modes are accessible via Single-Mode Approximation (SMA) and Random Phase Approximation (RPA), yielding dynamic structure factors and excitation gaps directly.

## Installation

```julia
using Pkg
Pkg.develop(url="https://github.com/ZongYongyue/MeanFieldTheories.jl")
```

## Quick Start

Square-lattice Hubbard model ($t=1$, $U=4$, half-filling on a 4×4 lattice) solved by real-space Hartree-Fock. The four cases below demonstrate the trade-off between spin-block structure, memory efficiency, and convergence to the global minimum.

```julia
using MeanFieldTheories

# Square lattice with periodic boundary conditions
unitcell = Lattice([Dof(:site, 1)], [QN(site=1)], [[0.0, 0.0]])
lattice  = Lattice(unitcell, [[1.0, 0.0], [0.0, 1.0]], (4, 4))
t = 1.0;  U = 4.0

# Operators (shared by all four cases below)
function make_ops(dofs)
    t_ops = generate_onebody(dofs, bonds(lattice, (:p, :p), 1),
        (delta, qn1, qn2) -> qn1.spin == qn2.spin ? -t : 0.0).ops
    U_ops = generate_twobody(dofs, bonds(lattice, (:p, :p), 0),
        (deltas, qn1, qn2, qn3, qn4) ->
            (qn1.spin, qn2.spin, qn3.spin, qn4.spin) == (1, 1, 2, 2) ? U : 0.0,
        order = (cdag, 1, c, 1, cdag, 1, c, 1)).ops
    vcat(t_ops, U_ops)
end
```

**Case 1 — spin blocks, single run:**

```julia
dofs_b = SystemDofs([Dof(:site, 16), Dof(:spin, 2, [:up, :down])], sortrule = [[2], 1])
result = solve_hf(dofs_b, make_ops(dofs_b), [8, 8])
```

```
============================================================
Hartree-Fock SCF Solver
============================================================
Building Hamiltonian  (144 operators)
           t matrix: (32, 32), nnz = 128      243.021ms
           U matrix: (1024, 1024), nnz = 32        28.811ms
System: N = 32, blocks = 2, particles = [8, 8] (total = 16)
T = 0,  mixing = DIIS(m=8),  tol = 1e-08,  max_iter = 1000
============================================================
G initialized   139.263ms
Iter    1  res = 7.805e-03  E = -24.155896  NCond = 1.4441
Iter    2  res = 7.185e-03  E = -36.510383  NCond = 16.0000
Iter    3  res = 3.212e-03  E = -9.235923   NCond = 16.0000
Iter    4  res = 1.971e-03  E = -9.709626   NCond = 16.0000
Iter    5  res = 1.767e-03  E = -9.770197   NCond = 16.0000
Iter   10  res = 5.730e-04  E = -10.546756  NCond = 16.0000
Iter   20  res = 3.884e-06  E = -10.114867  NCond = 16.0000
Iter   30  res = 2.159e-07  E = -10.112959  NCond = 16.0000
Iter   40  res = 1.055e-07  E = -10.112967  NCond = 16.0000
Iter   50  res = 1.544e-08  E = -10.112965  NCond = 16.0000
Iter   53  res = 9.972e-09 < 1.000e-08  CONVERGED
============================================================
SCF CONVERGED  (53 iterations)
  Band energy:        +2.3594869312
  Interaction energy: -12.4724545253
  Total energy:       -10.1129675941
  NCond:              16.000000
  Sz:                 -0.000000
  μ (block 1):       +2.0000000007
  μ (block 2):       +1.9999999993

  ── Timing Summary ────────────────────────────────────────────
  Phase                      Total         Avg   Calls
  ────────────────────────────────────────────────────────
  build_T                243.021ms   243.021ms       1
  build_U                 28.811ms    28.811ms       1
  initialize_green       139.263ms   139.263ms       1
  build_h_eff             71.039ms     1.340ms      53
  diagonalize              7.170ms   135.273μs      53
  update_green           761.247μs    14.363μs      53
  calc_energies          159.287μs     3.063μs      52
  ────────────────────────────────────────────────────────
  solve_hf (total)         1.927 s      1.927 s        1
  ────────────────────────────────────────────────────────
```

**Case 2 — spin blocks, 10 random restarts:**

```julia
result = solve_hf(dofs_b, make_ops(dofs_b), [8, 8], n_restarts = 10)
```

```
============================================================
Hartree-Fock SCF Solver
============================================================
Building Hamiltonian  (144 operators)
           t matrix: (32, 32), nnz = 128      255.201ms
           U matrix: (1024, 1024), nnz = 32        32.010ms
System: N = 32, blocks = 2, particles = [8, 8] (total = 16)
T = 0,  mixing = DIIS(m=8),  tol = 1e-08,  max_iter = 1000
Restarts: 10
============================================================
  Restart  1: E = -10.7471610309  (CONVERGED, 20 iters)
  Restart  2: E = -10.1129683059  (CONVERGED, 44 iters)
  Restart  3: E = -10.7471659564  (CONVERGED, 24 iters)
  Restart  4: E = -10.3525486035  (CONVERGED, 37 iters)
  Restart  5: E = -10.7471668476  (CONVERGED, 24 iters)
  Restart  6: E = -10.7471664768  (CONVERGED, 24 iters)
  Restart  7: E = -12.5665533115  (CONVERGED, 16 iters)
  Restart  8: E = -10.7471662060  (CONVERGED, 21 iters)
  Restart  9: E = -10.1129664405  (CONVERGED, 48 iters)
  Restart 10: E = -10.0596092120  (CONVERGED, 112 iters)
============================================================
Best result from 10 restarts:
SCF CONVERGED  (16 iterations)
  Band energy:        -4.5074893226
  Interaction energy: -8.0590639888
  Total energy:       -12.5665533115
  NCond:              16.000000
  Sz:                 +0.000000
  μ (block 1):       +2.0000000357
  μ (block 2):       +1.9999999632

  ── Timing Summary ────────────────────────────────────────────
  Phase                      Total         Avg   Calls
  ────────────────────────────────────────────────────────
  build_T                255.201ms   255.201ms       1
  build_U                 32.010ms    32.010ms       1
  initialize_green       141.842ms    14.184ms      10
  build_h_eff            918.422μs     2.482μs     370
  diagonalize             18.867ms    50.990μs     370
  update_green             1.137ms     3.074μs     370
  calc_energies          715.081μs     1.986μs     360
  ────────────────────────────────────────────────────────
  solve_hf (total)         1.993 s      1.993 s        1
  ────────────────────────────────────────────────────────
```

**Case 3 — no block structure, single run:**

```julia
dofs_nb = SystemDofs([Dof(:site, 16), Dof(:spin, 2, [:up, :down])])
result = solve_hf(dofs_nb, make_ops(dofs_nb), [16])
```

```
============================================================
Hartree-Fock SCF Solver
============================================================
Building Hamiltonian  (144 operators)
           t matrix: (32, 32), nnz = 128      213.982ms
           U matrix: (1024, 1024), nnz = 64        28.943ms
System: N = 32, blocks = 1, particles = [16] (total = 16)
T = 0,  mixing = DIIS(m=8),  tol = 1e-08,  max_iter = 1000
============================================================
G initialized   144.500ms
Iter    1  res = 3.908e-03  E = -24.191984  NCond = 1.4230
Iter    2  res = 3.601e-03  E = -36.222279  NCond = 16.0000
Iter    3  res = 1.402e-03  E = -8.867481   NCond = 16.0000
Iter    4  res = 8.227e-04  E = -10.085860  NCond = 16.0000
Iter    5  res = 6.719e-04  E = -10.538543  NCond = 16.0000
Iter   10  res = 1.618e-04  E = -13.119825  NCond = 16.0000
Iter   19  res = 5.990e-09 < 1.000e-08  CONVERGED
============================================================
SCF CONVERGED  (19 iterations)
  Band energy:        -4.5074871150
  Interaction energy: -8.0590647077
  Total energy:       -12.5665518227
  NCond:              16.000000
  Sz:                 -0.000005
  μ (block 1):       +1.9999997617

  ── Timing Summary ────────────────────────────────────────────
  Phase                      Total         Avg   Calls
  ────────────────────────────────────────────────────────
  build_T                213.982ms   213.982ms       1
  build_U                 28.943ms    28.943ms       1
  initialize_green       144.500ms   144.500ms       1
  build_h_eff             49.124μs     2.585μs      19
  diagonalize              2.088ms   109.899μs      19
  update_green           106.000μs     5.578μs      19
  calc_energies           53.664μs     2.981μs      18
  ────────────────────────────────────────────────────────
  solve_hf (total)         1.879 s      1.879 s        1
  ────────────────────────────────────────────────────────
```

**Case 4 — no block structure, 10 random restarts:**

```julia
result = solve_hf(dofs_nb, make_ops(dofs_nb), [16], n_restarts = 10)
```

```
============================================================
Hartree-Fock SCF Solver
============================================================
Building Hamiltonian  (144 operators)
           t matrix: (32, 32), nnz = 128      229.196ms
           U matrix: (1024, 1024), nnz = 64        29.206ms
System: N = 32, blocks = 1, particles = [16] (total = 16)
T = 0,  mixing = DIIS(m=8),  tol = 1e-08,  max_iter = 1000
Restarts: 10
============================================================
  Restart  1: E = -10.4979817905  (CONVERGED, 129 iters)
  Restart  2: E = -12.5665594104  (CONVERGED, 17 iters)
  Restart  3: E = -12.5665576904  (CONVERGED, 21 iters)
  Restart  4: E = -12.5665597585  (CONVERGED, 19 iters)
  Restart  5: E = -12.5665661434  (CONVERGED, 22 iters)
  Restart  6: E = -12.5665486867  (CONVERGED, 26 iters)
  Restart  7: E = -10.2971741891  (CONVERGED, 172 iters)
  Restart  8: E = -12.5665534178  (CONVERGED, 22 iters)
  Restart  9: E = -11.4002111362  (CONVERGED, 57 iters)
  Restart 10: E = -12.5665474992  (CONVERGED, 15 iters)
============================================================
Best result from 10 restarts:
SCF CONVERGED  (22 iterations)
  Band energy:        -4.5075083501
  Interaction energy: -8.0590577933
  Total energy:       -12.5665661434
  NCond:              16.000000
  Sz:                 -0.000009
  μ (block 1):       +1.9999994130

  ── Timing Summary ────────────────────────────────────────────
  Phase                      Total         Avg   Calls
  ────────────────────────────────────────────────────────
  build_T                229.196ms   229.196ms       1
  build_U                 29.206ms    29.206ms       1
  initialize_green       151.063ms    15.106ms      10
  build_h_eff              1.283ms     2.566μs     500
  diagonalize             51.957ms   103.914μs     500
  update_green             2.222ms     4.443μs     500
  calc_energies            1.041ms     2.124μs     490
  ────────────────────────────────────────────────────────
  solve_hf (total)         2.109 s      2.109 s        1
  ────────────────────────────────────────────────────────
```

> **NOTE** Declaring spin as a symmetry block (Cases 1–2 vs 3–4) halves the number of non-zeros in the interaction matrix (nnz 64 → 32), which reduces memory and speeds up each SCF iteration. However, this comes with an important caveat: enforcing spin conservation at the density-matrix level means the Green's function is constrained to be strictly block-diagonal throughout the iteration — off-diagonal spin correlations are permanently forbidden, not just absent at convergence. As a result, the blocked solver cannot use spin-mixed intermediate configurations as transient pathways to escape local minima in the energy landscape. The unblocked solver, by contrast, has full freedom in the density matrix during the SCF and can reach the global minimum via spin-mixed intermediates. The multi-restart statistics make this concrete: in 10 runs the unblocked solver finds the global minimum ($E \approx -12.57$) 7 out of 10 times (Case 4), while the blocked solver finds it only 1 out of 10 times (Case 2). The practical recommendation is to use block structure for efficiency, but always pair it with `n_restarts` to compensate for the reduced basin of attraction of the global minimum.

## Package Structure

- **quantumsystem**: Core quantum system definitions, lattice structures, and operator algebra
- **groundstate**: Ground state calculations using Hartree-Fock and mean-field methods
- **excitations**: Excited state calculations using RPA, TDHF, and single-mode approximation

## Citation

If you use this package in your research, please cite:

```bibtex
@software{meanfieldtheories,
  author = {Yong-Yue Zong},
  title = {MeanFieldTheories.jl: A Julia package for quantum many-body systems using mean-field theory and related methods},
  year = {2026},
  url = {https://github.com/ZongYongyue/MeanFieldTheories.jl}
}
```

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests on [GitHub](https://github.com/ZongYongyue/MeanFieldTheories.jl).

## License

This project is licensed under the MIT License - see the LICENSE file for details.
