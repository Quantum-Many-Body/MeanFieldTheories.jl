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

### Real-Space Hartree-Fock Approximation (`solve_hfr`)

Hubbard model ($t=1$, $U=4$, half-filling on a 4×4 square lattice) solved by real-space Hartree-Fock. 

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
        order = (cdag, :i, c, :i, cdag, :i, c, :i)).ops
    vcat(t_ops, U_ops)
end
```

```julia
dofs = SystemDofs([Dof(:site, 16), Dof(:spin, 2, [:up, :down])])
result = solve_hfr(dofs, make_ops(dofs), [16])
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
  solve_hfr (total)         1.879 s      1.879 s        1
  ────────────────────────────────────────────────────────
```
### Momentum-Space Hartree-Fock Approximation (`solve_hfk`)

t-V model ($$t=1$$, $$V=4$$ on  a 4×4 square lattice) solved by momentum-space Hartree-Fock. 

```julia
using MeanFieldTheories

dofs = SystemDofs([Dof(:site, 4), Dof(:spin, 2, [:up, :dn])])

unitcell = Lattice([Dof(:site, 4)],
                     [QN(site=1), QN(site=2), QN(site=3), QN(site=4)],
                     [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
                     supercell_vectors=[[2.0, 0.0], [0.0, 2.0]])

nn_bonds = bonds(unitcell, (:p, :p), 1)

onebody = generate_onebody(dofs, nn_bonds,
    (delta, qn1, qn2) -> qn1.spin == qn2.spin ? -1.0 : 0.0)

# V Σ_{i≠j, σσ'} n_{iσ} n_{jσ'} 
twobody = generate_twobody(dofs, nn_bonds,
    (deltas, qn1, qn2, qn3, qn4) ->
        qn1.spin == qn2.spin && qn3.spin == qn4.spin ? 4.0 : 0.0)

ks = build_kpoints([[2.0, 0.0], [0.0, 2.0]], (2, 2))

result = solve_hfk(dofs, onebody, twobody, ks, 16)
```

```
============================================================
Hartree-Fock SCF Solver (momentum space)
============================================================
  Nk = 4,  d = 8,  n_electrons = 16,  T = 0
  mixing = DIIS(m=8),  tol = 1e-08,  max_iter = 1000
============================================================
[23:25:44]  T(r): 6 terms   357.584μs
[23:25:44]  V(r): 5 triples   557.208μs
[23:25:44] G initialized    39.417μs
[23:25:44] Iter    1  res = 1.563e-02  E = -6.193238  NCond = 4.0000
[23:25:44] Iter   10  res = 3.392e-09 < 1.000e-08  CONVERGED
============================================================
[23:25:44] SCF CONVERGED  (10 iterations)
  Band energy:        -0.0095358050
  Interaction energy: -0.5570200781
  Total energy:       -0.5665558830
  NCond:              4.000000
  μ:                  +16.0000000000

  ── Timing Summary (k-space HF) ───────────────────────────
  Phase                        Total         Avg   Calls
  ──────────────────────────────────────────────────────────
  build_Tr                 357.584μs   357.584μs       1
  build_Tk                   7.625μs     7.625μs       1
  build_Vr                 557.208μs   557.208μs       1
  initialize_green_k        39.417μs    39.417μs       1
  build_heff_k               4.792ms   479.237μs      10
  diagonalize_k              4.512ms   451.200μs      10
  update_green_k           643.875μs    64.387μs      10
  calc_energies_k           56.251μs    28.125μs       2
  ──────────────────────────────────────────────────────────
  solve_hfk (total)         18.799ms    18.799ms       1
  ──────────────────────────────────────────────────────────
```

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

## Contact

Yong-Yue Zong — [zongyongyue@gmail.com](mailto:zongyongyue@gmail.com)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
