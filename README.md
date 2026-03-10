<img src="https://raw.githubusercontent.com/Quantum-Many-Body/MeanFieldTheories.jl/main/docs/src/logo.png" width="300">

# MeanFieldTheories.jl

**MeanFieldTheories.jl** is a Julia package for studying quantum many-body systems using mean-field theory and related methods. It provides a complete workflow from constructing many-body Hamiltonians to obtaining self-consistent ground states and calculating collective excitation spectra, covering methods such as Hartree-Fock (HF), Single-Mode Approximation (SMA) and Random Phase Approximation (RPA).

See documents: https://Quantum-Many-Body.github.io/MeanFieldTheories.jl

## Features

- **Fully customizable quantum system** Degrees of freedom (site, sublattice, spin, orbital, valley, …) are freely defined by the user via `SystemDofs`, with user-specified constraints.

-  **High flexibility for generating operator representations** DOF index constraints can be applied directly to `generate_onebody` and `generate_twobody` to select only the desired terms on each bond. 

-  **Highly free forms of interaction** Two-body interaction allows four different site index $(i,j,k,l)$. The creation-annihilation ordering of the operator string is also arbitrary and handled automatically.

- **Unrestricted Hartree-Fock in both real and momentum space.** All four Wick contraction channels (Hartree and Fock, both pairs) are kept open with no preset symmetry breaking.

- **Complete post-HF excitation spectrum.** On top of the mean-field ground state, collective modes are accessible via Single-Mode Approximation (SMA) and Random Phase Approximation (RPA), yielding dynamic structure factors and excitation gaps directly.

## Installation

```julia
using Pkg
Pkg.develop(url="https://github.com/Quantum-Many-Body/MeanFieldTheories.jl")
```

## Quick Start

### Real-Space Hartree-Fock Approximation (`solve_hfr`)

Hubbard model ($t=1$, $U=8$, half-filling on a 4×4 square lattice) solved by real-space Hartree-Fock.

```julia
using MeanFieldTheories

# Square lattice with periodic boundary conditions
unitcell = Lattice([Dof(:site, 1)], [QN(site=1)], [[0.0, 0.0]])
lattice  = Lattice(unitcell, [[1.0, 0.0], [0.0, 1.0]], (4, 4))
t = 1.0;  U = 8.0

# Operators (shared by all four cases below)
function make_ops(dofs)
    t_ops = generate_onebody(dofs, bonds(lattice, (:p, :p), 1),
        (delta, qn1, qn2) -> qn1.spin == qn2.spin ? -t : 0.0).ops
    # 1/2 Σ_i 2*U_{ii} n_i↑ * n_i↓: the 1/2 prefactor is built-in, so multiply by 2 to compensate
    U_ops = generate_twobody(dofs, bonds(lattice, (:p, :p), 0),
        (deltas, qn1, qn2, qn3, qn4) ->
            (qn1.spin, qn2.spin, qn3.spin, qn4.spin) == (1, 1, 2, 2) ? U : 0.0,
        order = (cdag, :i, c, :i, cdag, :i, c, :i)).ops
    vcat(t_ops, U_ops)
end

dofs = SystemDofs([Dof(:site, 16), Dof(:spin, 2, [:up, :down])])
result = solve_hfr(dofs, make_ops(dofs), [16])
```

Run log:
```
============================================================
Hartree-Fock SCF Solver
============================================================
[13:40:12] Building Hamiltonian  (144 operators)
               t matrix: (32, 32), nnz = 128      358.292μs
               U matrix: (1024, 1024), nnz = 64       124.417μs
  System: N = 32, blocks = 1, particles = [16] (total = 16)
  T = 0,  mixing = DIIS(m=8),  tol = 1e-08,  max_iter = 1000
============================================================
[13:40:12] G initialized    33.708μs
[13:40:12] Iter    1  res = 3.909e-03  E = -24.399771  NCond = 1.4220
[13:40:12] Iter    2  res = 3.631e-03  E = -49.605585  NCond = 16.0000
[13:40:12] Iter    3  res = 1.623e-03  E = +7.257954  NCond = 16.0000
[13:40:12] Iter    4  res = 1.389e-03  E = +3.645997  NCond = 16.0000
[13:40:12] Iter    5  res = 1.103e-03  E = +2.070912  NCond = 16.0000
[13:40:12] Iter   10  res = 3.143e-04  E = -13.013728  NCond = 16.0000
[13:40:12] Iter   20  res = 4.440e-05  E = -7.825859  NCond = 16.0000
[13:40:12] Iter   30  res = 7.137e-08  E = -7.389622  NCond = 16.0000
[13:40:12] Iter   32  res = 7.424e-09 < 1.000e-08  CONVERGED
============================================================
[13:40:12] SCF CONVERGED  (32 iterations)
  Band energy:        -1.0131544557
  Interaction energy: -6.3764731228
  Total energy:       -7.3896275785
  NCond:              16.000000
  Sz:                 -0.000040
  μ (block 1):       +3.9999998830

  ── Timing Summary ────────────────────────────────────────────
  Phase                      Total         Avg   Calls
  ────────────────────────────────────────────────────────
  build_T                358.292μs   358.292μs       1
  build_U                124.417μs   124.417μs       1
  initialize_green        33.708μs    33.708μs       1
  build_h_eff            192.080μs     6.002μs      32
  diagonalize              3.351ms   104.708μs      32
  update_green           130.665μs     4.083μs      32
  calc_energies           69.419μs     2.239μs      31
  ────────────────────────────────────────────────────────
  solve_hfr (total)        7.556ms     7.556ms       1
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

# 1/2Σ_{i≠j, σσ'} V_{ij} n_{iσ} n_{jσ'}
twobody = generate_twobody(dofs, nn_bonds,
    (deltas, qn1, qn2, qn3, qn4) ->
        qn1.spin == qn2.spin && qn3.spin == qn4.spin ? 4.0/2 : 0.0;
        order = (cdag, :i, c, :i, cdag, :j, c, :j))

ks = build_kpoints([[2.0, 0.0], [0.0, 2.0]], (2, 2))

result = solve_hfk(dofs, onebody, twobody, ks, 16)
```

Run log:
```
============================================================
Hartree-Fock SCF Solver (momentum space)
============================================================
  Nk = 4,  d = 8,  n_electrons = 16,  T = 0
  mixing = DIIS(m=8),  tol = 1e-08,  max_iter = 1000
============================================================
[13:52:05]  T(r): 6 terms   383.291μs
[13:52:05]  V(r): 5 triples   491.291μs
[13:52:05] G initialized   459.834μs
[13:52:05] Iter    1  res = 1.566e-02  E = -6.104777  NCond = 4.0000
[13:52:05] Iter   10  res = 6.553e-09 < 1.000e-08  CONVERGED
============================================================
[13:52:05] SCF CONVERGED  (10 iterations)
  Band energy:        -0.0095459283
  Interaction energy: -0.5570150688
  Total energy:       -0.5665609971
  NCond:              4.000000
  μ:                  +16.0000000000

  ── Timing Summary (k-space HF) ───────────────────────────
  Phase                        Total         Avg   Calls
  ──────────────────────────────────────────────────────────
  build_Tr                 383.291μs   383.291μs       1
  build_Tk                   7.000μs     7.000μs       1
  build_Vr                 491.291μs   491.291μs       1
  initialize_green_k       459.834μs   459.834μs       1
  build_heff_k               7.890ms   789.012μs      10
  diagonalize_k              4.702ms   470.216μs      10
  update_green_k             1.007ms   100.704μs      10
  calc_energies_k           89.709μs    44.854μs       2
  ──────────────────────────────────────────────────────────
  solve_hfk (total)         26.227ms    26.227ms       1
  ──────────────────────────────────────────────────────────
```

## Benchmark

### SDW-CDW Phase Diagram of Extended Hubbard Model

This benchmark reproduces the phase diagram of the extended Hubbard model on a 2D square lattice at half-filling:

$$H = -t \sum_{\langle ij \rangle,\sigma} c^\dagger_{i\sigma}c_{j\sigma} + U \sum_i n_{i\uparrow}n_{i\downarrow} + V \sum_{\langle ij \rangle} n_i n_j$$

The model parameters are $t=1$, $U=4$, with nearest-neighbor repulsion $V \in [0, 2]$. At half-filling, the system exhibits two distinct phases:
- **SDW/AFM phase** ($V/U \lesssim 1/4$): antiferromagnetic order with staggered magnetization $S(\pi,\pi) \neq 0$
- **CDW/CO phase** ($V/U \gtrsim 1/4$): charge density wave order with staggered charge density $N(\pi,\pi) \neq 0$

The calculation uses momentum-space unrestricted Hartree-Fock on a $2\times2$ magnetic unit cell with a $2\times2$ $k$-grid (4 $k$-points). For each $V$, SCF is initialized from two biased initial conditions (SDW and CDW), and the lower-energy converged state is taken as the ground state.

Run:
```
julia --project=benchmark benchmark/SDW_CDW/run.jl
```

Results:

![SDW_CDW](benchmark/SDW_CDW/sdw_cdw.png)

The calculated phase boundary at $V_c = U/4 = 1.0$ and the order parameter curves are in complete agreement with Fig. 5(b) of Ref. [1].

### Magnon Spectrum

This benchmark will compute the magnon excitation spectrum using SMA. (See Ref. [2] for the theoretical background.)

*Coming soon — to be added.*

## References

[1] T. Aoyama, K. Yoshimi, K. Ido, Y. Motoyama, T. Kawamura, T. Misawa, T. Kato, and A. Kobayashi, [H-wave – A Python package for the Hartree-Fock approximation and the random phase approximation](https://doi.org/10.1016/j.cpc.2024.109087), Computer Physics Communications 298, 109087 (2024).

[2] W.-X. Qiu and F. Wu, [Topological magnons and domain walls in twisted bilayer MoTe2](https://link.aps.org/doi/10.1103/sl5k-c825), Phys. Rev. B 112, 085132 (2025).


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
  url = {https://github.com/Quantum-Many-Body/MeanFieldTheories.jl}
}
```

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests on [GitHub](https://github.com/Quantum-Many-Body/MeanFieldTheories.jl).

## Contact

Yong-Yue Zong — [zongyongyue@gmail.com](mailto:zongyongyue@gmail.com)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
