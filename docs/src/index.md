<img src="https://raw.githubusercontent.com/ZongYongyue/SingleModeApproximation.jl/main/docs/src/logo.png" width="400">

# SingleModeApproximation.jl

**SingleModeApproximation.jl** is a Julia package for studying quantum many-body systems using the single-mode approximation (SMA) method. It provides a complete workflow from constructing many-body Hamiltonians to calculating collective excitations such as magnons, excitons and so on.

## Overview

The package implements a systematic approach to quantum many-body problems:

1. **Hamiltonian Construction**: Build many-body Hamiltonians with arbitrary degrees of freedom (spin, orbital, sublattice, valley, etc.) on various lattice geometries
2. **Hartree-Fock Approximation**: Obtain self-consistent ground states through mean-field treatment and variational minimization
3. **Single-Mode Approximation**: Calculate excitation spectra using momentum-dependent collective operators

This workflow follows the standard approach used in modern condensed matter physics, as exemplified in studies of topological magnons, spin-wave excitations, and collective modes in quantum materials.

See documents: https://zongyongyue.github.io/SingleModeApproximation.jl

## The Single-Mode Approximation Method

The single-mode approximation, introduced by Feynman (1954) for superfluid helium, is a powerful technique for studying collective excitations in interacting quantum systems. The key idea is to construct momentum-dependent operators that create elementary excitations from the ground state:

```
|k⟩ = Q†_k |Ψ₀⟩
```

where `Q†_k` is a collective operator (e.g., spin-flip operator for magnons) and `|Ψ₀⟩` is the Hartree-Fock ground state. The excitation energies are obtained by solving the eigenvalue problem in the subspace of single-particle-hole excitations, often combined with the Random Phase Approximation (RPA) to include correlation effects beyond mean-field.

### Physical Applications

This approach is particularly useful for calculating:

- **Magnon spectra** in magnetic systems (spin waves)
- **Exciton spectra** in semiconductors and insulators
- **Plasmon spectra** in metallic systems
- **Collective modes** in topological materials

For example, in the context of twisted bilayer MoTe₂, the workflow proceeds as:
- Single-particle Hamiltonian (tight-binding model with spin-orbit coupling)
- Mean-field calculation (Hartree-Fock self-consistency)
- Magnon spectrum (RPA excitations from the magnetic ground state)

## Key Features

- **Flexible quantum system definitions** with arbitrary degrees of freedom
- **Lattice structures** supporting square, honeycomb, triangular, kagome, and custom geometries
- **Symbolic operator algebra** with automatic fermionic anticommutation
- **High-level term generators** for hopping, Coulomb, Hund's coupling, exchange, pair hopping, and Ising interactions
- **Matrix and tensor builders** for efficient numerical calculations
- **Hartree-Fock solver** for ground and excited states
- **RPA/TDHF methods** for collective excitations

## Installation

```julia
using Pkg
Pkg.add("SingleModeApproximation")
```

Or for development:

```julia
using Pkg
Pkg.develop(url="https://github.com/zongyy/SingleModeApproximation.jl")
```

## Quick Start

```julia
using SingleModeApproximation

# Define system with 4 sites and spin
dofs = SystemDofs(site=1:4, spin=[:up, :dn])

# Create a square lattice
lattice = Lattice(:Square, 2, 2, pbc=true)

# Generate nearest-neighbor hopping
bonds_nn = bonds(lattice, 1)
hopping = generate_onebody(dofs, bonds_nn, -1.0)

# Build Hamiltonian matrix
H = build_onebody_matrix(dofs, hopping)

# Diagonalize to get single-particle spectrum
using LinearAlgebra
eigenvalues = eigvals(Hermitian(H))
```

## Documentation Contents

```@contents
Pages = ["quantumsystem.md", "hartreefock_real.md", "hartreefock_momentum.md", "singlemode.md"]
Depth = 2
```

## Package Structure

- **quantumsystem**: Core quantum system definitions, lattice structures, and operator algebra
- **groundstate**: Ground state calculations using Hartree-Fock and mean-field methods
- **excitations**: Excited state calculations using RPA, TDHF, and single-mode approximation

## Citation

If you use this package in your research, please cite:

```bibtex
@software{singlemodeapproximation,
  author = {Yong-Yue Zong},
  title = {SingleModeApproximation.jl: A Julia package for quantum many-body systems},
  year = {2025},
  url = {https://github.com/zongyy/SingleModeApproximation.jl}
}
```

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests on [GitHub](https://github.com/zongyy/SingleModeApproximation.jl).

## License

This project is licensed under the MIT License - see the LICENSE file for details.
