![MeanFieldTheories.jl Logo](assets/logo.png)

# MeanFieldTheories.jl

**MeanFieldTheories.jl** is a Julia package for studying quantum many-body systems using mean-field theory and related methods. It provides a complete workflow from constructing many-body Hamiltonians to obtaining self-consistent ground states and calculating collective excitation spectra, covering methods such as Hartree-Fock (HF), the Random Phase Approximation (RPA), and the Single-Mode Approximation (SMA).

## Overview

The package implements a systematic approach to quantum many-body problems:

1. **Hamiltonian Construction**: Build many-body Hamiltonians with arbitrary degrees of freedom (spin, orbital, sublattice, valley, etc.) on various lattice geometries
2. **Hartree-Fock Approximation**: Obtain self-consistent ground states through mean-field treatment and variational minimization
3. **Single-Mode Approximation**: Calculate excitation spectra using momentum-dependent collective operators

This workflow follows the standard approach used in modern condensed matter physics, as exemplified in studies of topological magnons, spin-wave excitations, and collective modes in quantum materials.

See documents: https://zongyongyue.github.io/MeanFieldTheories.jl


## Installation

```julia
using Pkg
Pkg.add("MeanFieldTheories")
```

Or for development:

```julia
using Pkg
Pkg.develop(url="https://github.com/ZongYongyue/MeanFieldTheories.jl")
```

## Quick Start

```julia
using MeanFieldTheories

# Define system with 4 sites and spin
dofs = SystemDofs(site=1:4, spin=[:up, :dn])

# Create a square lattice
lattice = Lattice(:Square, 2, 2, pbc=true)

# Generate nearest-neighbor hopping
bonds_nn = bonds(lattice, 1)
hopping = generate_onebody(dofs, bonds_nn, -1.0).ops

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
@software{meanfieldtheories,
  author = {Yong-Yue Zong},
  title = {MeanFieldTheories.jl: A Julia package for quantum many-body systems},
  year = {2025},
  url = {https://github.com/ZongYongyue/MeanFieldTheories.jl}
}
```

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests on [GitHub](https://github.com/ZongYongyue/MeanFieldTheories.jl).

## License

This project is licensed under the MIT License - see the LICENSE file for details.
