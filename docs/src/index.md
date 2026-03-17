```@raw html
<img src="https://raw.githubusercontent.com/Quantum-Many-Body/MeanFieldTheories.jl/main/docs/src/logo.png" width="300">
```

# MeanFieldTheories.jl

**MeanFieldTheories.jl** is a Julia package for studying quantum many-body systems using mean-field theory and related methods. It provides a complete workflow from constructing many-body Hamiltonians to obtaining self-consistent ground states and calculating collective excitation spectra.

See documents: https://Quantum-Many-Body.github.io/MeanFieldTheories.jl

## Features

- **Fully customizable quantum system** Degrees of freedom (site, sublattice, spin, orbital, valley, …) are freely defined by the user via `SystemDofs`, with user-specified constraints.

-  **High flexibility for generating operator representations** DOF index constraints can be applied directly to `generate_onebody` and `generate_twobody` to select only the desired terms on each bond.

-  **Highly free forms of interaction** Two-body interaction allows four different site index $(i,j,k,l)$. The creation-annihilation ordering of the operator string is also arbitrary and handled automatically.

- **Unrestricted Hartree-Fock in both real and momentum space.** All four Wick contraction channels (Hartree and Fock, both pairs) are kept open with no preset symmetry breaking.

- **Complete post-HF excitation spectrum.** On top of the mean-field ground state, collective modes are accessible, yielding dynamic structure factors and excitation gaps directly.

## Documentation Contents

```@contents
Pages = [
    "quantumsystem.md",
    "hartreefock_intro.md",
    "hartreefock_real.md",
    "hartreefock_momentum.md",
    "collective_excitations.md",
    "magnon.md",
    "plot.md",
    "SDW_CDW.md",
    "SM_AFM.md",
    "Cubic_AFM.md",
    "KMH.md",
    "api.md",
]
Depth = 2
```