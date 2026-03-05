"""
    MeanFieldTheories

A Julia package for quantum many-body systems using mean-field theory methods.

This package provides tools for:
- Defining quantum systems with arbitrary degrees of freedom
- Constructing lattice structures
- Generating operator terms with flexible orderings
- Building matrix and tensor representations
- Hartree-Fock calculations (ground state and excited states)

# Modules
- `quantumsystem`: Core quantum system definitions and operator algebra
- `groundstate`: Ground state calculations (Hartree-Fock, mean-field methods)
- `excitations`: Excited state calculations (RPA, TDHF, etc.)

# Example
```julia
using MeanFieldTheories

# Define system
dofs = SystemDofs(site=1:4, spin=[:up, :dn])
lattice = Lattice(:Square, 2, 2, pbc=true)

# Generate hopping terms
bonds = bonds(lattice, 1)
hopping = generate_onebody(dofs, bonds, -1.0).ops

# Build Hamiltonian
H = build_onebody_matrix(dofs, hopping)
```
"""
module MeanFieldTheories

# Export quantum system functionality
include("quantumsystem/freedom.jl")
include("quantumsystem/lattice.jl")
include("quantumsystem/operators.jl")

# Export ground state methods
include("groundstate/hartreefock_real.jl")
include("groundstate/hartreefock_momentum.jl")

# Export core types
export QuantumNumber, QN, Dof, SystemDofs
export Lattice, Bond
export Operator, FermionOp, Operators
export c, cdag

# Export lattice functions
export bonds, get_coordinate, is_positive_direction

# Export quantum number functions
export qn2linear, linear2qn
export ndofs, total_dim, site_spin_system

# Export operator generation functions
export generate_onebody, generate_twobody

# Export matrix/tensor builders
export build_onebody_matrix, build_interaction_tensor

# Export Hartree-Fock functions
export build_T, build_U, solve_hf#hartreefock_real
export build_Tr, build_Tk, build_Vr, build_Vk

# Export utility constants
export PRECISION, rd

end # module MeanFieldTheories
