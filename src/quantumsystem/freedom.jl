"""
Degree of Freedom Definitions for Quantum Systems

Provides types and functions for defining arbitrary quantum numbers
(site, spin, orbital, valley, layer, etc.) and their valid combinations.
"""

using LinearAlgebra

#==================== Global Precision Setting ====================#

"""
Global precision for numerical rounding (number of decimal places).
All floating-point values (coordinates, hopping amplitudes, interactions, phases, etc.)
are rounded to this precision to avoid floating-point comparison issues.
"""
const PRECISION = 6

"""
    rd(x) -> Number

Round a number to PRECISION decimal places. Works with Real and Complex.
"""
rd(x::Real) = round(x, digits=PRECISION)
rd(x::Complex) = Complex(rd(real(x)), rd(imag(x)))
rd(v::AbstractVector) = rd.(v)

#==================== Quantum Number Type ====================#

"""
    QuantumNumber{Names, Types}

A wrapper around NamedTuple for representing quantum numbers with cleaner display.

# Type Parameters
- `Names`: Tuple of field names (Symbols)
- `Types`: Tuple type for values

# Examples
```julia
# Create from keyword arguments
qn = QuantumNumber(site=1, spin=2)
qn.site  # 1
qn.spin  # 2

# Display: QN(site=1, spin=2)
```
"""
struct QuantumNumber{Names, Types<:Tuple}
    data::NamedTuple{Names, Types}

    # Inner constructor from NamedTuple
    function QuantumNumber{Names, Types}(nt::NamedTuple{Names, Types}) where {Names, Types<:Tuple}
        new{Names, Types}(nt)
    end
end

# Outer constructor from keyword arguments
QuantumNumber(; kwargs...) = QuantumNumber(NamedTuple(kwargs))

# Outer constructor from NamedTuple (generic)
QuantumNumber(nt::NamedTuple{Names, Types}) where {Names, Types<:Tuple} =
    QuantumNumber{Names, Types}(nt)

# Property access: qn.field -> qn.data.field
function Base.getproperty(qn::QuantumNumber, name::Symbol)
    if name === :data
        return getfield(qn, :data)
    else
        return getproperty(getfield(qn, :data), name)
    end
end

# Support tab completion
Base.propertynames(qn::QuantumNumber) = propertynames(getfield(qn, :data))

# Equality
Base.:(==)(a::QuantumNumber, b::QuantumNumber) = getfield(a, :data) == getfield(b, :data)
Base.hash(qn::QuantumNumber, h::UInt) = hash(getfield(qn, :data), h)

# Iteration and indexing (for compatibility)
Base.keys(qn::QuantumNumber) = keys(getfield(qn, :data))
Base.values(qn::QuantumNumber) = values(getfield(qn, :data))
Base.getindex(qn::QuantumNumber, key::Symbol) = getfield(qn, :data)[key]
Base.length(qn::QuantumNumber) = length(getfield(qn, :data))

# Display: QN(field1=val1, field2=val2)
function Base.show(io::IO, qn::QuantumNumber)
    print(io, "QN(")
    data = getfield(qn, :data)
    ks = keys(data)
    for (i, k) in enumerate(ks)
        i > 1 && print(io, ", ")
        print(io, k, "=", data[k])
    end
    print(io, ")")
end

# Convert to NamedTuple
Base.convert(::Type{NamedTuple}, qn::QuantumNumber) = getfield(qn, :data)
Base.NamedTuple(qn::QuantumNumber) = getfield(qn, :data)

# Type alias for convenience
const QN = QuantumNumber

#==================== Quantum Number Specification ====================#

"""
    Dof

Specification for a single degree of freedom (DOF).

# Fields
- `name::Symbol`: Name of the DOF (e.g., :site, :spin, :orbital)
- `size::Int`: Number of states for this DOF
- `labels::Union{Nothing, Vector}`: Optional labels for states (e.g., [:up, :down])
"""
struct Dof
    name::Symbol
    size::Int
    labels::Union{Nothing, Vector}

    function Dof(name::Symbol, size::Int, labels=nothing)
        if labels !== nothing && length(labels) != size
            error("Number of labels must match size")
        end
        new(name, size, labels)
    end
end

"""
    SystemDofs{Q}

Defines all degrees of freedom in the system, with optional constraints, custom ordering, and symmetry blocks.

# Type Parameter
- `Q`: QuantumNumber type with DOF names as keys

# Fields
- `dofs::Vector{Dof}`: List of degrees of freedom
- `valid_states::Vector{Q}`: All valid quantum number combinations (sorted)
- `blocks::Vector{UnitRange{Int}}`: Index ranges for symmetry blocks (single block [1:N] if no blocking)
- `qn_to_idx::Dict{Q, Int}`: Fast O(1) lookup from quantum numbers to linear indices

# Constructor
```julia
SystemDofs(dofs::Vector{Dof}; constraint=nothing, sortrule=collect(length(dofs):-1:1))
```

# Arguments
- `dofs`: Vector of Dof specifications
- `constraint`: Optional function `qn -> Bool` to filter valid states (not saved after construction)
- `sortrule`: Vector or nested vector specifying sort priority and blocking (not saved after construction)
  - Default: `[N, N-1, ..., 1]` (reverse order: first dof varies fastest, no blocking)
  - Elements are indices into `dofs` array
  - **Nested syntax for symmetry blocks**:
    - `[4, 3, 2, 1]` - No blocking
    - `[[4], 3, 2, 1]` - Block by 4th DOF (e.g., spin)
    - `[[4, 3], 2, 1]` - Block by 4th DOF, then by 3rd within each block
    - `[[4, 3, 2], 1]` - Block by 4th, 3rd, 2nd sequentially

# Design Rationale
- Default reverse order ensures first DOF (typically site) varies fastest for cache-friendly access
- Nested sortrule syntax naturally creates block-diagonal structure for conserved quantum numbers
- Blocks enable efficient matrix operations (multiplication, diagonalization) by exploiting sparsity

# Examples
```julia
# Example 1: Default ordering (reverse, no explicit blocks)
dofs = SystemDofs([
    Dof(:site, 3),
    Dof(:spin, 2)
])
# sortrule = [2, 1]: spin slow, site fast
# valid_states = [QN(site=1,spin=1), QN(site=2,spin=1), QN(site=3,spin=1),
#                 QN(site=1,spin=2), QN(site=2,spin=2), QN(site=3,spin=2)]
# blocks = [1:6]  (single block containing all states)

# Example 2: With symmetry blocks (spin conservation)
dofs = SystemDofs([
    Dof(:site, 3),
    Dof(:spin, 2)
], sortrule = [[2], 1])
# blocks = [1:3, 4:6]  (spin-up block, spin-down block)

# Example 3: Nested blocks (spin and valley conservation)
dofs = SystemDofs([
    Dof(:site, 4),
    Dof(:orbital, 2),
    Dof(:spin, 2),
    Dof(:valley, 2)
], sortrule = [[4, 3], 2, 1])
# blocks = [1:8, 9:16, 17:24, 25:32]  (K↑, K↓, K'↑, K'↓)

# Example 4: With constraint
dofs = SystemDofs([
    Dof(:site, 4),
    Dof(:spin, 2)
], constraint = qn -> qn.site <= 2, sortrule = [[2], 1])
# blocks = [1:2, 3:4]  (only first 2 sites, with spin blocks)
```
"""
struct SystemDofs{Q<:QuantumNumber}
    dofs::Vector{Dof}
    valid_states::Vector{Q}
    blocks::Vector{UnitRange{Int}}  # Always defined: single block [1:N] if no actual blocking
    qn_to_idx::Dict{Q, Int}
end

# Helper: Parse nested sortrule to extract blocking structure and flat order
function _parse_sortrule(sortrule::AbstractVector)
    flat_order = Int[]
    block_dofs = Vector{Int}[]  # Each element is a group of DOFs that define a block level

    for item in sortrule
        if item isa AbstractVector
            # Nested vector: defines blocking
            push!(block_dofs, collect(Int, item))
            append!(flat_order, item)
        else
            # Single integer: no blocking at this level
            push!(flat_order, item)
        end
    end

    return flat_order, block_dofs
end

# Helper: Generate blocks based on blocking DOFs
function _generate_blocks(valid_states::Vector{Q}, dofs::Vector{Dof}, block_dofs::Vector{Vector{Int}}) where {Q<:QuantumNumber}
    # If no blocking DOFs specified, return a single block containing all states
    if isempty(block_dofs)
        return [1:length(valid_states)]
    end

    # Group states by the first level of blocking DOFs
    first_block_dofs = block_dofs[1]
    blocks = UnitRange{Int}[]

    # Create a key for each state based on blocking DOFs
    state_groups = Dict{Any, Vector{Int}}()
    for (i, qn) in enumerate(valid_states)
        key = tuple([qn[dofs[dof_idx].name] for dof_idx in first_block_dofs]...)
        if !haskey(state_groups, key)
            state_groups[key] = Int[]
        end
        push!(state_groups[key], i)
    end

    # Sort groups by their keys and create ranges
    sorted_keys = sort(collect(keys(state_groups)))
    for key in sorted_keys
        indices = state_groups[key]
        # Check that indices are contiguous
        @assert indices == collect(minimum(indices):maximum(indices)) "Blocking structure requires contiguous indices. Check your sortrule."
        push!(blocks, minimum(indices):maximum(indices))
    end

    # TODO: Handle nested blocking (block_dofs[2:end]) if needed
    # For now, we only support one level of blocking

    return blocks
end

# Constructor with keyword arguments
function SystemDofs(
    dofs::Vector{Dof};
    constraint::Union{Nothing, Function} = nothing,
    sortrule = collect(length(dofs):-1:1)  # Accept both Vector{Int} and nested vectors
)
    # Parse sortrule to get flat order and blocking structure
    flat_order, block_dofs = _parse_sortrule(sortrule)

    # Validate flat order
    @assert length(flat_order) == length(dofs) "sortrule must cover all dofs"
    @assert sort(flat_order) == collect(1:length(dofs)) "sortrule must be a permutation of 1:$(length(dofs))"

    # Generate all valid states
    names = Tuple(dof.name for dof in dofs)
    Q = QuantumNumber{names, NTuple{length(dofs), Int}}
    valid_states = Q[]
    _enumerate_states!(valid_states, dofs, names, Int[], 1, constraint)

    # Sort according to flat order
    sort!(valid_states, by = qn -> tuple([qn[dofs[i].name] for i in flat_order]...))

    # Generate blocks if blocking DOFs are specified
    blocks = _generate_blocks(valid_states, dofs, block_dofs)

    # Build index lookup dictionary
    qn_to_idx = Dict(qn => i for (i, qn) in enumerate(valid_states))

    SystemDofs{Q}(dofs, valid_states, blocks, qn_to_idx)
end

# Helper: recursively enumerate states (unified for with/without constraint)
function _enumerate_states!(result::Vector{Q}, dofs::Vector{Dof}, names::Tuple,
                            current::Vector{Int}, depth::Int,
                            constraint::Union{Nothing, Function}) where {Q<:QuantumNumber}
    if depth > length(dofs)
        nt = NamedTuple{names}(Tuple(current))
        qn = QuantumNumber(nt)
        if constraint === nothing || constraint(qn)
            push!(result, qn)
        end
        return
    end

    for i in 1:dofs[depth].size
        push!(current, i)
        _enumerate_states!(result, dofs, names, current, depth + 1, constraint)
        pop!(current)
    end
end

# Accessor functions
"""
    ndofs(dofs::SystemDofs) -> Int

Return number of degrees of freedom.
"""
ndofs(d::SystemDofs) = length(d.dofs)

"""
    total_dim(dofs::SystemDofs) -> Int

Return total dimension (number of valid states).
"""
total_dim(d::SystemDofs) = length(d.valid_states)

# Convenience constructors
"""
    site_spin_system(Nsite::Int) -> SystemDofs

Create a simple site+spin system (standard case).
"""
function site_spin_system(Nsite::Int)
    return SystemDofs([
        Dof(:site, Nsite),
        Dof(:spin, 2, [:up, :down])
    ])
end

"""
    site_orbital_spin_system(Nsite::Int, Norbital::Int) -> SystemDofs

Create a site+orbital+spin system (multi-orbital case).
"""
function site_orbital_spin_system(Nsite::Int, Norbital::Int)
    return SystemDofs([
        Dof(:site, Nsite),
        Dof(:orbital, Norbital),
        Dof(:spin, 2, [:up, :down])
    ])
end

#==================== SystemDofs Display ====================#

"""
    Base.show(io::IO, dofs::SystemDofs)

Custom display for SystemDofs in REPL.
"""
function Base.show(io::IO, dofs::SystemDofs)
    n = length(dofs.dofs)
    dim = length(dofs.valid_states)

    println(io, "SystemDofs with $n degree(s) of freedom:")
    for (i, dof) in enumerate(dofs.dofs)
        print(io, "  $i. $(dof.name): $(dof.size) states")
        if dof.labels !== nothing
            print(io, " [", join(dof.labels, ", "), "]")
        end
        println(io)
    end
    print(io, "Total dimension: $dim")
end

#==================== Quantum Number Index Functions ====================#

"""
    qn2linear(dofs::SystemDofs, qn) -> Int

Convert quantum numbers to linear index (position in valid_states).
Uses O(1) hash table lookup for fast performance.
Accepts QuantumNumber or NamedTuple.

# Examples
```julia
dofs = SystemDofs([Dof(:site, 4), Dof(:spin, 2)])
idx = qn2linear(dofs, QN(site=1, spin=2))  # O(1) lookup
idx = qn2linear(dofs, (site=1, spin=2))    # also works
```
"""
function qn2linear(dofs::SystemDofs, qn::QuantumNumber)
    idx = get(dofs.qn_to_idx, qn, nothing)
    if isnothing(idx)
        error("Quantum numbers $qn not in valid states")
    end
    return idx
end

# Also accept NamedTuple for convenience
function qn2linear(dofs::SystemDofs, qn::NamedTuple)
    return qn2linear(dofs, QuantumNumber(qn))
end

"""
    linear2qn(dofs::SystemDofs, idx::Int) -> QuantumNumber

Convert linear index back to quantum numbers.

# Examples
```julia
dofs = site_spin_system(4)
qn = linear2qn(dofs, 2)  # returns QN(site=1, spin=2)
```
"""
function linear2qn(dofs::SystemDofs, idx::Int)
    return dofs.valid_states[idx]
end
