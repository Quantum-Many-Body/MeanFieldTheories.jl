"""
Lattice Structures for Quantum Systems

Provides types and functions for defining lattice structures,
coordinate mappings, and bond generation for many-body systems.
"""

#==================== Lattice Specification ====================#

"""
    Lattice{D, Q, T}

Represents a lattice structure mapping position states to real-space coordinates.

# Type Parameters
- `D`: Spatial dimension (e.g., 2 for 2D, 3 for 3D)
- `Q`: QuantumNumber type for position states
- `T`: Coordinate element type (subtype of Real)

# Fields
- `position_dofs::Vector{Dof}`: Degrees of freedom that define positions
- `position_states::Vector{Q}`: All valid position states
- `coordinates::Vector{SVector{D,T}}`: Coordinate for each position state (length D)
- `supercell_vectors::Union{Nothing, Vector{SVector{D,T}}}`: Supercell vectors for PBC (optional)

# Notes
- `position_states[i]` corresponds to `coordinates[i]`
- `supercell_vectors` is set automatically when using tiling constructor

# Examples
```julia
# 2D honeycomb lattice with 16 unit cells and 2 sublattices
position_dofs = [Dof(:moire_cell, 16), Dof(:sublattice, 2, [:A, :B])]
position_states = [QN(moire_cell=1, sublattice=1), QN(moire_cell=1, sublattice=2), ...]
coordinates = [[0.0, 0.0], [1.0, 0.0], ...]

lattice = Lattice(position_dofs, position_states, coordinates)
```
"""
struct Lattice{D, Q<:QuantumNumber, T<:Real}
    position_dofs::Vector{Dof}
    position_states::Vector{Q}
    coordinates::Vector{SVector{D,T}}
    supercell_vectors::Union{Nothing, Vector{SVector{D,T}}}
end

# Public constructor: accepts Vector{Vector{T}} and converts to SVector (backward compatible)
function Lattice(
    position_dofs::Vector{Dof},
    position_states::Vector{Q},
    coordinates::Vector{Vector{T}};
    supercell_vectors::Union{Nothing, Vector{Vector{T}}} = nothing
) where {Q<:QuantumNumber, T<:Real}
    D = length(coordinates[1])
    sv_coords    = SVector{D,T}.(coordinates)
    sv_supercell = isnothing(supercell_vectors) ? nothing : SVector{D,T}.(supercell_vectors)
    Lattice{D,Q,T}(position_dofs, position_states, sv_coords, sv_supercell)
end

# Internal constructor: already-converted SVectors (used by tiling constructor)
function Lattice(
    position_dofs::Vector{Dof},
    position_states::Vector{Q},
    coordinates::Vector{SVector{D,T}};
    supercell_vectors::Union{Nothing, Vector{SVector{D,T}}} = nothing
) where {D, Q<:QuantumNumber, T<:Real}
    Lattice{D,Q,T}(position_dofs, position_states, coordinates, supercell_vectors)
end

"""
    get_coordinate(lattice::Lattice, state) -> Vector{T}

Return the coordinate for a given quantum state. The input can be QuantumNumber
or NamedTuple, and can contain extra keys beyond the position DOFs.

# Examples
```julia
lattice = Lattice(...)  # with position_dofs = [:moire_cell, :sublattice]

# Exact position state
coord = get_coordinate(lattice, QN(moire_cell=1, sublattice=2))

# Full quantum state with extra DOFs
coord = get_coordinate(lattice, QN(moire_cell=1, sublattice=2, spin=1, valley=1))
```
"""
function get_coordinate(lattice::Lattice, state::QuantumNumber)
    # Extract position-related keys from the input state
    pos_keys = Tuple(dof.name for dof in lattice.position_dofs)
    pos_nt = NamedTuple{pos_keys}(Tuple(state[k] for k in pos_keys))
    pos_state = QuantumNumber(pos_nt)

    idx = findfirst(==(pos_state), lattice.position_states)
    if isnothing(idx)
        error("Position state $pos_state not found in lattice")
    end
    return lattice.coordinates[idx]
end

# Also accept NamedTuple
function get_coordinate(lattice::Lattice, state::NamedTuple)
    return get_coordinate(lattice, QuantumNumber(state))
end

"""
    Base.show(io::IO, lattice::Lattice)

Custom display for Lattice in REPL.
"""
function Base.show(io::IO, lattice::Lattice{D, Q, T}) where {D, Q, T}
    n = length(lattice.position_dofs)
    nstates = length(lattice.position_states)

    println(io, "Lattice{$D} with $n position degree(s) of freedom:")
    for (i, dof) in enumerate(lattice.position_dofs)
        print(io, "  $i. $(dof.name): $(dof.size) states")
        if dof.labels !== nothing
            print(io, " [", join(dof.labels, ", "), "]")
        end
        println(io)
    end
    println(io, "Number of position states: $nstates")
    if isnothing(lattice.supercell_vectors)
        print(io, "Supercell vectors: not set")
    else
        print(io, "Supercell vectors: set")
    end
end

"""
    Lattice(unitcell, unitcell_vectors, box_size)

Create a lattice by tiling a unit cell.

The first DOF of unitcell defines the unit cell type (with size=1 in unitcell).
After tiling, its size becomes Nx*Ny*... (total number of unit cells).
The supercell_vectors are automatically computed as box_size .* unitcell_vectors.

# Arguments
- `unitcell::Lattice`: The unit cell definition (first DOF has size=1)
- `unitcell_vectors::Vector{Vector{T}}`: Unit cell translation vectors [a1, a2, ...]
- `box_size::NTuple{D, Int}`: Number of unit cells in each direction (Nx, Ny, ...)

# Returns
A new `Lattice` with the first DOF's size expanded to prod(box_size).
Coordinates: R = (i1-1)*a1 + (i2-1)*a2 + ... + r_unitcell

# Examples
```julia
# Define honeycomb unit cell (first DOF = cell type with size 1)
unitcell = Lattice(
    [Dof(:honeycomb_cell, 1), Dof(:sublattice, 2, [:A, :B])],
    [QN(honeycomb_cell=1, sublattice=1), QN(honeycomb_cell=1, sublattice=2)],
    [[0.0, 0.0], [0.5, 0.289]]
)

# Tile with unit cell vectors
a1, a2 = [1.0, 0.0], [0.5, 0.866]
lattice = Lattice(unitcell, [a1, a2], (4, 4))
# Result: DOFs [honeycomb_cell (16), sublattice (2)], 32 sites total
# supercell_vectors = [4*a1, 4*a2] automatically set
```
"""
function Lattice(
    unitcell::Lattice{D, Q, T},
    unitcell_vectors::Vector{Vector{T}},
    box_size::NTuple{M, Int}
) where {D, Q, T, M}
    @assert M == D "box_size dimension ($M) must match lattice dimension ($D)"
    @assert length(unitcell_vectors) == D "Need $D unit cell vectors for $D-dimensional lattice"
    @assert unitcell.position_dofs[1].size == 1 "First DOF of unitcell must have size=1"

    # Number of unit cells
    n_cells = prod(box_size)

    # Compute supercell vectors (convert user-provided Vector{Vector{T}} to SVector)
    supercell_vectors = SVector{D,T}[rd(SVector{D,T}(box_size[i] .* unitcell_vectors[i])) for i in 1:D]

    # Create new DOFs: expand first DOF's size, keep others
    first_dof = unitcell.position_dofs[1]
    new_dofs = [Dof(first_dof.name, n_cells, first_dof.labels); unitcell.position_dofs[2:end]]

    # Get names from unitcell (same structure, just different values for first DOF)
    all_names = Tuple(dof.name for dof in unitcell.position_dofs)

    # Generate position_states and coordinates
    N = QuantumNumber{all_names, NTuple{length(all_names), Int}}
    position_states = N[]
    sv_coordinates  = SVector{D,T}[]

    # Iterate over all unit cells (linear index)
    cell_idx = 1
    for indices in Iterators.product([1:s for s in box_size]...)
        # Calculate unit cell origin: R0 = (i1-1)*a1 + (i2-1)*a2 + ...
        R0 = zero(SVector{D,T})
        for (dim, i) in enumerate(indices)
            R0 = R0 + (i - 1) * unitcell_vectors[dim]
        end
        R0 = rd(R0)

        # Add each site in the unit cell
        for (cell_state, cell_coord) in zip(unitcell.position_states, unitcell.coordinates)
            # Replace first DOF value with cell_idx, keep rest
            state_values = values(cell_state)
            new_nt = NamedTuple{all_names}((cell_idx, state_values[2:end]...))
            push!(position_states, QuantumNumber(new_nt))

            # Calculate coordinate
            new_coord = rd(R0 + cell_coord)
            push!(sv_coordinates, new_coord)
        end

        cell_idx += 1
    end

    return Lattice(new_dofs, position_states, sv_coordinates; supercell_vectors=supercell_vectors)
end

#==================== Bond Specification ====================#

"""
    Bond{Q, T}

Represents a bond connecting one or more sites.

# Fields
- `states::Vector{Q}`: Position states (length determines bond order: 1=onsite, 2=two-body, etc.)
- `coordinates::Vector{SVector{D,T}}`: Physical (unwrapped) coordinates for each site.
  For periodic bonds these may lie outside the simulation cell.
- `icoordinates::Vector{SVector{D,T}}`: Unit-cell origin for each site.
  `icoordinates[n]` is the lattice vector of the unit cell that contains `coordinates[n]`.
  For intra-cell bonds all entries are zero vectors. For bonds that cross a periodic
  boundary, the entry for the image site carries the lattice shift, e.g. `[-2.0, 0.0]`.

# Examples
```julia
# Onsite bond — icoordinates default to zero vectors
bond = Bond([state1], [coord1])

# Two-body intra-cell bond
bond = Bond([state1, state2], [coord1, coord2])

# Two-body bond crossing a periodic boundary (site 2 lives in cell [-2, 0])
bond = Bond([state1, state2], [coord1, coord2], [[0.0, 0.0], [-2.0, 0.0]])
```
"""
struct Bond{Q<:QuantumNumber, T<:Real, D}
    states::Vector{Q}
    coordinates::Vector{SVector{D,T}}
    icoordinates::Vector{SVector{D,T}}
end

# Public constructors: accept Vector{Vector{T}} and convert to SVector (backward compatible)
function Bond(
    states::Vector{Q},
    coordinates::Vector{Vector{T}},
    icoordinates::Vector{Vector{T}}
) where {Q<:QuantumNumber, T<:Real}
    D = length(coordinates[1])
    Bond{Q,T,D}(states, SVector{D,T}.(coordinates), SVector{D,T}.(icoordinates))
end

function Bond(states::Vector{Q}, coordinates::Vector{Vector{T}}) where {Q<:QuantumNumber, T<:Real}
    D = length(coordinates[1])
    sv = SVector{D,T}.(coordinates)
    Bond{Q,T,D}(states, sv, [zero(SVector{D,T}) for _ in states])
end

function Base.show(io::IO, bond::Bond)
    print(io, "Bond(", bond.states, ", ", bond.coordinates, ", icoords=", bond.icoordinates, ")")
end

"""
    bonds(lattice, boundary, neighbors)

Generate bonds based on neighbor specification.

# Arguments
- `lattice::Lattice`: The lattice structure (must have supercell_vectors set)
- `boundary::NTuple{D, Symbol}`: Boundary conditions, :p (periodic) or :o (open)
- `neighbors`: Neighbor specification:
  - `Int`: n-th nearest neighbor (0=onsite, 1=nearest, 2=next-nearest, ...)
  - `Vector{Int}`: multiple neighbor orders [n1, n2, ...]
  - `Vector{<:Real}`: specific distances [r1, r2, ...]

# Returns
`Vector{Bond}` with appropriate number of sites per bond

# Examples
```julia
# Create lattice with tiling (supercell_vectors auto-set)
lattice = Lattice(unitcell, [a1, a2], (4, 4))

# Onsite bonds - returns Vector{Bond} with 1 site per bond
onsite = bonds(lattice, (:p, :p), 0)
# onsite[1] displays as: Bond{1}(QN(cell=1, sub=1) @ [0.000, 0.000])

# Nearest neighbor bonds - returns Vector{Bond} with 2 sites per bond
nn = bonds(lattice, (:p, :o), 1)
# nn[1] displays as: Bond{2}(QN(cell=1, sub=1) @ [0.000, 0.000], QN(cell=1, sub=2) @ [0.500, 0.289])

# Multiple neighbor orders
nn_and_nnn = bonds(lattice, (:p, :p), [1, 2])

# Specific distances
specific = bonds(lattice, (:p, :p), [1.0, 1.732])
```
"""
function bonds(
    lattice::Lattice{D, Q, T},
    boundary::NTuple{D, Symbol},
    neighbors::Int
) where {D, Q, T}
    if neighbors == 0
        return _generate_onsite_bonds(lattice)
    else
        supercell_vectors = _get_supercell_vectors(lattice)
        return _generate_neighbor_bonds(lattice, supercell_vectors, boundary, neighbors)
    end
end

# Multiple neighbor orders
function bonds(
    lattice::Lattice{D, Q, T},
    boundary::NTuple{D, Symbol},
    neighbors::Vector{Int}
) where {D, Q, T}
    result = Bond{Q, T, D}[]
    for n in neighbors
        append!(result, bonds(lattice, boundary, n))
    end
    return result
end

# Specific distances
function bonds(
    lattice::Lattice{D, Q, T},
    boundary::NTuple{D, Symbol},
    distances::Vector{<:Real};
    tolerance::Real = 1e-6
) where {D, Q, T}
    supercell_vectors = _get_supercell_vectors(lattice)
    result = Bond{Q, T, D}[]
    for d in distances
        append!(result, _generate_distance_bonds(lattice, supercell_vectors, boundary, d, tolerance))
    end
    return result
end

# Helper: get supercell_vectors or error
function _get_supercell_vectors(lattice::Lattice)
    if isnothing(lattice.supercell_vectors)
        error("lattice.supercell_vectors is not set. Use Lattice(unitcell, vectors, box_size) or set supercell_vectors manually.")
    end
    return lattice.supercell_vectors
end

# Helper: generate onsite bonds
function _generate_onsite_bonds(lattice::Lattice{D, Q, T}) where {D, Q, T}
    result = Bond{Q, T, D}[]
    zero_sv = zero(SVector{D,T})
    for (state, coord) in zip(lattice.position_states, lattice.coordinates)
        push!(result, Bond{Q,T,D}([state], [coord], [zero_sv]))
    end
    return result
end

# Helper: compute minimum image distance considering PBC
function _min_image_distance(
    coord1::SVector{D,T},
    coord2::SVector{D,T},
    supercell_vectors::Vector{SVector{D,T}},
    boundary::NTuple{D, Symbol}
) where {D, T}
    _, dist, _ = _min_image_delta(coord1, coord2, supercell_vectors, boundary)
    return dist
end

# Helper: get actual delta vector with minimum image convention.
# Returns (delta, dist, icell_shift) where icell_shift is the lattice vector
# of the periodic image used — i.e. the unit-cell origin of coord2's image.
function _min_image_delta(
    coord1::SVector{D,T},
    coord2::SVector{D,T},
    supercell_vectors::Vector{SVector{D,T}},
    boundary::NTuple{D, Symbol}
) where {D, T}
    min_dist    = Inf
    best_delta  = coord2 - coord1
    best_shifts = ntuple(_ -> 0, D)

    for shifts in Iterators.product([boundary[i] == :p ? (-1, 0, 1) : (0,) for i in 1:D]...)
        delta = coord2 - coord1
        for (dim, shift) in enumerate(shifts)
            delta = delta + shift * supercell_vectors[dim]
        end
        dist = sqrt(sum(delta .^ 2))
        if dist < min_dist
            min_dist    = dist
            best_delta  = delta
            best_shifts = shifts
        end
    end

    # icell_shift: the lattice vector of the cell containing coord2's image
    icell_shift = sum(best_shifts[d] * supercell_vectors[d] for d in 1:D)

    return rd(best_delta), rd(min_dist), rd(icell_shift)
end

# Helper: generate neighbor bonds by order
function _generate_neighbor_bonds(
    lattice::Lattice{D, Q, T},
    supercell_vectors::Vector{SVector{D,T}},
    boundary::NTuple{D, Symbol},
    neighbor_order::Int
) where {D, Q, T}
    n_sites = length(lattice.position_states)

    # Compute all unique distances
    distances = Set{T}()
    for i in 1:n_sites
        for j in (i+1):n_sites
            d = _min_image_distance(
                lattice.coordinates[i],
                lattice.coordinates[j],
                supercell_vectors, boundary
            )
            if d > 1e-10  # exclude self
                push!(distances, d)  # already rounded by _min_image_distance
            end
        end
    end

    # Sort distances and get the n-th smallest
    sorted_distances = sort(collect(distances))
    if neighbor_order > length(sorted_distances)
        error("neighbor_order=$neighbor_order exceeds available distance levels ($(length(sorted_distances)))")
    end

    target_distance = sorted_distances[neighbor_order]
    tolerance = 1e-6

    return _generate_distance_bonds(lattice, supercell_vectors, boundary, target_distance, tolerance)
end

"""
    is_positive_direction(delta::Vector) -> Bool

Check if delta vector is in "positive" direction using lexicographic ordering.
The first non-zero component must be positive.

Used to generate unidirectional bonds, avoiding duplicates like (i→j) and (j→i).
"""
function is_positive_direction(delta::AbstractVector{T}) where T
    for d in delta
        rd_d = rd(d)
        if rd_d > 0
            return true
        elseif rd_d < 0
            return false
        end
    end
    return true  # zero vector (shouldn't happen for valid bonds)
end

# Helper: generate bonds at specific distance (unidirectional).
#
# Design: explicitly enumerate ALL periodic images of each site (analogous to
# tiling the lattice into a supercluster), then collect every image whose
# distance from site i equals the target distance AND whose displacement is in
# a "positive" direction (to keep exactly one of each i↔j pair).
function _generate_distance_bonds(
    lattice::Lattice{D, Q, T},
    supercell_vectors::Vector{SVector{D,T}},
    boundary::NTuple{D, Symbol},
    distance::Real,
    tolerance::Real
) where {D, Q, T}
    result   = Bond{Q, T, D}[]
    n_sites  = length(lattice.position_states)
    zero_sv  = zero(SVector{D,T})

    # Shift ranges: ±1 for periodic directions, 0 only for open directions.
    shift_ranges = [boundary[d] == :p ? (-1, 0, 1) : (0,) for d in 1:D]

    for i in 1:n_sites
        for j in 1:n_sites
            i == j && continue

            for shifts in Iterators.product(shift_ranges...)
                # Cell-origin displacement for this periodic image of site j.
                icell_shift = zero_sv
                for (d, s) in enumerate(shifts)
                    icell_shift = icell_shift + s * supercell_vectors[d]
                end

                delta = rd(lattice.coordinates[j] - lattice.coordinates[i] + icell_shift)
                dist  = rd(sqrt(sum(delta .^ 2)))

                abs(dist - distance) < tolerance || continue
                is_positive_direction(delta)      || continue

                coord1 = lattice.coordinates[i]
                coord2 = rd(coord1 + delta)

                push!(result, Bond{Q,T,D}(
                    [lattice.position_states[i], lattice.position_states[j]],
                    [coord1, coord2],
                    [zero_sv, rd(icell_shift)]
                ))
            end
        end
    end

    return result
end
