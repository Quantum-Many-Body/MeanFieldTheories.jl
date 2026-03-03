"""
Operator Construction for Quantum Systems

Provides types and functions for building Hamiltonian matrices
and interaction tensors from operator products.
"""

#==================== Abstract Operator Type ====================#

"""
    Operator

Abstract type for quantum operators.
Subtypes include FermionOp (and potentially SpinOp, BosonOp, etc. in the future).
"""
abstract type Operator end

#==================== Fermion Operator ====================#

"""
    FermionOp{Q} <: Operator

Represents a single fermion operator (creation or annihilation).

# Fields
- `qn::Q`: Quantum number identifying the state
- `dag::Bool`: true = creation operator (c†), false = annihilation operator (c)
"""
struct FermionOp{Q<:QuantumNumber} <: Operator
    qn::Q
    dag::Bool
end

"""
    c(qn) -> FermionOp

Create an annihilation operator for quantum state `qn`.
"""
c(qn::QuantumNumber) = FermionOp(qn, false)
c(qn::NamedTuple) = FermionOp(QuantumNumber(qn), false)

"""
    cdag(qn) -> FermionOp

Create a creation operator (c†) for quantum state `qn`.
"""
cdag(qn::QuantumNumber) = FermionOp(qn, true)
cdag(qn::NamedTuple) = FermionOp(QuantumNumber(qn), true)

# Display
function Base.show(io::IO, op::FermionOp)
    # Format quantum number as subscript: c†_{site,spin,...}
    qn_str = join(values(op.qn), ",")
    print(io, op.dag ? "c†" : "c", "_{", qn_str, "}")
end

#==================== Operator Product Type ====================#

"""
    Operators{T}

Represents a product of operators with a coefficient.

Operators are stored in their **original order** (no automatic reordering).
Reordering to canonical form (InterAll: c†c c†c...) happens when building
matrices/tensors via `build_onebody_matrix` or `build_interaction_tensor`.

# Type Parameters
- `T`: Numeric type for coefficient (e.g., Float64, ComplexF64)

# Fields
- `value::T`: Coefficient
- `ops::Vector{<:Operator}`: Operators in original order

# Examples
```julia
i = QN(site=1, spin=1)
j = QN(site=2, spin=1)

# One-body: -t * c†_i c_j
op = Operators(-t, [cdag(i), c(j)])

# Two-body: U * c†_i↑ c_i↑ c†_i↓ c_i↓
op = Operators(U, [cdag(i_up), c(i_up), cdag(i_dn), c(i_dn)])

# Pair hopping: stored as-is, reordered when building tensor
op = Operators(J, [cdag(i_up), cdag(i_dn), c(j_dn), c(j_up)])
```
"""
struct Operators{T<:Number}
    value::T
    ops::Vector{<:Operator}
end

# Constructor from value and Operator varargs
function Operators(value::Number, ops::Operator...)
    Operators(value, collect(ops))
end

# Display for single Operators
function Base.show(io::IO, op::Operators)
    # Show coefficient
    if op.value == 1
        # Skip coefficient if it's 1
    elseif op.value == -1
        print(io, "-")
    else
        print(io, op.value, " ")
    end

    # Show operators
    for fop in op.ops
        print(io, fop)
    end
end

# Display for vector of Operators (Hamiltonian-like sum)
function Base.show(io::IO, ::MIME"text/plain", ops::AbstractVector{<:Operators})
    if isempty(ops)
        print(io, "Empty operator list")
        return
    end

    n = length(ops)
    print(io, "Operators with ", n, n == 1 ? " term:" : " terms:")

    for i in 1:n
        print(io, "\n  ")
        if i > 1
            # Add sign
            if real(ops[i].value) >= 0
                print(io, "+ ")
            end
        end
        show(io, ops[i])
    end

end

#==================== Reordering Algorithm (Internal) ====================#

"""
    _reorder_to_interall(ops::Vector{<:FermionOp}) -> (sign, reordered)

Reorder fermion operators to c†c c†c... pattern (InterAll format).
Uses bubble sort with anticommutation sign tracking.

Target pattern: [c†, c, c†, c, ...]

# Returns
- `sign`: +1 or -1 from fermionic anticommutation
- `reordered`: Vector of FermionOp in InterAll order
"""
function _reorder_to_interall(ops::Vector{<:FermionOp})
    n = length(ops)
    ops = copy(ops)
    sign = 1

    # Target pattern: alternating c† and c
    target_dag = [isodd(i) for i in 1:n]

    for i in 1:n
        j = i
        while j <= n && ops[j].dag != target_dag[i]
            j += 1
        end

        if j > n
            error("Cannot reorder: wrong number of c† vs c operators. " *
                  "Expected $(count(target_dag)) creation and $(n - count(target_dag)) annihilation operators.")
        end

        while j > i
            ops[j-1], ops[j] = ops[j], ops[j-1]
            sign *= -1
            j -= 1
        end
    end

    return sign, ops
end

#==================== General Term Generators ====================#

"""
    generate_onebody(dofs, bonds, value; order=(cdag, 1, c, 2), hc=true)

Generate one-body terms from bonds.

# Arguments
- `dofs::SystemDofs`: Full DOF specification (position + internal DOFs)
- `bonds::Vector{Bond}`: Two-site bonds from bonds() function
- `value`: Coefficient, either:
  - `Number`: constant value for all internal DOF combinations
  - `Function(delta, qn1, qn2) -> Number`: custom function to constrain DOFs.

# Keyword Arguments
- `order::Tuple`: Format `(op_type1, site1, op_type2, site2)`
  - Default: `(cdag, 1, c, 2)` for standard hopping c†[site1] c[site2]
- `hc::Bool`: whether to include the Hermitian conjugate terms

# Returns
`NamedTuple` with three parallel `Vector` fields:
- `.ops::Vector{Operators}`: operator terms
- `.delta::Vector{Vector{Float64}}`: physical displacement for each term
  (`bond.coordinates[2] - bond.coordinates[1]`, unwrapped)
- `.irvec::Vector{Vector{Float64}}`: unit-cell displacement for each term
  (`bond.icoordinates[2] - bond.icoordinates[1]`); zero for intra-cell bonds,
  non-zero for bonds crossing a periodic boundary

For H.c. terms the sign of both `delta` and `irvec` is flipped.

# NOTICE
  - All internal DOFs are mixed unless constrained via the `value` function.

# Examples
```julia
result = generate_onebody(dofs, nn_bonds, -1.0)
result.ops    # Vector{Operators}
result.delta  # physical displacements
result.irvec  # unit-cell displacements (use this for momentum-space HF)

# Access operators only (compatible with real-space HF)
ops = generate_onebody(dofs, nn_bonds, -1.0).ops
```
"""
function generate_onebody(
    dofs::SystemDofs,
    bonds::Vector{<:Bond},
    value::Union{Number, Function};
    order::Tuple = (cdag, 1, c, 2),
    hc::Bool = true
)
    @assert length(order) == 4 "order must have 4 elements: (op_type1, site1, op_type2, site2)"
    op_type1, site1, op_type2, site2 = order
    @assert site1 in (1, 2) && site2 in (1, 2) "site indices must be 1 or 2"

    ops_list   = Operators[]
    delta_list = Vector{Float64}[]
    irvec_list = Vector{Float64}[]

    for bond in bonds
        @assert length(bond.states) == 2 "One-body generator requires 2-site bonds"

        s1, s2   = bond.states
        delta    = rd(bond.coordinates[2]  .- bond.coordinates[1])
        irvec    = rd(bond.icoordinates[2] .- bond.icoordinates[1])
        pos_keys = keys(s1)

        qn_at_site1 = [qn for qn in dofs.valid_states if all(qn[k] == s1[k] for k in pos_keys)]
        qn_at_site2 = [qn for qn in dofs.valid_states if all(qn[k] == s2[k] for k in pos_keys)]

        qn_list1 = site1 == 1 ? qn_at_site1 : qn_at_site2
        qn_list2 = site2 == 1 ? qn_at_site1 : qn_at_site2

        for qn1 in qn_list1, qn2 in qn_list2
            v = value isa Number ? value : value(delta, qn1, qn2)
            if !iszero(v)
                push!(ops_list,   Operators(v, [op_type1(qn1), op_type2(qn2)]))
                push!(delta_list, delta)
                push!(irvec_list, irvec)
                if hc
                    push!(ops_list,   Operators(conj(v), [op_type1(qn2), op_type2(qn1)]))
                    push!(delta_list, rd(-delta))
                    push!(irvec_list, rd(-irvec))
                end
            end
        end
    end

    return (ops=ops_list, delta=delta_list, irvec=irvec_list)
end

"""
    generate_twobody(dofs, bonds, value; order=(cdag, 1, c, 1, cdag, 2, c, 2)) -> Vector{Operators}

Generate two-body interaction terms from bonds.

# Arguments
- `dofs::SystemDofs`: Full DOF specification
- `bonds::Vector{Bond}`: Two-site bonds
- `value`: Coefficient, either:
  - `Number`: constant for all combinations
  - `Function(delta, qn1, qn2, qn3, qn4) -> Number`: custom function used to constrain degrees of freedom.

# Keyword Arguments
- `order::Tuple`: Format `(type1, site1, type2, site2, type3, site3, type4, site4)`
  - Default: `(cdag, 1, c, 1, cdag, 2, c, 2)` for density-density n_i n_j

# NOTICE
  - All internal degrees of freedom are counted in a mixed manner if one does not constrain degrees of freedom!!!

# Examples
```julia
# nearest-neighbor Coulomb interaction: ∑_{all internal dofs} V n_i n_j
ops = generate_twobody(dofs, nn_bonds, V)

# onsite Hubbard interaction: ∑_α U n_iα↑ n_iα↓
ops = generate_twobody(dofs, onsite_bonds, 
    (delta, qn1, qn2, qn3, qn4) ->
        (qn1.orbital == qn2.orbital == qn3.orbital == qn4.orbital) &&
        (qn1.spin, qn2.spin, qn3.spin, qn4.spin) == (1, 1, 2, 2) ? U : 0.0
    ,
    order = (cdag, 1, c, 1, cdag, 1, c, 1)
    )

# Pair hopping: ∑_{α≠β} J c†_iα↑ c†_iα↓ c_iβ↓ c_iβ↑
ops = = generate_twobody(dofs, onsite_bonds, 
    (delta, qn1, qn2, qn3, qn4) ->
        (qn1.orbital, qn3.orbital) == (qn2.orbital, qn4.orbital) &&
        (qn1.orbital !== qn3.orbital) &&
        (qn1.spin, qn2.spin, qn3.spin, qn4.spin) == (1,2,2,1) ? J : 0.0
    ,
    order = (cdag, 1, cdag, 1, c, 1, c, 1)
    )
```
"""
function generate_twobody(
    dofs::SystemDofs,
    bonds::Vector{<:Bond},
    value::Union{Number, Function};
    order::Tuple = (cdag, 1, c, 1, cdag, 2, c, 2)
)
    @assert length(order) == 8 "order must have 8 elements"
    op_type1, site1, op_type2, site2, op_type3, site3, op_type4, site4 = order
    sites = (site1, site2, site3, site4)
    @assert all(s in (1, 2) for s in sites) "site indices must be 1 or 2"

    result = Operators[]

    for bond in bonds
        if length(bond.states) == 1 #onsite twobody term
            s1, s2 = bond.states[1], bond.states[1]
            delta = rd(bond.coordinates[1] .- bond.coordinates[1])
        elseif length(bond.states) == 2 #different sites twobody term
            s1, s2 = bond.states
            delta = rd(bond.coordinates[2] .- bond.coordinates[1])
        else
            throw(ArgumentError("Not two-body interaction!"))
        end

        pos_keys = keys(s1)

        qn_at_site1 = [qn for qn in dofs.valid_states if all(qn[k] == s1[k] for k in pos_keys)]
        qn_at_site2 = [qn for qn in dofs.valid_states if all(qn[k] == s2[k] for k in pos_keys)]

        qn_lists = ntuple(i -> sites[i] == 1 ? qn_at_site1 : qn_at_site2, 4)

        for qn1 in qn_lists[1], qn2 in qn_lists[2], qn3 in qn_lists[3], qn4 in qn_lists[4]
            v = value isa Number ? value : value(delta, qn1, qn2, qn3, qn4)
            if !iszero(v)
                push!(result, Operators(v, [op_type1(qn1), op_type2(qn2), op_type3(qn3), op_type4(qn4)]))
            end
        end
    end

    return result
end

#==================== Matrix/Tensor Construction ====================#

"""
    build_onebody_matrix(dofs::SystemDofs, ops::AbstractVector{<:Operators})

Build Hermitian one-body Hamiltonian matrix from Operators.
Each Operators must have exactly 2 FermionOp.
Operators are reordered to c†c format internally.
"""
function build_onebody_matrix(dofs::SystemDofs, ops::AbstractVector{<:Operators})
    isempty(ops) && error("Cannot build matrix from empty operators list")

    T = typeof(first(ops).value)
    dim = total_dim(dofs)
    H = zeros(T, dim, dim)

    for op in ops
        length(op.ops) == 2 || error("One-body operator must have exactly 2 operators")
        all(o isa FermionOp for o in op.ops) || error("One-body matrix requires FermionOp")

        fermion_ops = Vector{FermionOp}(op.ops)
        sign, reordered = _reorder_to_interall(fermion_ops)

        i = qn2linear(dofs, reordered[1].qn)
        j = qn2linear(dofs, reordered[2].qn)
        H[i, j] += sign * op.value
    end

    if !ishermitian(H)
        throw(ArgumentError("H is not Hermitian"))
    end

    return H
end

"""
    build_interaction_tensor(dofs::SystemDofs, ops::AbstractVector{<:Operators})

Build two-body interaction tensor from Operators.
Each Operators must have exactly 4 FermionOp.
Operators are reordered to c†c c†c format (InterAll) internally.

The tensor V[i,j,k,l] corresponds to c†_i c_j c†_k c_l in InterAll format.
"""
function build_interaction_tensor(dofs::SystemDofs, ops::AbstractVector{<:Operators})
    isempty(ops) && error("Cannot build tensor from empty operators list")

    T = typeof(first(ops).value)
    dim = total_dim(dofs)
    V = zeros(T, dim, dim, dim, dim)

    for op in ops
        length(op.ops) == 4 || error("Two-body operator must have exactly 4 operators")
        all(o isa FermionOp for o in op.ops) || error("Interaction tensor requires FermionOp")

        fermion_ops = Vector{FermionOp}(op.ops)
        sign, reordered = _reorder_to_interall(fermion_ops)

        i = qn2linear(dofs, reordered[1].qn)
        j = qn2linear(dofs, reordered[2].qn)
        k = qn2linear(dofs, reordered[3].qn)
        l = qn2linear(dofs, reordered[4].qn)
        V[i, j, k, l] += sign * op.value
    end

    return V
end

