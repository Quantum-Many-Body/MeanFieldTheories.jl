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
Reordering to canonical form (creation-annihilation alternating order: c†c c†c...) happens when building
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
    _reorder_to_ca_alternating(ops::Vector{<:FermionOp}) -> (sign, reordered)

Reorder fermion operators to c†c c†c... pattern (creation-annihilation alternating order).
Uses bubble sort with anticommutation sign tracking.

Target pattern: [c†, c, c†, c, ...]

# Returns
- `sign`: +1 or -1 from fermionic anticommutation
- `reordered`: Vector of FermionOp in creation-annihilation alternating order
"""
function _reorder_to_ca_alternating(ops::Vector{<:FermionOp})
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

"""
    _reorder_to_ca_alternating_with_positions(ops, positions) -> (sign, reordered_ops, reordered_positions)

Same bubble-sort reordering as `_reorder_to_ca_alternating`, but also permutes a parallel
`positions` vector (one `Vector{Float64}` per operator) so that after reordering,
`reordered_positions[i]` always corresponds to `reordered_ops[i]`.

Used by `generate_twobody` to track which unit-cell position each operator ends up at
after reordering to creation-annihilation alternating order.
"""
function _reorder_to_ca_alternating_with_positions(
    ops::Vector{<:FermionOp},
    positions::Vector{<:AbstractVector{<:Real}}
)
    n = length(ops)
    ops = copy(ops)
    positions = copy(positions)
    sign = 1

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
            positions[j-1], positions[j] = positions[j], positions[j-1]
            sign *= -1
            j -= 1
        end
    end

    return sign, ops, positions
end

"""
    _reorder_to_ca_alternating_with_two_positions(ops, ipos, phys_pos)
        -> (sign, reordered_ops, reordered_ipos, reordered_phys_pos)

Same bubble-sort as `_reorder_to_ca_alternating_with_positions`, but permutes two parallel
position vectors (`ipos` and `phys_pos`) in a single sort pass, avoiding the cost of
running the sort twice when both unit-cell and physical positions are needed.
"""
function _reorder_to_ca_alternating_with_two_positions(
    ops::Vector{<:FermionOp},
    ipos::Vector{<:AbstractVector{<:Real}},
    phys_pos::Vector{<:AbstractVector{<:Real}}
)
    n = length(ops)
    ops      = copy(ops)
    ipos     = copy(ipos)
    phys_pos = copy(phys_pos)
    sign = 1

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
            ops[j-1],      ops[j]      = ops[j],      ops[j-1]
            ipos[j-1],     ipos[j]     = ipos[j],     ipos[j-1]
            phys_pos[j-1], phys_pos[j] = phys_pos[j], phys_pos[j-1]
            sign *= -1
            j -= 1
        end
    end

    return sign, ops, ipos, phys_pos
end

#==================== General Term Generators ====================#

"""
    generate_onebody(dofs, bonds, value; order=(cdag, :i, c, :j), hc=true)

Generate one-body terms from bonds.

# Arguments
- `dofs::SystemDofs`: Full DOF specification (position + internal DOFs)
- `bonds::Vector{Bond}`: One- or two-site bonds from bonds() function
- `value`: Coefficient, either:
  - `Number`: constant value for all internal DOF combinations
  - `Function(delta, qn1, qn2) -> Number`: custom function to constrain DOFs.
    `delta = coord(op1_site) - coord(op2_site)` in the user-specified order.

# Keyword Arguments
- `order::Tuple`: Format `(op_type1, label1, op_type2, label2)` where labels are
  **Symbols** acting as site indices.
  - Default: `(cdag, :i, c, :j)` — standard hopping c†_i c_j
  - Same label (e.g. `(cdag, :i, c, :i)`) — on-site/chemical-potential term
  - Any operator order is accepted; operators are reordered to c†c alternating
    form automatically with the fermionic sign absorbed into the coefficient.
- `hc::Bool`: whether to include the Hermitian conjugate terms.
  Ignored (no hc added) when both labels are identical (on-site terms), since
  c†_i c_i is already self-contained and adding h.c. would double-count.

# Returns
`NamedTuple` with three parallel `Vector` fields:
- `.ops::Vector{Operators}`: operator terms in c†c order, fermionic sign absorbed
- `.delta::Vector{SVector{D,T}}`: physical displacement `coord(op1) - coord(op2)`
  after reordering; sign is flipped for H.c. terms
- `.irvec::Vector{SVector{D,T}}`: unit-cell displacement after reordering;
  zero for intra-cell bonds, non-zero for bonds crossing a periodic boundary

# NOTICE
  - All internal DOFs are mixed unless constrained via the `value` function.

# Examples
```julia
result = generate_onebody(dofs, nn_bonds, -1.0)
result.ops    # Vector{Operators}
result.delta  # physical displacements
result.irvec  # unit-cell displacements (use this for momentum-space HF)

# On-site chemical potential (no h.c. added automatically)
result = generate_onebody(dofs, onsite_bonds, -μ, order=(cdag, :i, c, :i))

# Unconventional input order — reordered internally with correct sign
result = generate_onebody(dofs, nn_bonds, 1.0, order=(c, :i, cdag, :j))
```
"""
function generate_onebody(
    dofs::SystemDofs,
    bonds::Vector{<:Bond},
    value::Union{Number, Function};
    order::Tuple = (cdag, :i, c, :j),
    hc::Bool = true
)
    @assert length(order) == 4 "order must have 4 elements: (op_type1, label1, op_type2, label2)"
    op_type1, label1, op_type2, label2 = order
    @assert label1 isa Symbol && label2 isa Symbol "order labels must be Symbols (e.g. :i, :j)"

    unique_labels = unique([label1, label2])
    k = length(unique_labels)          # 1 = on-site, 2 = hopping
    label_col = Dict(l => idx for (idx, l) in enumerate(unique_labels))
    # site_for_op[i]: which bond site (1-based) the i-th operator acts on
    site_for_op = (label_col[label1], label_col[label2])

    SV         = eltype(first(bonds).coordinates)
    ops_list   = Operators[]
    delta_list = SV[]
    irvec_list = SV[]

    for bond in bonds
        @assert length(bond.states) in (1, 2) "One-body generator requires 1- or 2-site bonds"
        nb = length(bond.states)
        @assert nb >= k "bond has $nb sites but order requires $k distinct positions"

        pos_keys = keys(bond.states[1])
        qn_at = [[qn for qn in dofs.valid_states
                  if all(qn[key] == bond.states[s][key] for key in pos_keys)]
                 for s in 1:nb]

        raw_ipos     = SV[bond.icoordinates[site_for_op[1]], bond.icoordinates[site_for_op[2]]]
        raw_phys_pos = SV[bond.coordinates[site_for_op[1]],  bond.coordinates[site_for_op[2]]]
        raw_delta    = rd(raw_phys_pos[1] - raw_phys_pos[2])

        for qn1 in qn_at[site_for_op[1]], qn2 in qn_at[site_for_op[2]]
            v = value isa Number ? value : value(raw_delta, qn1, qn2)
            iszero(v) && continue

            raw_ops = FermionOp[op_type1(qn1), op_type2(qn2)]
            sign, reord_ops, reord_ipos, reord_phys_pos =
                _reorder_to_ca_alternating_with_two_positions(raw_ops, raw_ipos, raw_phys_pos)

            delta = rd(reord_phys_pos[1] - reord_phys_pos[2])
            irvec = rd(reord_ipos[1] - reord_ipos[2])

            push!(ops_list,   Operators(sign * v, reord_ops))
            push!(delta_list, delta)
            push!(irvec_list, irvec)

            # hc: only for hopping terms (2 distinct site labels)
            # On-site terms (k == 1) are already self-contained; adding h.c. would double-count
            if hc && k == 2
                hc_op_type1 = op_type2 === cdag ? c : cdag
                hc_op_type2 = op_type1 === cdag ? c : cdag
                hc_raw_ops      = FermionOp[hc_op_type1(qn2), hc_op_type2(qn1)]
                hc_raw_ipos     = SV[raw_ipos[2],     raw_ipos[1]]
                hc_raw_phys_pos = SV[raw_phys_pos[2], raw_phys_pos[1]]

                hc_sign, hc_reord_ops, hc_reord_ipos, hc_reord_phys_pos =
                    _reorder_to_ca_alternating_with_two_positions(
                        hc_raw_ops, hc_raw_ipos, hc_raw_phys_pos)

                hc_delta = rd(hc_reord_phys_pos[1] - hc_reord_phys_pos[2])
                hc_irvec = rd(hc_reord_ipos[1] - hc_reord_ipos[2])

                push!(ops_list,   Operators(hc_sign * conj(v), hc_reord_ops))
                push!(delta_list, hc_delta)
                push!(irvec_list, hc_irvec)
            end
        end
    end

    return (ops=ops_list, delta=delta_list, irvec=irvec_list)
end

# ---- helper: all injective k-tuples from {1..n} ----
# Returns every ordered selection of k distinct elements from {1,…,n},
# i.e., all injective functions {1..k} → {1..n} as Vector{Int} of length k.
function _injective_assignments(n::Int, k::Int)
    result = Vector{Vector{Int}}()
    _injective_rec!(result, Int[], n, k)
    return result
end
function _injective_rec!(result, cur, n, k)
    length(cur) == k && (push!(result, copy(cur)); return)
    for i in 1:n
        i in cur && continue
        push!(cur, i)
        _injective_rec!(result, cur, n, k)
        pop!(cur)
    end
end

"""
    generate_twobody(dofs, bonds, value; order=(cdag, :i, c, :i, cdag, :j, c, :j))

Generate two-body interaction terms from bonds.

# Arguments
- `dofs::SystemDofs`: Full DOF specification.
- `bonds::Vector{Bond}`: Bonds with 1–4 sites.
- `value`: Coefficient, either:
  - `Number`: constant for all internal DOF combinations.
  - `Function(deltas, qn1, qn2, qn3, qn4) -> Number`: custom amplitude function.
    `deltas[m] = coord(op_m) - coord(op_4)` for m = 1,2,3 (always 3 entries).

# Keyword Arguments
- `order::Tuple`: Format `(type1, label1, type2, label2, type3, label3, type4, label4)`
  where labels are **Symbols** acting as free Einstein summation indices.
  Each unique Symbol is an independent position index summed over all injective
  assignments to bond sites (different Symbols → different sites).

  Common patterns:
  - `(cdag, :i, c, :i, cdag, :j, c, :j)` — density-density  Σ_{i≠j} n_i n_j (default)
  - `(cdag, :i, c, :i, cdag, :i, c, :i)` — on-site  Σ_i n_i² (use with 1-site bonds)
  - `(cdag, :i, c, :j, cdag, :i, c, :j)` — exchange-type
  - `(cdag, :i, c, :j, cdag, :i, c, :k)` — three distinct positions (requires 3-site bond)

# Returns
`NamedTuple` with three parallel `Vector` fields:
- `.ops::Vector{Operators}`: operator terms in creation-annihilation alternating order
  (c†c c†c), fermionic sign absorbed into the coefficient.
- `.delta::Vector{NTuple{3, SVector{D,T}}}`: physical displacements `(δ1, δ2, δ3)`,
  `δm = coord(op_m) - coord(op_4)` after reordering.
- `.irvec::Vector{NTuple{3, SVector{D,T}}}`: unit-cell displacements `(τ1, τ2, τ3)`,
  `τm = icoord(op_m) - icoord(op_4)` after reordering.

# Examples
```julia
# Nearest-neighbor Coulomb: V Σ_{i≠j,σσ'} n_{iσ} n_{jσ'}  (full ordered-pair sum)
twobody = generate_twobody(dofs, nn_bonds, V)

# Onsite Hubbard U: Σ_i U n_{i↑} n_{i↓}
twobody = generate_twobody(dofs, onsite_bonds,
    (deltas, qn1, qn2, qn3, qn4) ->
        (qn1.spin == qn2.spin) && (qn3.spin == qn4.spin) && (qn1.spin !== qn3.spin) ? U/2 : 0.0,
    order = (cdag, :i, c, :i, cdag, :i, c, :i))

# NN Coulomb in x-direction only (filter by deltas[1] = coord(i) - coord(j))
twobody = generate_twobody(dofs, nn_bonds,
    (deltas, qn1, qn2, qn3, qn4) -> abs(deltas[1][1]) ≈ 1.0 && iszero(deltas[1][2]) ? V : 0.0)

# Pair hopping: Σ_{i≠j} J c†_{i↑} c†_{i↓} c_{j↓} c_{j↑}
twobody = generate_twobody(dofs, nn_bonds,
    (deltas, qn1, qn2, qn3, qn4) ->
        (qn1.spin, qn2.spin, qn3.spin, qn4.spin) == (1, 2, 2, 1) ? J : 0.0,
    order = (cdag, :i, cdag, :i, c, :j, c, :j))
```
"""
function generate_twobody(
    dofs::SystemDofs,
    bonds::Vector{<:Bond},
    value::Union{Number, Function};
    order::Tuple = (cdag, :i, c, :i, cdag, :j, c, :j)
)
    @assert length(order) == 8 "order must have 8 elements"
    op_types = (order[1], order[3], order[5], order[7])
    labels   = (order[2], order[4], order[6], order[8])
    @assert all(l isa Symbol for l in labels) "order labels must be Symbols (e.g. :i, :j, :k)"

    # Map each unique label to an index; ops_label_idx[i] = column in the assignment vector
    unique_labels = unique(collect(labels))
    k             = length(unique_labels)
    label_col     = Dict(l => idx for (idx, l) in enumerate(unique_labels))
    ops_col       = ntuple(i -> label_col[labels[i]], 4)  # which assignment column each op uses

    SV         = eltype(first(bonds).coordinates)
    ops_list   = Operators[]
    delta_list = NTuple{3, SV}[]
    irvec_list = NTuple{3, SV}[]

    raw_ops      = Vector{FermionOp}(undef, 4)
    raw_ipos     = Vector{SV}(undef, 4)
    raw_phys_pos = Vector{SV}(undef, 4)

    for bond in bonds
        nb = length(bond.states)
        @assert nb >= k "bond has $nb sites but order requires $k distinct positions"

        # pos_keys: position-like keys in bond.states (e.g. :site), used to filter
        # dofs.valid_states by position while ignoring internal DOFs like :spin.
        pos_keys = keys(bond.states[1])

        # qn_at[s]: all valid QNs (with internal DOFs) at bond site s.
        qn_at = [[qn for qn in dofs.valid_states
                  if all(qn[key] == bond.states[s][key] for key in pos_keys)]
                 for s in 1:nb]

        # Enumerate all injective assignments of unique_labels → bond sites {1..nb}.
        # Each assignment is a Vector{Int} of length k where assignment[col] = bond site.
        # This generates the full ordered sum Σ_{distinct i,j,...} over all positions.
        for assignment in _injective_assignments(nb, k)
            # site_for_op[i] = bond site index assigned to the i-th operator
            site_for_op = ntuple(i -> assignment[ops_col[i]], 4)

            # deltas[m] = coord(op_m site) - coord(op_4 site), m = 1,2,3.
            # Passed to the value function so users can filter by bond direction.
            ref_coord = bond.coordinates[site_for_op[4]]
            deltas = SV[rd(bond.coordinates[site_for_op[m]] - ref_coord) for m in 1:3]

            qn_lists = ntuple(i -> qn_at[site_for_op[i]], 4)

            for qn1 in qn_lists[1], qn2 in qn_lists[2], qn3 in qn_lists[3], qn4 in qn_lists[4]
                v = value isa Number ? value : value(deltas, qn1, qn2, qn3, qn4)
                iszero(v) && continue

                raw_ops[1] = op_types[1](qn1); raw_ops[2] = op_types[2](qn2)
                raw_ops[3] = op_types[3](qn3); raw_ops[4] = op_types[4](qn4)
                for i in 1:4
                    raw_ipos[i]     = bond.icoordinates[site_for_op[i]]
                    raw_phys_pos[i] = bond.coordinates[site_for_op[i]]
                end

                # Reorder to c†c c†c format; fermionic sign absorbed into coefficient.
                sign, reord_ops, reord_ipos, reord_phys_pos =
                    _reorder_to_ca_alternating_with_two_positions(raw_ops, raw_ipos, raw_phys_pos)

                δ1 = rd(reord_phys_pos[1] - reord_phys_pos[4])
                δ2 = rd(reord_phys_pos[2] - reord_phys_pos[4])
                δ3 = rd(reord_phys_pos[3] - reord_phys_pos[4])
                τ1 = rd(reord_ipos[1] - reord_ipos[4])
                τ2 = rd(reord_ipos[2] - reord_ipos[4])
                τ3 = rd(reord_ipos[3] - reord_ipos[4])

                push!(ops_list,   Operators(sign * v, reord_ops))
                push!(delta_list, (δ1, δ2, δ3))
                push!(irvec_list, (τ1, τ2, τ3))
            end
        end
    end

    return (ops=ops_list, delta=delta_list, irvec=irvec_list)
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
        sign, reordered = _reorder_to_ca_alternating(fermion_ops)

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
Operators are reordered to creation-annihilation alternating order (c†c c†c) internally.

The tensor V[i,j,k,l] corresponds to c†_i c_j c†_k c_l in creation-annihilation alternating order.
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
        sign, reordered = _reorder_to_ca_alternating(fermion_ops)

        i = qn2linear(dofs, reordered[1].qn)
        j = qn2linear(dofs, reordered[2].qn)
        k = qn2linear(dofs, reordered[3].qn)
        l = qn2linear(dofs, reordered[4].qn)
        V[i, j, k, l] += sign * op.value
    end

    return V
end


function opsum(terms...)
    if isempty(terms) 
        return (ops=Operators[], delta=NTuple{3}[], irvec=NTuple{3}[])
    else
    return (ops   = vcat((term.ops for term in terms)...),
            delta = vcat((term.delta for term in terms)...),
            irvec = vcat((term.irvec for term in terms)...))
    end
end
