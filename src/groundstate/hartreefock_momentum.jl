"""
Hartree-Fock mean-field approximation in momentum space.

Implements the momentum-space (k-space) formulation of unrestricted Hartree-Fock (UHF),
exploiting translational symmetry to reduce computational cost from O(N³) to O(Nk·d³)
where Nk is the number of k-points and d is the internal dimension per unit cell.

Key assumptions:
- The Hamiltonian has discrete translational symmetry (Bloch's theorem applies).
- The ground state preserves (possibly a subgroup of) the lattice translational symmetry.
  If spontaneous symmetry breaking is expected (e.g. antiferromagnetic, CDW), the user
  must pre-specify an enlarged magnetic/modulated unit cell so that k-diagonal Green's
  function G_{ab}(k) is a valid ansatz.

The single-particle Green's function is stored as a 3-tensor G[k_idx, a, b] of shape
(Nk, d, d) rather than the (N, N) matrix used in real-space HF.
"""

# ──────────────── Preprocessing: kinetic term ────────────────

"""
    build_Tr(dofs, ops, irvec) -> NamedTuple

Parse one-body operators and build the real-space hopping table T_{ab}(r).

Scans `irvec` to find all unique displacement vectors, then accumulates hopping
amplitudes into one `d_int × d_int` matrix per displacement. Non-one-body operators
(i.e. those with `length(op.ops) ≠ 2`) in `ops` are silently skipped, so the full
operator list from `generate_onebody` (which may include metadata entries) can be
passed directly.

# Arguments
- `dofs::SystemDofs`: DOFs of one magnetic unit cell.
- `ops::Vector{<:Operators}`: operator terms; only those with exactly 2 FermionOps
  (one-body) are used. `ops` may contain operators of other sizes — they are ignored.
- `irvec::Vector{<:AbstractVector{<:Real}}`: displacement vector per operator term,
  paired one-to-one with `ops`.

# Returns
`NamedTuple (mats, rvec)` where:
- `mats::Vector{Matrix}`: one `d_int × d_int` hopping matrix per unique displacement;
  element type matches `eltype` of the operator values (e.g. `Float64` or `ComplexF64`).
- `rvec::Vector{Vector{Float64}}`: the corresponding displacement vectors, same order as `mats`.

If `ops` is empty, emits a warning and returns `(mats=Matrix{Float64}[], rvec=Vector{Float64}[])`.
"""
function build_Tr(
    dofs::SystemDofs,
    ops::Vector{<:Operators},
    irvec::Vector{<:AbstractVector{<:Real}}
)
    d_int = length(dofs.valid_states)
    if isempty(ops)
        @warn "No operators found"
        return (mats=Matrix{Float64}[], delta=Vector{Float64}[])
    end
    T = eltype(map(op -> op.value, ops))

    delta = unique(Vector{Float64}.(irvec))
    mats = [zeros(T, d_int, d_int) for _ in delta]

    for (op, r) in zip(ops, irvec)
        length(op.ops) == 2 || continue
        sign, reord = _reorder_to_ca_alternating(Vector{FermionOp}(op.ops))
        a = findfirst(==(reord[1].qn), dofs.valid_states)
        b = findfirst(==(reord[2].qn), dofs.valid_states)
        idx = findfirst(==(Vector{Float64}(r)), delta)
        mats[idx][a, b] += sign * op.value
    end

    return (mats=mats, delta=delta)
end

"""
    build_Tk(T_r) -> Function

Return a closure that evaluates the kinetic Hamiltonian at a given k-point:

    T_{ab}(k) = Σ_r  T_{ab}(r) · exp(i k · r)

# Arguments
- `T_r`: Real-space hopping table from `build_Tr` (NamedTuple with `mats` and `delta`).

# Returns
A callable `T_func(k)` returning `Matrix{ComplexF64}` of shape `(d_int, d_int)`,
where `k` is a momentum vector. Returns `nothing` if `T_r.mats` is empty (no hopping
terms); the caller should skip the kinetic contribution in that case.
"""
function build_Tk(T_r)
    isempty(T_r.mats) && return nothing
    d_int = size(T_r.mats[1], 1)
    return function(k::AbstractVector{<:Real})
        T = zeros(ComplexF64, d_int, d_int)
        for (mat, r) in zip(T_r.mats, T_r.delta)
            T .+= mat .* cis(dot(k, r))
        end
        return T
    end
end


# ──────────────── Preprocessing: interaction term ────────────────

"""
    build_Vr(dofs, ops, irvec) -> NamedTuple

Parse two-body operators and build the real-space interaction table V̄^{abcd}(τ1, τ2, τ3).

Scans `irvec` for unique (τ1,τ2,τ3) triples, then accumulates interaction amplitudes
into one `d_int×d_int×d_int×d_int` array per unique triple. Non-four-body operators
in `ops` are silently skipped.

# Arguments
- `dofs::SystemDofs`: DOFs of one magnetic unit cell; `d_int = length(dofs.valid_states)`.
- `ops::Vector{<:Operators}`: Two-body operators in creation-annihilation alternating order (4 FermionOp entries);
  typically `generate_twobody(...).ops`.
- `irvec`: Per-operator unit-cell displacements `(τ1, τ2, τ3)` paired one-to-one with
  `ops`; typically `generate_twobody(...).irvec`.

# Returns
`NamedTuple (mats, taus)` where:
- `mats::Vector{Array{T,4}}`: one `d_int×d_int×d_int×d_int` array per unique (τ1,τ2,τ3);
  `mats[n][a,b,c,d]` = V̄^{abcd} for the n-th displacement triple.
- `taus::Vector{NTuple{3,Vector{Float64}}}`: the corresponding (τ1,τ2,τ3) triples,
  same order as `mats`.

# Notes
- Raw V̄ values are stored without antisymmetrization; the HF self-energy
  symmetrization is handled inside `build_heff_k!`.
- The density-density special case (all τ1=τ3=0) is detected downstream in
  `build_heff_k!` to select the real-space path.
"""
function build_Vr(
    dofs::SystemDofs,
    ops::AbstractVector{<:Operators},
    irvec::AbstractVector{<:NTuple{3}}
)
    d_int = length(dofs.valid_states)
    if isempty(ops)
        @warn "No two-body operators found"
        return (mats=Array{Float64,4}[], taus=NTuple{3,Vector{Float64}}[])
    end

    T = eltype(map(op -> op.value, ops))

    taus = unique(NTuple{3,Vector{Float64}}.(irvec))
    mats = [zeros(T, d_int, d_int, d_int, d_int) for _ in taus]

    for (op, τs) in zip(ops, irvec)
        length(op.ops) == 4 || continue
        # ops from generate_twobody are already in creation-annihilation alternating order (c†c c†c)
        a = qn2linear(dofs, op.ops[1].qn)
        b = qn2linear(dofs, op.ops[2].qn)
        c = qn2linear(dofs, op.ops[3].qn)
        d = qn2linear(dofs, op.ops[4].qn)
        idx = findfirst(==(NTuple{3,Vector{Float64}}(τs)), taus)
        mats[idx][a, b, c, d] += op.value
    end

    return (mats=mats, taus=taus)
end

"""
    build_Vk(V_r, kgrid) -> Function

Fourier-transform V̄^{abcd}(τ1, τ2, τ3) to the three-momentum interaction kernel:

    Ṽ^{abcd}(k1, k2, k3) = Σ_{τ1,τ2,τ3}  V̄^{abcd}(τ1, τ2, τ3)
                            · exp(-i k1·τ1 + i k2·τ2 - i k3·τ3)

Returns a closure `(k1_idx, k2_idx, k3_idx) -> Array{ComplexF64, 4}` that evaluates
the interaction kernel at any three k-points on demand. The fourth momentum
k4 = k1 + k3 - k2 is fixed by momentum conservation.

# Arguments
- `V_r`: Real-space interaction table from `build_Vr`.
- `kgrid`: k-grid struct providing `k_points` and `nk`.

# Returns
A callable `Ṽ(k1_idx, k2_idx, k3_idx)` returning `Array{ComplexF64, 4}`
of shape `(d_int, d_int, d_int, d_int)`.

# Notes
- Only needed for the general (non-density-density) path in `build_heff_k!`.
- For density-density interactions (all τ1=τ2, τ3=0 in V_r.entries),
  `build_heff_k!` uses real-space directly from `V_r` without calling this.
"""
function build_Vk(V_r)
    isempty(V_r.mats) && return nothing
    d_int = size(V_r.mats[1], 1)
    return function(k1::AbstractVector{<:Real}, k2::AbstractVector{<:Real}, k3::AbstractVector{<:Real})
        V = zeros(ComplexF64, d_int, d_int, d_int, d_int)
        for (mat, (τ1, τ2, τ3)) in zip(V_r.mats, V_r.taus)
            phase = cis(-dot(k1, τ1) + dot(k2, τ2) - dot(k3, τ3))
            V .+= mat .* phase
        end
        return V
    end
end

"""
    build_Uk(V_k) -> Function

Return a closure that, given two k-points `k` and `q`, assembles the
antisymmetrized HF interaction matrix of shape `(d²,d²)`:

    U[α+(β-1)d, μ+(ν-1)d] =  Ṽ^{μναβ}(k,k,q)   [Hartree 1]
                             + Ṽ^{αβμν}(q,q,k)   [Hartree 2]
                             - Ṽ^{μβαν}(k,q,q)   [Fock 1]
                             - Ṽ^{ανμβ}(q,k,k)   [Fock 2]

This is the k-space analogue of the real-space `build_U`, encoding all four
Wick contraction channels (Theory §4). The row index α+(β-1)*d corresponds
to Σ^{αβ}(q) and the column index μ+(ν-1)*d corresponds to G^{μν}(k),
both following Julia's column-major convention (i.e. `vec(M)[α+(β-1)*d] == M[α,β]`).

Once `U_func = build_Uk(V_k)`, the HF self-energy is

    Σ(q) = (1/Nk) Σ_k reshape(U_func(k, q) * vec(G(k)), d, d)

Returns `nothing` if `V_k` is `nothing`.
"""
function build_Uk(V_k)
    V_k === nothing && return nothing
    return function(k::AbstractVector{<:Real}, q::AbstractVector{<:Real})
        V1 = V_k(k, k, q)   # Ṽ^{μναβ}(k,k,q): Hartree 1
        V2 = V_k(q, q, k)   # Ṽ^{αβμν}(q,q,k): Hartree 2
        V3 = V_k(k, q, q)   # Ṽ^{μβαν}(k,q,q): Fock 1
        V4 = V_k(q, k, k)   # Ṽ^{ανμβ}(q,k,k): Fock 2
        d = size(V1, 1)
        # Assemble B[α,β,μ,ν] = V1[μ,ν,α,β] + V2[α,β,μ,ν] - V3[μ,β,α,ν] - V4[α,ν,μ,β]
        # via permutedims (permutedims(A,(p1,p2,p3,p4))[i,j,k,l] = A at A's dim pk = index in position k):
        #   V1[μ,ν,α,β] → permutedims(V1,(3,4,1,2))[α,β,μ,ν]
        #   V2[α,β,μ,ν] → no permutation
        #   V3[μ,β,α,ν] → permutedims(V3,(3,2,1,4))[α,β,μ,ν]
        #   V4[α,ν,μ,β] → permutedims(V4,(1,4,3,2))[α,β,μ,ν]
        B = permutedims(V1, (3,4,1,2)) .+ V2 .-
            permutedims(V3, (3,2,1,4)) .- permutedims(V4, (1,4,3,2))
        # reshape (d,d,d,d) → (d²,d²) column-major: row = α+(β-1)*d, col = μ+(ν-1)*d
        return reshape(B, d^2, d^2)
    end
end

# ──────────────── Real-space kernels ────────────────

"""
    _tau_case(τ1, τ2, τ3) -> Symbol

Classify a displacement triple (τ1,τ2,τ3) into one of four real-space formulation cases
(Theory §6). Priority A > B > C; a triple satisfying multiple conditions (e.g. (0,0,0))
is assigned to the highest-priority case.

- `:A` — density-density:  τ1 == τ2, τ3 == 0
- `:B` — exchange-type:    τ1 == 0,  τ2 == τ3
- `:C` — pair-hopping:     τ1 == τ3, τ2 == 0
- `:general` — none of the above; requires full three-momentum evaluation.
"""
function _tau_case(τ1, τ2, τ3)
    if τ1 == τ2 && all(iszero, τ3)
        return :A
    elseif all(iszero, τ1) && τ2 == τ3
        return :B
    elseif τ1 == τ3 && all(iszero, τ2)
        return :C
    else
        return :general
    end
end

"""
    _classify_Vr(V_r) -> NamedTuple

Partition the entries of `V_r` into four real-space formulation cases based on the
(τ1,τ2,τ3) structure of each entry (Theory §6), following priority A > B > C.

# Returns
NamedTuple with fields `A`, `B`, `C`, `general`, each a `(mats, taus)` NamedTuple:
- `.A`       — Case A: density-density (τ1=τ2=τ, τ3=0)
- `.B`       — Case B: exchange-type   (τ1=0, τ2=τ3=τ)
- `.C`       — Case C: pair-hopping    (τ1=τ3=τ, τ2=0)
- `.general` — no single-displacement structure; full Ṽ(k1,k2,k3) required.
"""
function _classify_Vr(V_r)
    T = eltype(V_r.mats[1])

    mats_A = Array{T,4}[]; taus_A = NTuple{3,Vector{Float64}}[]
    mats_B = Array{T,4}[]; taus_B = NTuple{3,Vector{Float64}}[]
    mats_C = Array{T,4}[]; taus_C = NTuple{3,Vector{Float64}}[]
    mats_G = Array{T,4}[]; taus_G = NTuple{3,Vector{Float64}}[]

    for (mat, τs) in zip(V_r.mats, V_r.taus)
        case = _tau_case(τs...)
        if case == :A
            push!(mats_A, mat); push!(taus_A, τs)
        elseif case == :B
            push!(mats_B, mat); push!(taus_B, τs)
        elseif case == :C
            push!(mats_C, mat); push!(taus_C, τs)
        else
            push!(mats_G, mat); push!(taus_G, τs)
        end
    end

    return (A=(mats=mats_A, taus=taus_A),
            B=(mats=mats_B, taus=taus_B),
            C=(mats=mats_C, taus=taus_C),
            general=(mats=mats_G, taus=taus_G))
end

"""
    build_Wr_A(mats_a, taus_a) -> NamedTuple

Assemble the Hartree and Fock real-space kernels for Case A interactions
(density-density, τ1=τ2=τ, τ3=0). See Theory §6.1.

Using free indices (a,b) and summation indices (c,d):

**Hartree** (q-independent, contracted with Ḡ^{cd}):
    Σ_H^{ab} = Σ_{cd} [W̃^{cdab}(0) + W̃^{abcd}(0)] Ḡ^{cd}
    hartree[a,b,c,d] = Σ_r [ permutedims(W(r),(3,4,1,2)) + W(r) ]
                             ─────────────────────────────
                              W^{cdab}(r)       W^{abcd}(r)

**Fock** (real-space, contracted with G^{cd}(r) then transformed to q):
    Σ_F^{ab}(q) = -Σ_r [W^{cbad}(r) + W^{adcb}(-r)] G^{cd}(r) · exp(iq·r)
Using hermiticity W^{abcd}(-r) = conj(W^{dcba}(r)), one obtains W^{adcb}(-r) = conj(W^{bcda}(r)):
    fock.mats[n][a,b,c,d] = permutedims(W(r),(3,2,1,4)) + conj(permutedims(W(r),(4,1,2,3)))
                             ────────────────────────────   ──────────────────────────────────
                              W^{cbad}(r)                   conj(W^{bcda}(r)) = W^{adcb}(-r)
    fock.delta[n] = τ (the single displacement τ1=τ2 of the taus triple)

# Returns
`(hartree, fock)` where `hartree::Matrix{T}` of shape `(d²,d²)` and
`fock::(mats=Vector{Matrix{T}}, delta)`. The contraction is
`reshape(hartree * vec(Ḡ), d, d)` and `reshape(fock.mats[n] * vec(G(r)), d, d)`.
"""
function build_Wr_A(mats_a, taus_a)
    isempty(mats_a) && return (hartree=nothing, fock=(mats=nothing, delta=nothing))
    T = ComplexF64
    d = size(mats_a[1], 1)

    hartree = zeros(T, d, d, d, d)
    delta   = unique([τ1 for (τ1, _, _) in taus_a])
    fock_mats = [zeros(T, d, d, d, d) for _ in delta]

    for (W, (τ1, _, _)) in zip(mats_a, taus_a)
        hartree .+= permutedims(W, (3,4,1,2)) .+ W
        idx = findfirst(==(τ1), delta)
        fock_mats[idx] .+= permutedims(W, (3,2,1,4)) .+ conj(permutedims(W, (4,1,2,3)))
    end

    return (hartree=reshape(hartree, d^2, d^2),
            fock=(mats=[reshape(m, d^2, d^2) for m in fock_mats], delta=delta))
end

"""
    build_Wr_B(mats_b, taus_b) -> NamedTuple

Assemble the Hartree and Fock real-space kernels for Case B interactions
(exchange-type, τ1=0, τ2=τ3=τ). See Theory §6.2. Case B is the complement of Case A.

**Hartree** (real-space, contracted with G^{cd}(r) then transformed to q):
    Σ_H^{ab}(q) = Σ_r [W^{cdab}(-r) + W^{abcd}(r)] G^{cd}(r) · exp(iq·r)
Case B hermiticity: [W^{abcd}(τ)]* = W^{badc}(-τ)  (note: -τ, unlike Case A).
Derivation of W^{cdab}(-r) = conj(W^{dcba}(r)):
  substitute a→d,b→c,c→b,d→a in the hermiticity relation:
  [W^{dcba}(τ)]* = W^{cdab}(-τ)  ⟹  W^{cdab}(-r) = conj(W^{dcba}(r)).
    hartree.mats[n][a,b,c,d] = conj(permutedims(W(r),(4,3,2,1))) + W(r)
                                ──────────────────────────────────   ──────
                                 W^{cdab}(-r)                        W^{abcd}(r)
    hartree.delta[n] = τ (= τ2 = τ3 of the taus triple)

**Fock** (q-independent, contracted with Ḡ^{cd}):
    Σ_F^{ab} = -Σ_{cd} [W̃^{cbad}(0) + W̃^{adcb}(0)] Ḡ^{cd}
    fock[a,b,c,d] = Σ_r [ permutedims(W(r),(3,2,1,4)) + permutedims(W(r),(1,4,3,2)) ]
                          ────────────────────────────   ──────────────────────────────
                           W^{cbad}(r)                   W^{adcb}(r)

# Returns
`(hartree, fock)` where `hartree::(mats=Vector{Matrix{T}}, delta)` and
`fock::Matrix{T}` of shape `(d²,d²)`. The contraction is
`reshape(hartree.mats[n] * vec(G(r)), d, d)` and `reshape(fock * vec(Ḡ), d, d)`.
"""
function build_Wr_B(mats_b, taus_b)
    isempty(mats_b) && return (hartree=(mats=nothing, delta=nothing), fock=nothing)
    T = ComplexF64
    d = size(mats_b[1], 1)

    fock  = zeros(T, d, d, d, d)
    delta = unique([τ2 for (_, τ2, _) in taus_b])
    hartree_mats = [zeros(T, d, d, d, d) for _ in delta]

    for (W, (_, τ2, _)) in zip(mats_b, taus_b)
        idx = findfirst(==(τ2), delta)
        hartree_mats[idx] .+= conj(permutedims(W, (4,3,2,1))) .+ W
        fock .+= permutedims(W, (3,2,1,4)) .+ permutedims(W, (1,4,3,2))
    end

    return (hartree=(mats=[reshape(m, d^2, d^2) for m in hartree_mats], delta=delta),
            fock=reshape(fock, d^2, d^2))
end

"""
    build_Wr_C(mats_c, taus_c) -> NamedTuple

Assemble the real-space kernels for Case C interactions
(pair-hopping, τ1=τ3=τ, τ2=0). See Theory §6.3.

The full self-energy is:

    Σ^{ab}(q) = Σ_τ Σ_{cd} K^{ab,cd}(τ) · G^{cd}(-τ) · exp(iq·τ)

where K is split into Hartree and Fock channels:

    K_H(r)[a,b,c,d] = W^{cdab}(-r) + W^{abcd}(-r)    ← Hartree (direct)
    K_F(r)[a,b,c,d] = W^{cbad}(-r) + W^{adcb}(-r)    ← Fock (exchange, applied with − sign)

Both channels contract with G(-τ) = adjoint(G(τ)), not with Ḡ.

Case C hermiticity: [W^{abcd}(τ)]* = W^{dcba}(-τ), so W^{efgh}(-r) = conj(W^{hgfe}(r)).
Each term mapped to its source at +r:
    W^{cdab}(-r) = conj(W^{badc}(r)) = conj(permutedims(W(r),(2,1,4,3)))
    W^{abcd}(-r) = conj(W^{dcba}(r)) = conj(permutedims(W(r),(4,3,2,1)))
    W^{cbad}(-r) = conj(W^{dabc}(r)) = conj(permutedims(W(r),(2,3,4,1)))
    W^{adcb}(-r) = conj(W^{bcda}(r)) = conj(permutedims(W(r),(4,1,2,3)))

# Returns
`(hartree, fock)` where each is `(mats=Vector{Matrix{T}}, delta)`.
Contraction: `reshape(K * vec(adjoint(G(τ))), d, d)` for both channels;
the Fock channel is subtracted in `build_heff_k!`.
"""
function build_Wr_C(mats_c, taus_c)
    isempty(mats_c) && return (hartree=(mats=nothing, delta=nothing),
                                fock=(mats=nothing, delta=nothing))
    T = ComplexF64
    d = size(mats_c[1], 1)

    delta        = unique([τ1 for (τ1, _, _) in taus_c])
    hartree_mats = [zeros(T, d, d, d, d) for _ in delta]
    fock_mats    = [zeros(T, d, d, d, d) for _ in delta]

    for (W, (τ1, _, _)) in zip(mats_c, taus_c)
        idx = findfirst(==(τ1), delta)
        hartree_mats[idx] .+= conj(permutedims(W, (2,1,4,3))) .+
                               conj(permutedims(W, (4,3,2,1)))
        fock_mats[idx]    .+= conj(permutedims(W, (2,3,4,1))) .+
                               conj(permutedims(W, (4,1,2,3)))
    end

    return (hartree=(mats=[reshape(m, d^2, d^2) for m in hartree_mats], delta=delta),
            fock   =(mats=[reshape(m, d^2, d^2) for m in fock_mats],    delta=delta))
end

# ──────────────── k-point utilities ────────────────

"""
    build_kpoints(unitcell_vectors, box_size) -> Vector{Vector{Float64}}

Generate the k-point grid for a D-dimensional lattice.

    k = (m1/n1)*b1 + (m2/n2)*b2 + ...  for m_i in 0:n_i-1

where b_i are reciprocal lattice vectors satisfying a_i · b_j = 2π δ_{ij}.

# Arguments
- `unitcell_vectors::Vector{Vector{Float64}}`: Direct lattice vectors [a1, a2, ...].
- `box_size::NTuple{D,Int}`: Number of unit cells along each direction.
"""
function build_kpoints(
    unitcell_vectors::Vector{Vector{Float64}},
    box_size::NTuple{D, Int}
) where D
    D == length(unitcell_vectors) ||
        error("Dimension mismatch: $(length(unitcell_vectors)) vectors for D=$D")
    A = hcat(unitcell_vectors...)   # D×D, columns = unit cell vectors
    B = 2π * inv(A)'                # D×D, columns = reciprocal vectors b_i
    kpoints = Vector{Float64}[]
    for idx in Iterators.product(ntuple(i -> 0:box_size[i]-1, D)...)
        k = sum((idx[i] / box_size[i]) .* B[:,i] for i in 1:D)
        push!(kpoints, k)
    end
    return kpoints
end

# ──────────────── Green's function utilities ────────────────

"""
    initialize_green_k(Nk, d; G_init=nothing, n_fill=nothing, rng=default_rng()) -> Array{ComplexF64, 3}

Initialize the k-space Green's function stored as `G[a, b, k_idx]` of shape `(d, d, Nk)`
(column-major: each `G[:,:,ki]` is a contiguous `d×d` matrix in memory).

If `G_init` is provided, validates shape and Hermiticity at each k and returns it.

Otherwise, each `G(k)` is a random Hermitian matrix with eigenvalues in `[0, 1]` and
mean filling `n_fill / (Nk * d)` per orbital. This initialization has physical magnitude
(unlike a small perturbation around zero), so the first SCF step produces a non-trivial
Hartree–Fock field that can break symmetry and escape paramagnetic saddle points.
If `n_fill` is not provided, eigenvalues are drawn uniformly from `[0, 1]`.

# Arguments
- `Nk::Int`: Number of k-points
- `d::Int`: Internal dimension per unit cell

# Keyword Arguments
- `G_init`: Pre-initialized G array of shape `(d, d, Nk)`, or `nothing`
- `n_fill::Union{Nothing,Int}`: Total electron number; used to set the mean occupation
  per k-point so that the initial filling is approximately correct.
- `rng`: Random number generator

# Returns
`Array{ComplexF64, 3}` of shape `(d, d, Nk)`.
"""
function initialize_green_k(
    Nk::Int,
    d::Int;
    G_init = nothing,
    n_fill::Union{Nothing, Int} = nothing,
    rng::AbstractRNG = default_rng()
)
    if G_init !== nothing
        size(G_init) == (d, d, Nk) ||
            error("G_init shape $(size(G_init)) ≠ ($d, $d, $Nk)")
        for k in 1:Nk
            Gk = @view G_init[:,:,k]
            norm(Gk - Gk') < 1e-10 || error("G_init[:,:,$k] is not Hermitian")
        end
        return ComplexF64.(G_init)
    end
    target_fill = n_fill !== nothing ? n_fill / Nk : d / 2.0
    G_k = zeros(ComplexF64, d, d, Nk)
    for k in 1:Nk
        F = svd(randn(rng, ComplexF64, d, d))
        U = F.U
        occ = rand(rng, d)
        occ .*= target_fill / sum(occ)
        occ  = clamp.(occ, 0.0, 1.0)
        G_k[:,:,k] = U * Diagonal(occ) * U'
    end
    return G_k
end

"""
    green_k_to_tau(G_k, kpoints, taus) -> Vector{Matrix{ComplexF64}}

Compute G(τ) at each displacement in `taus` via direct Fourier sum:

    G^{ab}(τ) = (1/Nk) Σ_k  G^{ab}(k) · exp(-i k · τ)

# Arguments
- `G_k::Array{ComplexF64, 3}`: Shape `(d, d, Nk)`.
- `kpoints::Vector{Vector{Float64}}`: k-point list of length Nk.
- `taus::Vector{Vector{Float64}}`: displacement vectors at which to evaluate G(r).
"""
function green_k_to_tau(
    G_k::Array{ComplexF64, 3},
    kpoints::Vector{Vector{Float64}},
    taus::Vector{Vector{Float64}}
)
    d = size(G_k, 1)
    G_taus = [zeros(ComplexF64, d, d) for _ in taus]
    green_k_to_tau!(G_taus, G_k, kpoints, taus)
    return G_taus
end

"""
    green_k_to_tau!(G_taus, G_k, kpoints, taus)

In-place version of `green_k_to_tau`. Overwrites each matrix in `G_taus`.
`G_taus` must be a `Vector` of `d×d` matrices (zeroed here before accumulation).
`G_k` has shape `(d, d, Nk)`; each `G_k[:,:,ki]` is a contiguous `d×d` slice.
"""
function green_k_to_tau!(
    G_taus::Vector{Matrix{ComplexF64}},
    G_k::Array{ComplexF64, 3},
    kpoints::Vector{Vector{Float64}},
    taus::Vector{Vector{Float64}}
)
    Nk = length(kpoints)
    for Gτ in G_taus; fill!(Gτ, zero(ComplexF64)); end
    for (ki, k) in enumerate(kpoints)
        Gk = @view G_k[:,:,ki]          # contiguous d×d block
        for (n, τ) in enumerate(taus)
            phase = cis(-dot(k, τ))
            @. G_taus[n] += Gk * phase
        end
    end
    inv_Nk = 1 / Nk
    for Gτ in G_taus; Gτ .*= inv_Nk; end
    return G_taus
end

# ──────────────── Effective Hamiltonian ────────────────

# ──────────────── τ collection helper ────────────────

"""
    _collect_taus_k(wr_A, wr_B, wr_C) -> (taus_needed, tau_idx)

Collect all displacement vectors τ needed for computing G(τ) in `build_heff_k!`,
and return a Dict mapping each τ to its index in the returned vector.
Case C Hartree taus are always included (needed even when include_fock=false).
"""
function _collect_taus_k(wr_A, wr_B, wr_C)
    taus = Vector{Float64}[]
    wr_A !== nothing && wr_A.fock.delta    !== nothing && append!(taus, wr_A.fock.delta)
    wr_B !== nothing && wr_B.hartree.delta !== nothing && append!(taus, wr_B.hartree.delta)
    wr_C !== nothing && wr_C.hartree.delta !== nothing && append!(taus, wr_C.hartree.delta)
    wr_C !== nothing && wr_C.fock.delta    !== nothing && append!(taus, wr_C.fock.delta)
    unique!(taus)
    return taus, Dict(τ => i for (i, τ) in enumerate(taus))
end

"""
    build_heff_k!(H_k, T_k_func, wr_A, wr_B, wr_C, V_k_func, G_k, kpoints,
                  G_taus_buf, g_adj_buf, f_buf, taus_needed, tau_idx; include_fock=true)

Build H_eff^{ab}(q) = T^{ab}(q) + Σ^{ab}(q) in-place for all q via direct Fourier sum.

**Self-energy (Cases A/B/C, direct sum O(N_τ × Nk × d²)):**

    Σ^{ab}(q) = Σ_τ reshape(K(τ) * vec(G(τ)), d, d) · exp(i q·τ)

where K(τ) is the (d²×d²) kernel from `build_Wr_A/B/C` and
G(τ) = (1/Nk) Σ_k G(k) exp(-ik·τ).  Case C uses G(-τ) = adjoint(G(τ)).

**General case** (`V_k_func` ≠ nothing): O(Nk² · d⁴) via `build_Uk`.

Pre-allocated buffers (`G_taus_buf`, `g_adj_buf`, `f_buf`) and the τ lookup
(`taus_needed`, `tau_idx`) must be prepared by `_collect_taus_k` + manual
allocation before the SCF loop (see `_run_scf_k`).

The inner q-loops are parallelized with `Threads.@threads`.
`include_fock=false` skips A.fock, B.fock, and Case C Fock.
Case C Hartree is always included.
"""
function build_heff_k!(
    H_k::Array{ComplexF64, 3},
    T_k_func,
    wr_A, wr_B, wr_C,
    V_k_func,
    G_k::Array{ComplexF64, 3},
    kpoints::Vector{Vector{Float64}},
    G_taus_buf::Vector{Matrix{ComplexF64}},
    g_adj_buf::Matrix{ComplexF64},
    f_buf::Vector{ComplexF64},
    taus_needed::Vector{Vector{Float64}},
    tau_idx::Dict{Vector{Float64},Int};
    include_fock::Bool = true
)
    Nk = length(kpoints)
    d  = size(G_k, 1)          # G_k layout: (d, d, Nk)

    # ── Kinetic term ──
    if T_k_func !== nothing
        Threads.@threads for qi in 1:Nk
            H_k[:,:,qi] = T_k_func(kpoints[qi])
        end
    else
        H_k .= 0
    end

    # ── G(τ) at all needed displacements (in-place) ──
    if !isempty(G_taus_buf)
        green_k_to_tau!(G_taus_buf, G_k, kpoints, taus_needed)
    end

    # ── G̅ = (1/Nk) Σ_k G(k)  [sum over last dim] ──
    G_bar = dropdims(sum(G_k, dims=3), dims=3) ./ Nk   # d×d

    # ── Inner helper: accumulate f_buf contribution to H_k for all q ──
    # f_buf is read-only during the threaded section; H_k[:,:,qi] unique per qi.
    function _accum_q!(sign::Int, τ::Vector{Float64})
        Threads.@threads for qi in 1:Nk
            phase = sign * cis(dot(kpoints[qi], τ))
            @inbounds for j in 1:d, i in 1:d
                H_k[i, j, qi] += f_buf[i + (j-1)*d] * phase
            end
        end
    end

    # ── Case A: density-density ──
    if wr_A !== nothing
        # Hartree (q-independent): +K_H · vec(G̅)
        if wr_A.hartree !== nothing
            mul!(f_buf, wr_A.hartree, vec(G_bar))
            Threads.@threads for qi in 1:Nk
                @inbounds for j in 1:d, i in 1:d
                    H_k[i, j, qi] += f_buf[i + (j-1)*d]
                end
            end
        end
        # Fock (q-dependent): -Σ_τ K(τ) · vec(G(τ)) · exp(iq·τ)
        if include_fock && wr_A.fock.mats !== nothing
            for (K, τ) in zip(wr_A.fock.mats, wr_A.fock.delta)
                mul!(f_buf, K, vec(G_taus_buf[tau_idx[τ]]))
                _accum_q!(-1, τ)
            end
        end
    end

    # ── Case B: exchange-type ──
    if wr_B !== nothing
        # Hartree (q-dependent): +Σ_τ K(τ) · vec(G(τ)) · exp(iq·τ)
        if wr_B.hartree.mats !== nothing
            for (K, τ) in zip(wr_B.hartree.mats, wr_B.hartree.delta)
                mul!(f_buf, K, vec(G_taus_buf[tau_idx[τ]]))
                _accum_q!(+1, τ)
            end
        end
        # Fock (q-independent): -K_F · vec(G̅)
        if include_fock && wr_B.fock !== nothing
            mul!(f_buf, wr_B.fock, vec(G_bar))
            Threads.@threads for qi in 1:Nk
                @inbounds for j in 1:d, i in 1:d
                    H_k[i, j, qi] -= f_buf[i + (j-1)*d]
                end
            end
        end
    end

    # ── Case C: pair-hopping (G(-τ) = adjoint(G(τ))) ──
    # Hartree channel: always. Fock channel: only if include_fock.
    if wr_C !== nothing
        if wr_C.hartree.mats !== nothing
            for (K, τ) in zip(wr_C.hartree.mats, wr_C.hartree.delta)
                g_adj_buf .= adjoint(G_taus_buf[tau_idx[τ]])
                mul!(f_buf, K, vec(g_adj_buf))
                _accum_q!(+1, τ)
            end
        end
        if include_fock && wr_C.fock.mats !== nothing
            for (K, τ) in zip(wr_C.fock.mats, wr_C.fock.delta)
                g_adj_buf .= adjoint(G_taus_buf[tau_idx[τ]])
                mul!(f_buf, K, vec(g_adj_buf))
                _accum_q!(-1, τ)
            end
        end
    end

    # ── General case: O(Nk² · d⁴) ──
    if V_k_func !== nothing
        U_func = build_Uk(V_k_func)
        for (qi, q) in enumerate(kpoints)
            for (ki, k) in enumerate(kpoints)
                U = U_func(k, q)
                @view(H_k[:,:,qi]) .+= reshape(U * vec(@view G_k[:,:,ki]), d, d) ./ Nk
            end
        end
    end
end

"""
    energy_bands(dofs, onebody, twobody, kgrid, G_k, qpoints; include_fock=true)
        -> (bands::Matrix{Float64}, H_q::Array{ComplexF64,3})

Compute strict HF band energies along arbitrary q-points from a converged G_k
on a uniform k-grid. Returns the band matrix (d × Nq) and the full H(q).
"""
function energy_bands(
    dofs::SystemDofs,
    onebody,
    twobody,
    kgrid::Vector{Vector{Float64}},
    G_k::Array{ComplexF64, 3},
    qpoints::Vector{Vector{Float64}};
    include_fock::Bool = true
)
    Nk = length(kgrid)
    d  = length(dofs.valid_states)

    T_r = build_Tr(dofs, onebody.ops, onebody.irvec)
    T_k_func = build_Tk(T_r)

    V_r = build_Vr(dofs, twobody.ops, twobody.irvec)
    classified = isempty(V_r.mats) ? nothing : _classify_Vr(V_r)
    wr_A = classified !== nothing ? build_Wr_A(classified.A.mats, classified.A.taus) : nothing
    wr_B = classified !== nothing ? build_Wr_B(classified.B.mats, classified.B.taus) : nothing
    wr_C = classified !== nothing ? build_Wr_C(classified.C.mats, classified.C.taus) : nothing
    V_k_func = (classified !== nothing && !isempty(classified.general.mats)) ?
               build_Vk((mats=classified.general.mats, taus=classified.general.taus)) : nothing

    taus_needed, tau_idx = _collect_taus_k(wr_A, wr_B, wr_C)
    G_taus_buf = [zeros(ComplexF64, d, d) for _ in taus_needed]
    green_k_to_tau!(G_taus_buf, G_k, kgrid, taus_needed)
    G_bar = dropdims(sum(G_k, dims=3), dims=3) ./ Nk

    g_adj_buf = zeros(ComplexF64, d, d)
    f_buf = zeros(ComplexF64, d * d)

    Nq = length(qpoints)
    H_q = Array{ComplexF64, 3}(undef, d, d, Nq)

    for (qi, q) in enumerate(qpoints)
        H = T_k_func !== nothing ? T_k_func(q) : zeros(ComplexF64, d, d)

        if wr_A !== nothing
            if wr_A.hartree !== nothing
                mul!(f_buf, wr_A.hartree, vec(G_bar))
                H .+= reshape(f_buf, d, d)
            end
            if include_fock && wr_A.fock.mats !== nothing
                for (K, τ) in zip(wr_A.fock.mats, wr_A.fock.delta)
                    mul!(f_buf, K, vec(G_taus_buf[tau_idx[τ]]))
                    H .+= reshape(f_buf, d, d) .* (-cis(dot(q, τ)))
                end
            end
        end

        if wr_B !== nothing
            if wr_B.hartree.mats !== nothing
                for (K, τ) in zip(wr_B.hartree.mats, wr_B.hartree.delta)
                    mul!(f_buf, K, vec(G_taus_buf[tau_idx[τ]]))
                    H .+= reshape(f_buf, d, d) .* cis(dot(q, τ))
                end
            end
            if include_fock && wr_B.fock !== nothing
                mul!(f_buf, wr_B.fock, vec(G_bar))
                H .-= reshape(f_buf, d, d)
            end
        end

        if wr_C !== nothing
            if wr_C.hartree.mats !== nothing
                for (K, τ) in zip(wr_C.hartree.mats, wr_C.hartree.delta)
                    g_adj_buf .= adjoint(G_taus_buf[tau_idx[τ]])
                    mul!(f_buf, K, vec(g_adj_buf))
                    H .+= reshape(f_buf, d, d) .* cis(dot(q, τ))
                end
            end
            if include_fock && wr_C.fock.mats !== nothing
                for (K, τ) in zip(wr_C.fock.mats, wr_C.fock.delta)
                    g_adj_buf .= adjoint(G_taus_buf[tau_idx[τ]])
                    mul!(f_buf, K, vec(g_adj_buf))
                    H .+= reshape(f_buf, d, d) .* (-cis(dot(q, τ)))
                end
            end
        end

        if V_k_func !== nothing
            U_func = build_Uk(V_k_func)
            for (ki, k) in enumerate(kgrid)
                U = U_func(k, q)
                H .+= reshape(U * vec(@view G_k[:,:,ki]), d, d) ./ Nk
            end
        end

        H_q[:,:,qi] = H
    end

    bands = Matrix{Float64}(undef, d, Nq)
    for qi in 1:Nq
        bands[:, qi] = eigvals(Hermitian(@view H_q[:,:,qi]))
    end

    return bands, H_q
end

# ──────────────── Diagonalization and occupation ────────────────

"""
    diagonalize_heff_k(H_k) -> (eigenvalues, eigenvectors)

Diagonalize H_eff(k) at each k-point independently (parallelized over threads).
`H_k` has layout `(d, d, Nk)`; each `H_k[:,:,ki]` is a contiguous Hermitian `d×d` matrix.

# Returns
- `eigenvalues::Matrix{Float64}`: Shape `(d, Nk)`, sorted ascending per k.
- `eigenvectors::Array{ComplexF64, 3}`: Shape `(d, d, Nk)`; `evecs[:,n,ki]` is the n-th eigenstate.
"""
function diagonalize_heff_k(H_k::Array{ComplexF64, 3})
    d, _, Nk = size(H_k)
    evals = Matrix{Float64}(undef, d, Nk)
    evecs = Array{ComplexF64, 3}(undef, d, d, Nk)
    Threads.@threads for ki in 1:Nk
        F = eigen(Hermitian(@view H_k[:,:,ki]))
        evals[:,ki]   = F.values
        evecs[:,:,ki] = F.vectors
    end
    return evals, evecs
end

"""
    find_chemical_potential_k(eigenvalues, n_electrons, temperature; ene_cutoff=100.0) -> Float64

Find μ such that total occupation = n_electrons.

At T=0: μ placed in the midgap between the n_electrons-th and (n_electrons+1)-th levels.
At T>0: bisection on the global Fermi-Dirac sum.
"""
function find_chemical_potential_k(
    eigenvalues::Matrix{Float64},
    n_electrons::Int,
    temperature::Float64;
    ene_cutoff::Float64 = 100.0
)
    all_evals = sort(vec(eigenvalues))
    if temperature == 0.0
        e_occ   = all_evals[n_electrons]
        e_unocc = n_electrons < length(all_evals) ? all_evals[n_electrons+1] : e_occ + 1.0
        return (e_occ + e_unocc) / 2
    end
    fermi(mu) = sum(1 / (exp(clamp((e - mu)/temperature, -ene_cutoff, ene_cutoff)) + 1)
                    for e in all_evals)
    lo, hi = all_evals[1] - 10temperature, all_evals[end] + 10temperature
    for _ in 1:100
        mid = (lo + hi) / 2
        fermi(mid) < n_electrons ? (lo = mid) : (hi = mid)
        hi - lo < 1e-12 && break
    end
    return (lo + hi) / 2
end

"""
    update_green_k(eigenvectors, eigenvalues, mu, temperature; ene_cutoff=100.0) -> Array{ComplexF64, 3}

Construct G(k) from eigenstates and Fermi-Dirac occupations:

    G^{ab}(k) = Σ_n  f(ε_{kn} - μ) · u_{a,n}(k)* · u_{b,n}(k)

where `eigenvectors[:, n, ki]` is the n-th eigenstate at k-point ki (column-major).
In matrix form: G(k) = conj(V · diag(f) · V†) = conj(V) · diag(f) · Vᵀ.
Layouts: `eigenvectors` shape `(d, d, Nk)`, `eigenvalues` shape `(d, Nk)`.
Returns `G_new` of shape `(d, d, Nk)`.
"""
function update_green_k(
    eigenvectors::Array{ComplexF64, 3},
    eigenvalues::Matrix{Float64},
    mu::Float64,
    temperature::Float64;
    ene_cutoff::Float64 = 100.0
)
    d, _, Nk = size(eigenvectors)
    G_new = Array{ComplexF64, 3}(undef, d, d, Nk)
    for ki in 1:Nk
        v = @view eigenvectors[:,:,ki]   # contiguous d×d matrix
        w = @view eigenvalues[:,ki]
        dist = temperature == 0.0 ?
            Float64.(w .<= mu + 1e-12) :
            [1 / (exp(clamp((e-mu)/temperature, -ene_cutoff, ene_cutoff)) + 1) for e in w]
        G_new[:,:,ki] = conj((v .* dist') * v')
    end
    return G_new
end

# ──────────────── Energy calculation ────────────────

"""
    calculate_energies_k(eigenvalues, mu, n_electrons, temperature, G_k, H_k, T_k_func, kpoints)

    E_band  = (1/Nk) Σ_{k,n}  ε_{kn} · f(ε_{kn} - μ)
    E_int   = -(1/2Nk) Σ_k  Re Tr[ (H_eff(k) - T(k)) · G(k) ]
    E_total = E_band + E_int
"""
function calculate_energies_k(
    eigenvalues::Matrix{Float64},
    mu::Float64,
    temperature::Float64,
    G_k::Array{ComplexF64, 3},
    H_k::Array{ComplexF64, 3},
    T_k_func,
    kpoints::Vector{Vector{Float64}};
    ene_cutoff::Float64 = 100.0
)
    d, _, Nk = size(G_k)          # layout: (d, d, Nk)
    E_band = if temperature == 0.0
        sum(eigenvalues[n,ki] for n in 1:d, ki in 1:Nk
            if eigenvalues[n,ki] <= mu + 1e-12) / Nk
    else
        sum(eigenvalues[n,ki] /
            (exp(clamp((eigenvalues[n,ki]-mu)/temperature, -ene_cutoff, ene_cutoff)) + 1)
            for n in 1:d, ki in 1:Nk) / Nk
    end
    E_int = 0.0
    for (ki, k) in enumerate(kpoints)
        T_k = T_k_func !== nothing ? T_k_func(k) : zeros(ComplexF64, d, d)
        E_int += real(tr(((@view H_k[:,:,ki]) .- T_k) * (@view G_k[:,:,ki])))
    end
    E_int *= -1 / (2Nk)
    return (band=E_band, interaction=E_int, total=E_band + E_int)
end

# ──────────────── DIIS extrapolation ────────────────

function _diis_extrapolate_k(
    G_hist::Vector{Array{ComplexF64, 3}},
    R_hist::Vector{Array{ComplexF64, 3}}
)
    m = length(G_hist)
    B = zeros(Float64, m+1, m+1)
    for i in 1:m, j in i:m
        v = real(dot(vec(R_hist[i]), vec(R_hist[j])))
        B[i,j] = v; B[j,i] = v
    end
    B[1:m, m+1] .= -1.0
    B[m+1, 1:m] .= -1.0
    rhs = zeros(Float64, m+1); rhs[m+1] = -1.0
    c = try (B \ rhs)[1:m] catch; vcat(zeros(Float64, m-1), 1.0) end
    return sum(c[i] .* G_hist[i] for i in 1:m)
end

# ──────────────── Internal SCF loop ────────────────

function _run_scf_k(
    G_k::Array{ComplexF64, 3},
    T_k_func,
    wr_A, wr_B, wr_C, V_k_func,
    kpoints::Vector{Vector{Float64}},
    n_electrons::Int,
    temperature::Float64,
    max_iter::Int,
    tol::Float64,
    diis_m::Int,
    ene_cutoff::Float64,
    include_fock::Bool,
    verbose::Bool,
    timings::Dict{String, Tuple{Int64, Int}};
    symmetry_breaking_field::Union{Nothing, Matrix{ComplexF64}} = nothing,
    n_warmup::Int = 0
)
    d, _, Nk = size(G_k)           # layout: (d, d, Nk)
    H_k = zeros(ComplexF64, d, d, Nk)
    G_hist = Array{ComplexF64, 3}[]
    R_hist = Array{ComplexF64, 3}[]
    energies = (band=0.0, interaction=0.0, total=0.0)
    mu = 0.0
    evals = Matrix{Float64}(undef, d, Nk)
    evecs = Array{ComplexF64, 3}(undef, d, d, Nk)
    converged = false
    residual  = Inf
    iter = 0

    # ── Pre-allocate SCF buffers (reused every iteration) ──
    taus_needed, tau_idx = _collect_taus_k(wr_A, wr_B, wr_C)
    G_taus_buf = [zeros(ComplexF64, d, d) for _ in taus_needed]
    g_adj_buf  = zeros(ComplexF64, d, d)
    f_buf      = zeros(ComplexF64, d^2)

    for i in 1:max_iter
        iter = i
        G_old = copy(G_k)

        t0 = Int64(time_ns())
        build_heff_k!(H_k, T_k_func, wr_A, wr_B, wr_C, V_k_func, G_k, kpoints,
                      G_taus_buf, g_adj_buf, f_buf, taus_needed, tau_idx;
                      include_fock=include_fock)
        _accum!(timings, "build_heff_k", Int64(time_ns()) - t0)

        # Symmetry-breaking warmup field: add random Hermitian field to H_k
        # for the first n_warmup iterations, then remove it.
        if symmetry_breaking_field !== nothing && i <= n_warmup
            decay = 1.0 - (i - 1) / n_warmup   # linearly decay 1 → 0
            @inbounds for ki in 1:Nk
                H_k[:, :, ki] .+= decay .* symmetry_breaking_field
            end
        end

        t0 = Int64(time_ns())
        evals, evecs = diagonalize_heff_k(H_k)
        _accum!(timings, "diagonalize_k", Int64(time_ns()) - t0)

        mu = find_chemical_potential_k(evals, n_electrons, temperature;
                                       ene_cutoff=ene_cutoff)

        t0 = Int64(time_ns())
        G_new = update_green_k(evecs, evals, mu, temperature; ene_cutoff=ene_cutoff)
        _accum!(timings, "update_green_k", Int64(time_ns()) - t0)

        residual = norm(G_new .- G_old) / (Nk * d^2)

        # Mix
        if diis_m > 0
            push!(G_hist, G_new); push!(R_hist, G_new .- G_old)
            length(G_hist) > diis_m && (popfirst!(G_hist); popfirst!(R_hist))
            G_k = length(G_hist) >= 2 ? _diis_extrapolate_k(G_hist, R_hist) : G_new
        else
            G_k = G_new
        end

        # Energies (first iter, every 10, and at convergence)
        if i == 1 || i % 10 == 0 || residual < tol
            t0 = Int64(time_ns())
            energies = calculate_energies_k(evals, mu, temperature,
                                            G_k, H_k, T_k_func, kpoints;
                                            ene_cutoff=ene_cutoff)
            _accum!(timings, "calc_energies_k", Int64(time_ns()) - t0)
        end

        ncond = real(sum(G_k[a,a,ki] for a in 1:d, ki in 1:Nk)) / Nk

        if residual < tol
            converged = true
            verbose && println(@sprintf("%s Iter %4d  res = %.3e < %.3e  CONVERGED",
                                        _now_str(), i, residual, tol))
            verbose && flush(stdout)
            break
        end
        if verbose && (i == 1 || i % 10 == 0)
            println(@sprintf("%s Iter %4d  res = %.3e  E = %+.6f  NCond = %.4f",
                             _now_str(), i, residual, energies.total, ncond))
            flush(stdout)
        end
    end

    ncond = real(sum(G_k[a,a,ki] for a in 1:d, ki in 1:Nk)) / Nk
    return (G_k=G_k, eigenvalues=evals, eigenvectors=evecs,
            energies=energies, mu=mu,
            converged=converged, iterations=iter, residual=residual, ncond=ncond)
end

# ──────────────── Timing utilities ────────────────

const _PHASE_ORDER_K = ["build_Tr", "build_Tk", "build_Vr", "initialize_green_k",
                        "build_heff_k", "diagonalize_k", "update_green_k",
                        "calc_energies_k", "solve_hfk"]

function _print_timing_table_k(timings::Dict{String, Tuple{Int64, Int}})
    W = 22
    sep = "  " * "─"^58
    println()
    println("  ── Timing Summary (k-space HF) " * "─"^27)
    println(@sprintf("  %-*s  %10s  %10s  %6s", W, "Phase", "Total", "Avg", "Calls"))
    println(sep)
    for key in filter(k -> k != "solve_hfk", _PHASE_ORDER_K)
        haskey(timings, key) || continue
        ns, cnt = timings[key]
        println(@sprintf("  %-*s  %s  %s  %6d", W, key, _fmt_ns(ns), _fmt_ns(ns ÷ cnt), cnt))
    end
    println(sep)
    if haskey(timings, "solve_hfk")
        ns, cnt = timings["solve_hfk"]
        println(@sprintf("  %-*s  %s  %s  %6d", W, "solve_hfk (total)", _fmt_ns(ns), _fmt_ns(ns ÷ cnt), cnt))
    end
    println(sep)
    println()
end

# ──────────────── Public API ────────────────

"""
    solve_hfk(dofs, onebody, twobody, kpoints, n_electrons; kwargs...)

Solve Hartree-Fock in momentum space via SCF iteration.

Each k-point is diagonalized independently (parallelized over `Threads.nthreads()` threads).
The magnetic unit cell is encoded in `dofs`; `kpoints` samples its Brillouin zone.
For broken-symmetry phases (AFM, CDW, …) pass an enlarged magnetic unit cell.

# Arguments
- `dofs::SystemDofs`: Internal DOFs of one magnetic unit cell.
- `onebody`: Result of `generate_onebody(...)`; needs `.ops` and `.irvec`.
- `twobody`: Result of `generate_twobody(...)`; needs `.ops` and `.irvec`.
- `kpoints::Vector{Vector{Float64}}`: k-points (use `build_kpoints` for uniform grids).
- `n_electrons::Int`: Total electron number across all k-points.

# Keyword Arguments
- `temperature::Float64 = 0.0`
- `max_iter::Int = 1000`
- `tol::Float64 = 1e-8`: Convergence threshold ‖ΔG‖_F / (Nk·d²).
- `diis_m::Int = 8`: DIIS history window (0 = disabled).
- `G_init = nothing`: Initial G_k of shape `(d, d, Nk)`; random if `nothing`.
- `ene_cutoff::Float64 = 100.0`
- `n_restarts::Int = 1`: Random restarts; returns lowest-energy converged result.
- `seed::Union{Nothing,Int} = nothing`
- `include_fock::Bool = true`
- `verbose::Bool = true`
- `field_strength::Float64 = 0.0`: Amplitude of the random symmetry-breaking field
  added to H_k for the first `n_warmup` iterations of each restart. The field is a
  random Hermitian matrix (same for all k, independent per restart), linearly decayed
  to zero over the warmup phase. Set to a value comparable to the hopping scale (e.g.
  `1.0`) combined with `n_restarts > 1` to escape paramagnetic saddle points without
  prior knowledge of the ordered phase.
- `n_warmup::Int = 15`: Number of warmup iterations during which the symmetry-breaking
  field is active (linearly decayed to zero). Only used when `field_strength > 0`.
  Should be well below the typical SCF convergence count so the system can converge
  freely after the field has vanished.

# Returns
NamedTuple: `G_k, eigenvalues, eigenvectors, energies, mu, kpoints, converged, iterations, residual, ncond`.
"""
function solve_hfk(
    dofs::SystemDofs,
    onebody,
    twobody,
    kpoints::Vector{Vector{Float64}},
    n_electrons::Int;
    temperature::Float64       = 0.0,
    max_iter::Int              = 1000,
    tol::Float64               = 1e-8,
    diis_m::Int                = 8,
    G_init                     = nothing,
    ene_cutoff::Float64        = 100.0,
    n_restarts::Int            = 1,
    seed::Union{Nothing, Int}  = nothing,
    include_fock::Bool         = true,
    verbose::Bool              = true,
    field_strength::Float64    = 1.0,
    n_warmup::Int              = 15
)
    solve_start = Int64(time_ns())
    timings = Dict{String, Tuple{Int64, Int}}()

    Nk = length(kpoints)
    d  = length(dofs.valid_states)

    verbose && println("="^60)
    verbose && println("Hartree-Fock SCF Solver (momentum space)")
    verbose && println("="^60)
    if verbose
        println(@sprintf("  Nk = %d,  d = %d,  n_electrons = %d,  T = %.4g",
                         Nk, d, n_electrons, temperature))
        mixing_str = diis_m > 0 ? "DIIS(m=$diis_m)" : "last-iterate"
        println(@sprintf("  mixing = %s,  tol = %.2g,  max_iter = %d",
                         mixing_str, tol, max_iter))
        n_restarts > 1 && println("  Restarts: $n_restarts")
        println("="^60); flush(stdout)
    end

    # ── Preprocessing ──
    t0 = Int64(time_ns())
    T_r = build_Tr(dofs, onebody.ops, onebody.irvec)
    _accum!(timings, "build_Tr", Int64(time_ns()) - t0)

    t0 = Int64(time_ns())
    T_k_func = build_Tk(T_r)
    _accum!(timings, "build_Tk", Int64(time_ns()) - t0)

    t0 = Int64(time_ns())
    V_r = build_Vr(dofs, twobody.ops, twobody.irvec)
    _accum!(timings, "build_Vr", Int64(time_ns()) - t0)

    if verbose
        println(@sprintf("%s  T(r): %d terms  %s", _now_str(),
                         length(T_r.mats), _fmt_ns(timings["build_Tr"][1])))
        println(@sprintf("%s  V(r): %d triples  %s", _now_str(),
                         length(V_r.mats), _fmt_ns(timings["build_Vr"][1])))
        flush(stdout)
    end

    classified = isempty(V_r.mats) ? nothing : _classify_Vr(V_r)
    wr_A = classified !== nothing ? build_Wr_A(classified.A.mats, classified.A.taus) : nothing
    wr_B = classified !== nothing ? build_Wr_B(classified.B.mats, classified.B.taus) : nothing
    wr_C = classified !== nothing ? build_Wr_C(classified.C.mats, classified.C.taus) : nothing
    V_k_func = (classified !== nothing && !isempty(classified.general.mats)) ?
               build_Vk((mats=classified.general.mats, taus=classified.general.taus)) : nothing

    @assert 0 < n_electrons <= Nk * d  "n_electrons out of range [1, $(Nk*d)]"
    @assert temperature >= 0            "temperature must be non-negative"
    @assert n_restarts >= 1             "n_restarts must be >= 1"

    rng = seed !== nothing ? MersenneTwister(seed) : default_rng()
    best_result = nothing

    for restart in 1:n_restarts
        if n_restarts > 1
            verbose && println("-"^60)
            verbose && println(_now_str() * @sprintf(" Restart %d / %d", restart, n_restarts))
            verbose && println("-"^60)
        end

        t0 = Int64(time_ns())
        G_k = G_init !== nothing && restart == 1 ?
              initialize_green_k(Nk, d; G_init=G_init) :
              initialize_green_k(Nk, d; n_fill=n_electrons, rng=rng)
        _accum!(timings, "initialize_green_k", Int64(time_ns()) - t0)
        n_restarts == 1 && verbose &&
            println(_now_str() * @sprintf(" G initialized  %s",
                                          _fmt_ns(timings["initialize_green_k"][1])))

        sbf = if field_strength > 0
            H_rand = randn(rng, ComplexF64, d, d)
            Matrix{ComplexF64}((H_rand + H_rand') .* (field_strength / 2))
        else
            nothing
        end

        result = _run_scf_k(G_k, T_k_func, wr_A, wr_B, wr_C, V_k_func,
                            kpoints, n_electrons, temperature,
                            max_iter, tol, diis_m, ene_cutoff,
                            include_fock, n_restarts > 1 ? false : verbose, timings;
                            symmetry_breaking_field=sbf, n_warmup=n_warmup)

        if n_restarts > 1 && verbose
            println(@sprintf("  Restart %d: E = %+.10f  (%s, %d iters)",
                             restart, result.energies.total,
                             result.converged ? "CONVERGED" : "NOT CONVERGED",
                             result.iterations))
        end

        if best_result === nothing ||
           (result.converged && (!best_result.converged ||
            result.energies.total < best_result.energies.total))
            best_result = result
        end
    end

    if verbose
        println("="^60)
        n_restarts > 1 && println("Best result from $n_restarts restarts:")
        if best_result.converged
            println(_now_str() * @sprintf(" SCF CONVERGED  (%d iterations)", best_result.iterations))
        else
            @warn "SCF NOT CONVERGED (residual = $(best_result.residual))"
        end
        println(@sprintf("  Band energy:        %+.10f", best_result.energies.band))
        println(@sprintf("  Interaction energy: %+.10f", best_result.energies.interaction))
        println(@sprintf("  Total energy:       %+.10f", best_result.energies.total))
        println(@sprintf("  NCond:              %.6f",   best_result.ncond))
        println(@sprintf("  μ:                  %+.10f", best_result.mu))
        total_ns = Int64(time_ns()) - solve_start
        _accum!(timings, "solve_hfk", total_ns)
        _print_timing_table_k(timings)
        flush(stdout)
    end

    return merge(best_result, (kpoints=kpoints,))
end
