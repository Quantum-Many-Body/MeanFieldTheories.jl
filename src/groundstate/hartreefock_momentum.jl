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

using SparseArrays
using LinearAlgebra
using Random
using Printf
using Dates
using FFTW


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
  `build_heff_k!` to select the FFT convolution path.
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
  `build_heff_k!` uses FFT convolution directly from `V_r` without calling this.
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

# ──────────────── Green's function utilities ────────────────

"""
    initialize_green_k(Nk, d_int; G_init=nothing, rng=Random.default_rng()) -> Array{ComplexF64, 3}

Initialize the k-space Green's function G[k_idx, a, b] of shape (Nk, d_int, d_int).

If `G_init` is provided, validates shape and Hermiticity at each k and returns it.
Otherwise fills each G(k) with small random Hermitian perturbation around zero.

# Arguments
- `Nk::Int`: Number of k-points
- `d_int::Int`: Internal dimension per unit cell

# Keyword Arguments
- `G_init`: Pre-initialized G array of shape (Nk, d_int, d_int), or `nothing`
- `rng`: Random number generator

# Returns
`Array{ComplexF64, 3}` of shape `(Nk, d_int, d_int)`.
"""
function initialize_green_k(
    Nk::Int,
    d_int::Int;
    G_init = nothing,
    rng::AbstractRNG = Random.default_rng()
)
  
end

"""
    green_k_to_r(G_k, kgrid) -> Array{ComplexF64, 3}

Transform G_{ab}(k) to real-space G_{ab}(r) via inverse FFT:

    G_{ab}(r) = (1/Nk) Σ_k  G_{ab}(k) · exp(-i k · r)

# Arguments
- `G_k::Array{ComplexF64, 3}`: Shape (Nk, d_int, d_int)
- `kgrid`: k-grid with grid_shape for multi-dimensional IFFT

# Returns
`Array{ComplexF64, 3}` of shape (Nk, d_int, d_int), where the first index is
now a real-space displacement index (same ordering as k-grid by duality).
"""
function green_k_to_r(G_k::Array{ComplexF64, 3}, kgrid)
  
end

# ──────────────── Effective Hamiltonian ────────────────

"""
    build_heff_k!(H_k, T_k, V_r, G_k, G_r, kgrid; include_fock=true)

Build the effective single-particle Hamiltonian H_eff(k) in-place:

    H_eff^{αβ}(q) = T^{αβ}(q) + Σ^{αβ}(q)

The self-energy Σ^{αβ}(q) is computed from `V_r` via one of two paths,
selected automatically based on the structure of `V_r.entries`:

**Density-density path** (all τ1=τ2, τ3=0 in `V_r.entries`):

- Hartree (q-independent):

      Σ_H^{αβ} = Σ_{μν} W^{μναβ}(r) G^{μν}(r=0)   [summed over r; = W̃^{μναβ}(0) Ḡ^{μν}]
               = Σ_{μν} W^{αβμν}(r) G^{μν}(r=0)   [symmetry: both terms are equal]

- Fock (q-dependent, FFT-accelerated, O(Nk log Nk)):

      Σ_F^{αβ}(q) = -FFT_r→q [ Σ_{μν} ½[W^{μβαν}(r) + W^{ανμβ}(r)] G^{μν}(r) ]

**General path** (non-zero τ1 or τ3 present; direct O(Nk²·d⁴) summation):

    Σ^{αβ}(q) = (1/2N) Σ_k Σ_{μν} [
        Ṽ^{μναβ}(k,k,q) + Ṽ^{αβμν}(q,q,k)   (Hartree)
       -Ṽ^{μβαν}(k,q,q) - Ṽ^{ανμβ}(q,k,k)   (Fock)
    ] G^{μν}(k)

where Ṽ is obtained from `build_Vk(V_r, kgrid)`.

# Arguments
- `H_k::Array{ComplexF64, 3}`: Output (Nk, d_int, d_int), modified in-place
- `T_k::Array{ComplexF64, 3}`: Kinetic term (Nk, d_int, d_int)
- `V_r`: Real-space interaction table from `build_Vr`
- `G_k::Array{ComplexF64, 3}`: Current G^{αβ}(k), shape (Nk, d_int, d_int)
- `G_r::Array{ComplexF64, 3}`: Current G^{αβ}(r), shape (Nk, d_int, d_int)
- `kgrid`: k-grid struct (needed for `build_Vk` in the general path)

# Keyword Arguments
- `include_fock::Bool = true`: Include Fock exchange (set false for Hartree-only)
"""
function build_heff_k!(
    H_k::Array{ComplexF64, 3},
    T_k::Union{Nothing, Array{ComplexF64, 3}},
    V_r,
    G_k::Array{ComplexF64, 3},
    G_r::Array{ComplexF64, 3},
    kgrid;
    include_fock::Bool = true
)

end

# ──────────────── Diagonalization and occupation ────────────────

"""
    diagonalize_heff_k(H_k) -> (eigenvalues, eigenvectors)

Diagonalize H_eff(k) at each k-point independently.

# Arguments
- `H_k::Array{ComplexF64, 3}`: Shape (Nk, d_int, d_int); H_k[k,:,:] must be Hermitian

# Returns
- `eigenvalues::Matrix{Float64}`: Shape (Nk, d_int); sorted ascending at each k
- `eigenvectors::Array{ComplexF64, 3}`: Shape (Nk, d_int, d_int);
  `eigenvectors[k, :, n]` is the n-th eigenstate at k-point k
"""
function diagonalize_heff_k(H_k::Array{ComplexF64, 3})
  
end

"""
    find_chemical_potential_k(eigenvalues, n_electrons, temperature; ene_cutoff=100.0) -> Float64

Find the global chemical potential μ enforcing total electron number:

    Σ_{k,n} f(ε_{kn}, μ) = n_electrons

Uses bisection on the global spectrum.

# Arguments
- `eigenvalues::Matrix{Float64}`: Shape (Nk, d_int); all band energies
- `n_electrons::Int`: Target total electron count (over all k-points)
- `temperature::Float64`: Temperature (0 for ground state, uses midgap)

# Keyword Arguments
- `ene_cutoff::Float64 = 100.0`: Overflow guard for Fermi-Dirac

# Returns
`Float64` chemical potential μ.

# Notes
In momentum-space HF, all k-points share a single μ (unlike real-space HF with
per-block μ), because particle number conservation is global.
"""
function find_chemical_potential_k(
    eigenvalues::Matrix{Float64},
    n_electrons::Int,
    temperature::Float64;
    ene_cutoff::Float64 = 100.0
)
  
end

"""
    update_green_k(eigenvectors, eigenvalues, mu, temperature; ene_cutoff=100.0) -> Array{ComplexF64, 3}

Construct the new Green's function G_{ab}(k) from eigenstates and occupations:

    G_{ab}(k) = Σ_n  f(ε_{kn}, μ) · u_{ka,n}^* · u_{kb,n}

where u_{k,n} is the n-th eigenvector at k and f is the Fermi-Dirac distribution.

# Arguments
- `eigenvectors::Array{ComplexF64, 3}`: Shape (Nk, d_int, d_int)
- `eigenvalues::Matrix{Float64}`: Shape (Nk, d_int)
- `mu::Float64`: Chemical potential
- `temperature::Float64`: Temperature

# Returns
`Array{ComplexF64, 3}` of shape (Nk, d_int, d_int).
"""
function update_green_k(
    eigenvectors::Array{ComplexF64, 3},
    eigenvalues::Matrix{Float64},
    mu::Float64,
    temperature::Float64;
    ene_cutoff::Float64 = 100.0
)
  
end

# ──────────────── Energy calculation ────────────────

"""
    calculate_energies_k(G_k, G_r, T_k, W_r_sym, eigenvalues, mu, n_electrons, temperature; ene_cutoff=100.0)

Calculate HF total energy in momentum space.

**Band energy (T = 0):**

    E_band = (1/Nk) Σ_{k,n∈occ} ε_{kn}

**Band energy (T > 0, grand potential):**

    E_band = μ · n_electrons - T · (1/Nk) Σ_{k,n} ln(1 + exp(-(ε_{kn} - μ)/T))

**Interaction energy (double-counting correction):**

    E_int = -½ · (1/Nk) Σ_k Tr[ Σ^HF(k) · G(k) ]

where Σ^HF = Σ^H + Σ^F is the total self-energy (Hartree + Fock), so that
E_total = E_band + E_int avoids double-counting the interaction.

# Returns
`NamedTuple (band, interaction, total)` of `Float64`.
"""
function calculate_energies_k(
    G_k::Array{ComplexF64, 3},
    G_r::Array{ComplexF64, 3},
    T_k::Array{ComplexF64, 3},
    V_r,
    eigenvalues::Matrix{Float64},
    mu::Float64,
    n_electrons::Int,
    temperature::Float64;
    ene_cutoff::Float64 = 100.0
)
  
end

# ──────────────── DIIS for momentum-space G ────────────────

# DIIS extrapolation for the k-space Green's function.
# G_hist and R_hist store the last m iterates and residuals, reshaped as matrices.
# Falls back to most-recent iterate when the DIIS matrix is near-singular.
function _diis_extrapolate_k(
    G_hist::Vector{Array{ComplexF64, 3}},
    R_hist::Vector{Array{ComplexF64, 3}}
)

end

# ──────────────── Internal SCF loop ────────────────

# Internal: run one SCF loop from initial G_k; returns NamedTuple with all results.
function _run_scf_k(
    G_k::Array{ComplexF64, 3},
    T_k::Union{Nothing, Array{ComplexF64, 3}},
    V_r,
    kgrid,
    n_electrons::Int,
    temperature::Float64,
    max_iter::Int,
    tol::Float64,
    mix_alpha::Float64,
    diis_m::Int,
    ene_cutoff::Float64,
    include_fock::Bool,
    verbose::Bool,
    timings::Dict{String, Tuple{Int64, Int}}
)

end

# ──────────────── Timing utilities (mirrors hartreefock_real.jl) ────────────────

const _PHASE_ORDER_K = ["build_Tr", "build_Tk", "build_Vr", "initialize_green_k",
                        "build_heff_k", "diagonalize_k", "update_green_k",
                        "calc_energies_k", "solve_hfk"]

function _print_timing_table_k(timings::Dict{String, Tuple{Int64, Int}}, total_ns::Int64)
    W = 22
    sep = "  " * "─"^58
    println()
    println("  ── Timing Summary (k-space HF) " * "─"^32)
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
    solve_hfk(dofs, lattice, ops, n_electrons; kwargs...)

Solve Hartree-Fock equations in momentum space using self-consistent field (SCF) iteration.

Exploits translational symmetry: the k-space Green's function G_{ab}(k) is block-diagonal
in k, so each k-point is diagonalized independently in O(d³) instead of O(N³).

The unit cell (and thus the k-grid) is fixed by `lattice`. If the ground state is expected
to spontaneously break translational symmetry (antiferromagnetism, CDW, etc.), pass a
`lattice` constructed with the **enlarged magnetic unit cell** so that the ansatz
G_{ab}(k) δ_{k,k'} remains valid for the broken-symmetry phase.

# Arguments
- `dofs::SystemDofs`: Full system DOFs. Position DOFs must match `lattice.position_dofs`.
- `lattice::Lattice`: Lattice structure with `supercell_vectors` set. Defines the k-grid and
  spatial structure needed to decompose operators into T(k) and W(r).
- `ops`: All operators: one-body (2 FermionOp) and two-body (4 FermionOp).
- `n_electrons::Int`: Total electron number (summed over all k-points and bands).

# Keyword Arguments
- `temperature::Float64 = 0.0`: Temperature. 0 selects T=0 step occupation; >0 uses Fermi-Dirac.
- `max_iter::Int = 1000`: Maximum SCF iterations per restart.
- `tol::Float64 = 1e-6`: Convergence tolerance: ‖G_new - G_old‖_F / (Nk·d²).
- `mix_alpha::Float64 = 0.5`: Linear mixing parameter (0 < α ≤ 1).
- `diis_m::Int = 8`: DIIS history window. Set to 0 to disable DIIS.
- `G_init = nothing`: Initial G[k, a, b] array of shape (Nk, d_int, d_int). If `nothing`,
  a random Hermitian initialization is used.
- `ene_cutoff::Float64 = 100.0`: Overflow guard for Fermi-Dirac at low T.
- `n_restarts::Int = 1`: Number of random restarts. Returns the lowest-energy converged result.
- `seed::Union{Nothing, Int} = nothing`: Random seed for reproducibility.
- `include_fock::Bool = true`: Include Fock exchange. Set false for Hartree-only.
- `verbose::Bool = true`: Print iteration information and timing summary.

# Returns
`NamedTuple` with fields:
- `G_k::Array{ComplexF64, 3}`: Converged G_{ab}(k), shape (Nk, d_int, d_int)
- `G_r::Array{ComplexF64, 3}`: G_{ab}(r) = IFFT[G_{ab}(k)], shape (Nk, d_int, d_int)
- `eigenvalues::Matrix{Float64}`: Band energies, shape (Nk, d_int), sorted per k
- `eigenvectors::Array{ComplexF64, 3}`: Eigenstates, shape (Nk, d_int, d_int)
- `energies::NamedTuple`: `(band, interaction, total)` energies
- `mu::Float64`: Chemical potential
- `kgrid`: k-grid used (for post-processing)
- `converged::Bool`: Whether SCF converged within `max_iter`
- `iterations::Int`: Number of SCF iterations performed
- `residual::Float64`: Final residual ‖ΔG‖ / (Nk·d²)
- `ncond::Float64`: Total electron number Σ_k Tr[G(k)] / Nk (should equal n_electrons)

# Examples
```julia
# 2D Hubbard model on 8×8 lattice, half-filling, with spin blocks
dofs = SystemDofs([Dof(:site, 64), Dof(:spin, 2)], sortrule=[[2], 1])
unitcell = Lattice([Dof(:site, 1)], [QN(site=1)], [[0.0, 0.0]])
lattice = Lattice(unitcell, [[1.0, 0.0], [0.0, 1.0]], (8, 8))

# Build operators from bonds
nn_bonds = bonds(lattice, (:p, :p), 1)
onebody  = generate_onebody(magcell_dofs, nn_bonds, -1.0)
twobody  = generate_twobody(magcell_dofs, ...)

result = solve_hfk(dofs, lattice, onebody, twobody, 64;  # half-filling: 64 electrons on 64 sites
                   temperature=0.0, n_restarts=5, seed=42)
println("Total energy: ", result.energies.total)
println("Converged:    ", result.converged)
```
"""
function solve_hfk(
    dofs::SystemDofs,
    lattice::Lattice,
    onebody::NamedTuple,
    twobody::AbstractVector{<:Operators},
    n_electrons::Int;
    temperature::Float64 = 0.0,
    max_iter::Int = 1000,
    tol::Float64 = 1e-6,
    mix_alpha::Float64 = 0.5,
    diis_m::Int = 8,
    G_init = nothing,
    ene_cutoff::Float64 = 100.0,
    n_restarts::Int = 1,
    seed::Union{Nothing, Int} = nothing,
    include_fock::Bool = true,
    verbose::Bool = true
)
    solve_start = Int64(time_ns())
    timings = Dict{String, Tuple{Int64, Int}}()

    verbose && println("="^60)
    verbose && println("Hartree-Fock SCF Solver (momentum space)")
    verbose && println("="^60)

    # ── Build k-grid ──
    kgrid = build_kgrid(lattice)
    d_int = length(dofs.valid_states)
    Nk    = kgrid.nk

    if verbose
        println(@sprintf("  k-grid: Nk = %d,  d_int = %d,  N = %d", Nk, d_int, Nk * d_int))
        println(@sprintf("  n_electrons = %d,  T = %.4g", n_electrons, temperature))
        println(_now_str() * " Building T(k) and W(r)  ($(length(onebody.ops)) + $(length(twobody)) operators)")
        flush(stdout)
    end

    # ── Preprocessing ──
    t0 = Int64(time_ns())
    T_r = build_Tr(dofs, onebody.ops, onebody.irvec)
    _accum!(timings, "build_Tr", Int64(time_ns()) - t0)

    t0 = Int64(time_ns())
    T_k = build_Tk(T_r, kgrid)
    _accum!(timings, "build_Tk", Int64(time_ns()) - t0)
    if verbose
        tk_info = T_k === nothing ? "nothing (no hopping)" : string(size(T_k))
        println(@sprintf("               T(k): %s  %s", tk_info, _fmt_ns(timings["build_Tk"][1])))
    end

    t0 = Int64(time_ns())
    V_r = build_Vr(dofs, lattice, twobody)
    _accum!(timings, "build_Vr", Int64(time_ns()) - t0)
    verbose && println(@sprintf("               V(r): %d entries  %s", length(V_r.entries), _fmt_ns(timings["build_Vr"][1])))

    # ── Validation ──
    @assert 0 < mix_alpha <= 1  "mix_alpha must be in (0, 1]"
    @assert temperature >= 0    "temperature must be non-negative"
    @assert n_restarts >= 1     "n_restarts must be >= 1"
    @assert 0 < n_electrons <= Nk * d_int "n_electrons out of range"

    if verbose
        mixing_str = diis_m > 0 ? "DIIS(m=$diis_m)" : "linear(α=$(mix_alpha))"
        println(@sprintf("  mixing = %s,  tol = %.2g,  max_iter = %d", mixing_str, tol, max_iter))
        n_restarts > 1 && println("  Restarts: $n_restarts")
        println("="^60)
    end

    rng = seed !== nothing ? MersenneTwister(seed) : Random.default_rng()
    best_result = nothing

    for restart in 1:n_restarts
        if n_restarts > 1 && verbose
            println("-"^60)
            println(_now_str() * @sprintf(" Restart %d / %d", restart, n_restarts))
            println("-"^60)
        end

        t0 = Int64(time_ns())
        G_k = G_init !== nothing && restart == 1 ?
              initialize_green_k(Nk, d_int, G_init=G_init) :
              initialize_green_k(Nk, d_int, rng=rng)
        _accum!(timings, "initialize_green_k", Int64(time_ns()) - t0)

        result = _run_scf_k(G_k, T_k, V_r, kgrid, n_electrons,
                            temperature, max_iter, tol, mix_alpha, diis_m, ene_cutoff,
                            include_fock, n_restarts > 1 ? false : verbose, timings)

        if n_restarts > 1 && verbose
            println(@sprintf("  Restart %d: E = %+.10f  (%s, %d iters)",
                             restart, result.energies.total,
                             result.converged ? "CONVERGED" : "NOT CONVERGED", result.iterations))
        end

        if best_result === nothing ||
           (result.converged && (!best_result.converged || result.energies.total < best_result.energies.total))
            best_result = result
        end
    end

    ncond = real(sum(best_result.G_k[k, a, a] for k in 1:Nk, a in 1:d_int)) / Nk

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
        println(@sprintf("  NCond:              %.6f",   ncond))
        println(@sprintf("  μ:                  %+.10f", best_result.mu))
        total_ns = Int64(time_ns()) - solve_start
        _accum!(timings, "solve_hfk", total_ns)
        _print_timing_table_k(timings, total_ns)
        flush(stdout)
    end

    return merge(best_result, (ncond=ncond, kgrid=kgrid))
end
