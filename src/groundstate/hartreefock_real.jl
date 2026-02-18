"""
Hartree-Fock mean-field approximation for quantum many-body systems.
"""

using SparseArrays
using LinearAlgebra
using Random
using Printf
using Dates

# ──────────────── Shared internal helpers ────────────────

_in_same_block(i, j, blocks) = any(b -> i in b && j in b, blocks)

function _dict_to_coo(d::Dict{Tuple{Int,Int}, T}) where T
    rows = Vector{Int}(undef, length(d))
    cols = Vector{Int}(undef, length(d))
    vals = Vector{T}(undef, length(d))
    idx = 0
    for ((r, c), v) in d
        iszero(v) && continue
        idx += 1
        rows[idx] = r; cols[idx] = c; vals[idx] = v
    end
    resize!(rows, idx); resize!(cols, idx); resize!(vals, idx)
    return rows, cols, vals
end

# ──────────────── Build Hamiltonian matrices ────────────────

"""
    build_t_matrix(dofs, ops)

Build sparse one-body Hamiltonian (N×N) from one-body operators.
Only stores elements within the same symmetry block (from `dofs.blocks`).
Returns a sparse Hermitian matrix.
"""
function build_t_matrix(dofs::SystemDofs, ops::AbstractVector{<:Operators})
    N = total_dim(dofs)
    blocks = dofs.blocks
    onebody = filter(op -> length(op.ops) == 2 && all(o isa FermionOp for o in op.ops), ops)
    isempty(onebody) && error("No one-body operators found")
    T = typeof(first(onebody).value)
    acc = Dict{Tuple{Int,Int}, T}()
    for op in onebody
        sign, reord = _reorder_to_interall(Vector{FermionOp}(op.ops))
        i, j = qn2linear(dofs, reord[1].qn), qn2linear(dofs, reord[2].qn)
        _in_same_block(i, j, blocks) || continue
        acc[(i,j)] = get(acc, (i,j), zero(T)) + sign * op.value
    end
    t = sparse(_dict_to_coo(acc)..., N, N)
    ishermitian(t) || throw(ArgumentError("t matrix is not Hermitian"))
    return t
end

"""
    build_U_matrix(dofs, ops; include_fock=true)

Build sparse HF interaction matrix (N²×N²) from two-body operators.
Applies the antisymmetrization formula directly, skipping the V tensor.
Only stores columns where (k,l) share a symmetry block.
Returns a sparse matrix of size (N²×N²).

Set `include_fock=false` for Hartree-only (no exchange) calculation.
The 4-term decomposition is:
- Hartree: `U_{ij,kl} += V_{ijkl} + V_{klij}`
- Fock:    `U_{ij,kl} -= V_{kjil} + V_{ilkj}` (skipped if include_fock=false)
"""
function build_U_matrix(dofs::SystemDofs, ops::AbstractVector{<:Operators};
                        include_fock::Bool = true)
    N = total_dim(dofs)
    blocks = dofs.blocks
    twobody = filter(op -> length(op.ops) == 4 && all(o isa FermionOp for o in op.ops), ops)
    isempty(twobody) && error("No two-body operators found")
    T = typeof(first(twobody).value)
    acc = Dict{Tuple{Int,Int}, T}()
    add!(r, c, v) = (acc[(r,c)] = get(acc, (r,c), zero(T)) + v)
    for op in twobody
        sign, reord = _reorder_to_interall(Vector{FermionOp}(op.ops))
        i, j, k, l = (qn2linear(dofs, reord[n].qn) for n in 1:4)
        v = sign * op.value
        # Hartree terms
        _in_same_block(k, l, blocks) && add!(i+(j-1)*N, k+(l-1)*N, +v)
        _in_same_block(i, j, blocks) && add!(k+(l-1)*N, i+(j-1)*N, +v)
        # Fock (exchange) terms
        if include_fock
            _in_same_block(i, l, blocks) && add!(k+(j-1)*N, i+(l-1)*N, -v)
            _in_same_block(k, j, blocks) && add!(i+(l-1)*N, k+(j-1)*N, -v)
        end
    end
    return sparse(_dict_to_coo(acc)..., N^2, N^2)
end

# ──────────────── Timing utilities ────────────────

_now_str() = let t = Dates.now()
    @sprintf("[%02d:%02d:%02d]", Dates.hour(t), Dates.minute(t), Dates.second(t))
end

function _fmt_ns(ns::Int64)
    s = ns / 1.0e9
    s >= 1.0    && return @sprintf("%8.3f s ", s)
    s >= 1.0e-3 && return @sprintf("%8.3fms", s * 1.0e3)
    return @sprintf("%8.3fμs", s * 1.0e6)
end

function _accum!(d::Dict{String, Tuple{Int64, Int}}, key::String, ns::Int64)
    prev_ns, cnt = get(d, key, (Int64(0), 0))
    d[key] = (prev_ns + ns, cnt + 1)
end

const _PHASE_ORDER = ["build_t_matrix", "build_U_matrix", "initialize_green",
                      "build_h_eff", "diagonalize", "update_green", "calc_energies",
                      "solve_hf"]

function _print_timing_table(timings::Dict{String, Tuple{Int64, Int}}, total_ns::Int64)
    W = 20
    sep = "  " * "─"^56
    println()
    println("  ── Timing Summary " * "─"^44)
    println(@sprintf("  %-*s  %10s  %10s  %6s", W, "Phase", "Total", "Avg", "Calls"))
    println(sep)
    for key in filter(k -> k != "solve_hf", _PHASE_ORDER)
        haskey(timings, key) || continue
        ns, cnt = timings[key]
        println(@sprintf("  %-*s  %s  %s  %6d", W, key, _fmt_ns(ns), _fmt_ns(ns ÷ cnt), cnt))
    end
    println(sep)
    if haskey(timings, "solve_hf")
        ns, cnt = timings["solve_hf"]
        println(@sprintf("  %-*s  %s  %s  %6d", W, "solve_hf (total)", _fmt_ns(ns), _fmt_ns(ns ÷ cnt), cnt))
    end
    println(sep)
    println()
end

# ──────────────── Public API ────────────────

"""
    solve_hf(dofs, ops, block_occupations; kwargs...)

Solve Hartree-Fock equations using self-consistent field (SCF) iteration.

When the interaction is strong (large U/t), the HF energy landscape has multiple
local minima. Use `n_restarts > 1` to run multiple random initializations and
automatically select the lowest-energy converged solution.

# Arguments
- `dofs::SystemDofs`: System degrees of freedom
- `ops`: All operators (one-body with 2 FermionOp and two-body with 4 FermionOp)
- `block_occupations`: Particle number — `Int` for single block, `Vector{Int}` for multiple blocks

# Keyword Arguments
- `temperature::Float64 = 0.0`: Temperature (0 for ground state)
- `max_iter::Int = 1000`: Maximum SCF iterations per restart
- `tol::Float64 = 1e-6`: Convergence tolerance for Green's function residual
- `mix_alpha::Float64 = 0.5`: Mixing parameter (0 < α ≤ 1). Smaller (0.2–0.4) for strongly correlated systems.
- `G_init = nothing`: Initial Green's function for the first restart
- `ene_cutoff::Float64 = 100.0`: Energy cutoff for exp overflow at finite T
- `n_restarts::Int = 1`: Number of random restarts; returns lowest-energy converged result
- `seed::Union{Nothing, Int} = nothing`: Random seed for reproducibility
- `include_fock::Bool = true`: Include Fock exchange terms. Set to `false` for Hartree-only.
- `verbose::Bool = true`: Print iteration information

# Returns
NamedTuple: `(G, eigenvalues, eigenvectors, energies, mu_list, converged, iterations, residual, ncond, sz)`
- `ncond`: Total particle number `Σ G[i,i]`
- `sz`: Total spin Sz (only if dofs contains a `:spin` Dof with size=2, otherwise `nothing`)
"""
function solve_hf(
    dofs::SystemDofs,
    ops::AbstractVector{<:Operators},
    block_occupations;
    temperature::Float64 = 0.0,
    max_iter::Int = 1000,
    tol::Float64 = 1e-6,
    mix_alpha::Float64 = 0.5,
    G_init = nothing,
    ene_cutoff::Float64 = 100.0,
    n_restarts::Int = 1,
    seed::Union{Nothing, Int} = nothing,
    include_fock::Bool = true,
    verbose::Bool = true
)
    N = total_dim(dofs)
    blocks = dofs.blocks
    solve_start = Int64(time_ns())
    timings = Dict{String, Tuple{Int64, Int}}()

    verbose && println("="^60)
    verbose && println("Hartree-Fock SCF Solver")
    verbose && println("="^60)
    if verbose; println(_now_str() * " Building Hamiltonian  ($(length(ops)) operators)"); flush(stdout); end

    t0 = Int64(time_ns()); t_matrix = build_t_matrix(dofs, ops); _accum!(timings, "build_t_matrix", Int64(time_ns()) - t0)
    if verbose
        println(@sprintf("               t matrix: %s, nnz = %-6d  %s", string(size(t_matrix)), nnz(t_matrix), _fmt_ns(timings["build_t_matrix"][1])))
        flush(stdout)
    end

    t0 = Int64(time_ns()); U_matrix = build_U_matrix(dofs, ops, include_fock=include_fock); _accum!(timings, "build_U_matrix", Int64(time_ns()) - t0)
    if verbose
        println(@sprintf("               U matrix: %s, nnz = %-6d  %s", string(size(U_matrix)), nnz(U_matrix), _fmt_ns(timings["build_U_matrix"][1])))
        flush(stdout)
    end

    @assert ishermitian(t_matrix) "t_matrix must be Hermitian"
    @assert 0 < mix_alpha <= 1   "mix_alpha must be in (0, 1]"
    @assert temperature >= 0     "temperature must be non-negative"
    @assert n_restarts >= 1      "n_restarts must be >= 1"

    block_occ = block_occupations isa Integer ? [Int(block_occupations)] : collect(Int, block_occupations)
    @assert length(block_occ) == length(blocks) "block_occupations length must match number of blocks"
    for (i, (block, n_occ)) in enumerate(zip(blocks, block_occ))
        @assert 0 <= n_occ <= length(block) "Block $i occupation $n_occ out of range [0, $(length(block))]"
    end
    temperature > 0 && length(blocks) > 1 && @warn "Finite T with multiple blocks: each block gets its own chemical potential"

    if verbose
        println(@sprintf("  System: N = %d, blocks = %d, particles = %s (total = %d)", N, length(blocks), string(block_occ), sum(block_occ)))
        println(@sprintf("  T = %.4g,  α = %.2f,  tol = %.2g,  max_iter = %d", temperature, mix_alpha, tol, max_iter))
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
        G = G_init !== nothing && restart == 1 ? initialize_green(N, blocks, G_init=G_init) :
                                                 initialize_green(N, blocks, rng=rng)
        _accum!(timings, "initialize_green", Int64(time_ns()) - t0)
        if n_restarts == 1 && verbose
            println(_now_str() * @sprintf(" G initialized  %s", _fmt_ns(timings["initialize_green"][1])))
            flush(stdout)
        end

        result = _run_scf(G, t_matrix, U_matrix, N, blocks, block_occ,
                          temperature, max_iter, tol, mix_alpha, ene_cutoff,
                          n_restarts > 1 ? false : verbose, timings)

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

    ncond = real(tr(best_result.G))
    sz    = _calculate_sz(best_result.G, dofs)

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
        sz !== nothing && println(@sprintf("  Sz:                 %+.6f", sz))
        for (i, μ) in enumerate(best_result.mu_list)
            println(@sprintf("  μ (block %d):       %+.10f", i, μ))
        end
        total_ns = Int64(time_ns()) - solve_start
        _accum!(timings, "solve_hf", total_ns)
        _print_timing_table(timings, total_ns)
        flush(stdout)
    end

    return merge(best_result, (ncond=ncond, sz=sz))
end

# ──────────────── Internal SCF loop ────────────────

# Internal: run one SCF loop from initial G; returns NamedTuple with all results.
function _run_scf(
    G::Matrix{ComplexF64},
    t_matrix::AbstractMatrix,
    U_matrix::AbstractMatrix,
    N::Int,
    blocks::Vector{UnitRange{Int}},
    block_occ::Vector{Int},
    temperature::Float64,
    max_iter::Int,
    tol::Float64,
    mix_alpha::Float64,
    ene_cutoff::Float64,
    verbose::Bool,
    timings::Dict{String, Tuple{Int64, Int}}
)
    G = copy(G)
    G_old = copy(G)
    converged = false
    iteration = 0
    residual = Inf
    mu_list = zeros(Float64, length(blocks))
    eigenvalues  = [Vector{Float64}(undef, length(b)) for b in blocks]
    eigenvectors = [Matrix{ComplexF64}(undef, length(b), length(b)) for b in blocks]
    Nsite = N ÷ length(blocks)

    for iter in 1:max_iter
        iteration = iter
        copyto!(G_old, G)

        t0 = Int64(time_ns())
        h_eff = t_matrix + reshape(U_matrix * vec(G), N, N)
        _accum!(timings, "build_h_eff", Int64(time_ns()) - t0)

        t0 = Int64(time_ns())
        eigenvalues, eigenvectors = diagonalize_blocks!(h_eff, blocks, eigenvalues, eigenvectors)
        _accum!(timings, "diagonalize", Int64(time_ns()) - t0)

        t0 = Int64(time_ns())
        if temperature == 0.0
            G_new = update_green(eigenvectors, blocks, block_occ)
            mu_list = _calculate_fermi_level(eigenvalues, block_occ)
        else
            mu_list = find_chemical_potentials(eigenvalues, eigenvectors, blocks, block_occ, temperature, ene_cutoff=ene_cutoff)
            G_new = update_green(eigenvectors, blocks, block_occ,
                                 eigenvalues=eigenvalues, mu_list=mu_list,
                                 temperature=temperature, ene_cutoff=ene_cutoff)
        end
        _accum!(timings, "update_green", Int64(time_ns()) - t0)

        residual = norm(G_new - G_old) / (length(blocks) * Nsite * Nsite)

        if residual < tol
            converged = true
            G = G_new
            if verbose
                println(@sprintf("%s Iter %4d  res = %.3e < %.3e  CONVERGED", _now_str(), iter, residual, tol))
                flush(stdout)
            end
            break
        end

        @. G = (1 - mix_alpha) * G_old + mix_alpha * G_new

        t0 = Int64(time_ns())
        current_energies = calculate_energies(G, U_matrix, eigenvalues, block_occ,
                                              mu_list=mu_list, temperature=temperature,
                                              ene_cutoff=ene_cutoff)
        _accum!(timings, "calc_energies", Int64(time_ns()) - t0)

        if verbose && (iter % 10 == 0 || iter <= 5)
            println(@sprintf("%s Iter %4d  res = %.3e  E = %+.6f  NCond = %.4f",
                             _now_str(), iter, residual, current_energies.total, real(tr(G))))
            flush(stdout)
        end
    end

    return (G=G, eigenvalues=eigenvalues, eigenvectors=eigenvectors,
            energies=calculate_energies(G, U_matrix, eigenvalues, block_occ,
                                        mu_list=mu_list, temperature=temperature,
                                        ene_cutoff=ene_cutoff),
            mu_list=mu_list, converged=converged, iterations=iteration, residual=residual)
end

# ──────────────── Helper functions ────────────────

# Calculate total Sz if dofs contains a :spin Dof with size=2, otherwise return nothing.
# Convention: spin label 1 → +½ (up), spin label 2 → -½ (down).
function _calculate_sz(G::AbstractMatrix, dofs::SystemDofs)
    spin_idx = findfirst(d -> d.name == :spin && d.size == 2, dofs.dofs)
    spin_idx === nothing && return nothing
    Sz = 0.0
    for i in axes(G, 1)
        s = linear2qn(dofs, i).spin == 1 ? +0.5 : -0.5
        Sz += real(G[i, i]) * s
    end
    return Sz
end

"""
    initialize_green(N, blocks; G_init=nothing, rng=Random.default_rng())

Initialize Green's function (N×N). If `G_init` is provided, validates and returns it.
Otherwise, fills each block with small random numbers in `[-0.005, 0.005]` and symmetrizes.
"""
function initialize_green(
    N::Int,
    blocks::Vector{UnitRange{Int}};
    G_init=nothing,
    rng::AbstractRNG=Random.default_rng()
)
    if G_init !== nothing
        @assert size(G_init) == (N, N) "G_init must be N×N"
        @assert ishermitian(G_init)    "G_init must be Hermitian"
        return Matrix{ComplexF64}(G_init)
    end
    G = zeros(ComplexF64, N, N)
    rand_mat = rand(rng, Float64, N, N)
    for block in blocks, i in block, j in block
        G[i,j] = 0.01 * (rand_mat[i,j] - 0.5)
    end
    return (G + G') / 2
end

"""
    diagonalize_blocks!(h_eff, blocks, eigenvalues, eigenvectors)

Diagonalize `h_eff` block by block, storing results in pre-allocated arrays in-place.
Returns `(eigenvalues, eigenvectors)`.
"""
function diagonalize_blocks!(
    h_eff::AbstractMatrix,
    blocks::Vector{UnitRange{Int}},
    eigenvalues::Vector{Vector{Float64}},
    eigenvectors::Vector{Matrix{ComplexF64}}
)
    for (i, block) in enumerate(blocks)
        eig = eigen(Hermitian(h_eff[block, block]))
        eigenvalues[i]  .= eig.values
        eigenvectors[i] .= eig.vectors
    end
    return eigenvalues, eigenvectors
end

"""
    update_green(eigenvectors, blocks, block_occ; eigenvalues, mu_list, temperature, ene_cutoff)

Construct the single-particle Green's function (density matrix) G of size (N×N).

The Green's function is defined block-by-block. For block b with eigenvectors U and
occupation weights f_n, the formula is (Aoyama 2024, Eq. 9):

    G[b, b] = conj(U) * Diagonal(f) * Uᵀ
            = Σ_n f_n * conj(u_n) ⊗ u_n

where u_n is the n-th column of U (the n-th eigenstate of h_eff in block b).

**T = 0** (step occupation):
    f_n = 1  for n ≤ block_occ[b]   (lowest states occupied)
    f_n = 0  otherwise
    → G[b,b] = conj(U_occ) * U_occ^T   (U_occ = first n_occ columns of U)

**T > 0** (Fermi-Dirac occupation):
    f_n = 1 / (exp((ε_n - μ) / T) + 1)   with overflow guard: f_n = 0 if (ε_n-μ)/T > ene_cutoff
    → requires `eigenvalues` and `mu_list` (chemical potential per block)

Off-diagonal blocks (between different symmetry blocks) remain zero, reflecting
the block-diagonal structure imposed by symmetry.

# Arguments
- `eigenvectors`: Eigenvectors for each block, shape (block_size × block_size)
- `blocks`: Block index ranges (e.g. spin-up = 1:N/2, spin-down = N/2+1:N)
- `block_occ`: Number of particles in each block

# Keyword Arguments
- `eigenvalues`: Eigenvalues of h_eff for each block (required for T > 0)
- `mu_list`: Chemical potential for each block (required for T > 0)
- `temperature`: Temperature; 0.0 selects T=0 branch
- `ene_cutoff`: Cutoff to prevent exp overflow: if (ε-μ)/T > cutoff, f = 0

# Returns
`Matrix{ComplexF64}` of size (N×N), Hermitian and block-diagonal.
"""
function update_green(
    eigenvectors::Vector{Matrix{ComplexF64}},
    blocks::Vector{UnitRange{Int}},
    block_occ::Vector{Int};
    eigenvalues::Union{Nothing,Vector{Vector{Float64}}} = nothing,
    mu_list::Union{Nothing,Vector{Float64}} = nothing,
    temperature::Float64 = 0.0,
    ene_cutoff::Float64 = 100.0
)
    N = sum(length(b) for b in blocks)
    G = zeros(ComplexF64, N, N)
    if temperature == 0.0
        # T=0: f_n = θ(n_occ - n), i.e. occupy the lowest n_occ eigenstates
        for (idx, n_occ) in enumerate(block_occ)
            n_occ == 0 && continue
            U_occ = eigenvectors[idx][:, 1:n_occ]          # (block_size × n_occ)
            G[blocks[idx], blocks[idx]] .= conj(U_occ) * transpose(U_occ)
        end
    else
        # T>0: f_n = Fermi-Dirac, with cutoff to avoid exp overflow
        fermi(ε, μ) = (ε-μ)/temperature > ene_cutoff ? 0.0 : 1.0/(exp((ε-μ)/temperature)+1.0)
        for (idx, block) in enumerate(blocks)
            f = fermi.(eigenvalues[idx], mu_list[idx])      # occupation weights (n_states,)
            U = eigenvectors[idx]                            # (block_size × block_size)
            G[block, block] .= conj(U) * Diagonal(f) * transpose(U)
        end
    end
    return G
end

# μ = midpoint of HOMO-LUMO gap at T=0
function _calculate_fermi_level(eigenvalues::Vector{Vector{Float64}}, block_occ::Vector{Int})
    [let n = n_occ, e = evals
         n == 0          ? e[1] - 1.0 :
         n == length(e)  ? e[end] + 1.0 :
                           (e[n] + e[n+1]) / 2.0
     end for (evals, n_occ) in zip(eigenvalues, block_occ)]
end

"""
    find_chemical_potentials(eigenvalues, eigenvectors, blocks, block_occ, temperature; ene_cutoff=100.0)

Find chemical potential μ for each symmetry block at finite temperature, by enforcing
particle number conservation:

    Σ_n  [Σ_i |U_in|²] · f(ε_n, μ) = N_block

where f(ε, μ) = 1/(1 + exp((ε-μ)/T)) is the Fermi-Dirac distribution,
and Σ_i |U_in|² is the norm of eigenstate n (= 1 for normalized eigenvectors).

Denoting the left-hand side as n(μ), we seek the root of Δn(μ) = n(μ) - N_block = 0.

**Algorithm** (mirrors uhfr.py):
1. **Bisection** (robust, always converges if root is bracketed):
   - Bracket: [ε_min, ε_max] of the block's eigenvalue spectrum
   - If Δn(ε_min) · Δn(ε_max) ≥ 0, root is not bracketed → fall through to Newton
2. **Newton's method** (fast convergence near root, fallback when bisection fails):
   - Initial guess: μ = ε_min
   - Newton step: μ ← μ - Δn(μ) / Δn'(μ)
   - Derivative: Δn'(μ) = Σ_n [Σ_i |U_in|²] · f'(ε_n, μ)
     where f'(ε, μ) = -exp(x) / (T·(exp(x)+1)²), x = (ε-μ)/T
3. **Error** if both methods fail (should not happen for physical inputs)

Overflow protection: if (ε-μ)/T > `ene_cutoff`, f = 0 (avoids exp overflow for
deeply unoccupied states at low T).

# Arguments
- `eigenvalues`: Eigenvalues of h_eff for each block (must be sorted ascending)
- `eigenvectors`: Eigenvectors for each block (columns = eigenstates)
- `blocks`: Block index ranges
- `block_occ`: Target particle number for each block
- `temperature`: Temperature (must be > 0)
- `ene_cutoff`: Overflow cutoff for Fermi-Dirac (default 100.0, same as uhfr.py)

# Returns
`Vector{Float64}` of chemical potentials, one per block.
"""
function find_chemical_potentials(
    eigenvalues::Vector{Vector{Float64}},
    eigenvectors::Vector{Matrix{ComplexF64}},
    blocks::Vector{UnitRange{Int}},
    block_occ::Vector{Int},
    temperature::Float64;
    ene_cutoff::Float64 = 100.0
)
    @assert temperature > 0

    # Fermi-Dirac and its derivative w.r.t. μ, with overflow guard
    fermi(ε, μ)  = (ε-μ)/temperature > ene_cutoff ? 0.0 : 1.0/(exp((ε-μ)/temperature)+1.0)
    dfermi(ε, μ) = let x=(ε-μ)/temperature
        abs(x) > ene_cutoff ? 0.0 : -exp(x)/(temperature*(exp(x)+1.0)^2)
    end

    # Δn(μ) = n(μ) - n_target  and its derivative
    # n_eigen[j] = Σ_i |U_ij|² (= 1 for normalized columns, but computed explicitly for safety)
    delta_n(evals, evecs, μ, n_occ) =
        dot(vec(sum(abs2, evecs, dims=1)), fermi.(evals, μ)) - n_occ
    ddelta_n(evals, evecs, μ) =
        dot(vec(sum(abs2, evecs, dims=1)), dfermi.(evals, μ))

    # Step 1: Bisection — reliable if root is bracketed within [ε_min, ε_max]
    function find_mu_bisection(evals, evecs, n_occ; tol=2e-12, max_iter=100)
        f1 = delta_n(evals, evecs, evals[1], n_occ)
        f2 = delta_n(evals, evecs, evals[end], n_occ)
        f1 * f2 >= 0 && return (nothing, false)    # root not bracketed
        lo, hi = evals[1], evals[end]
        for _ in 1:max_iter
            mid = (lo + hi) / 2
            fm = delta_n(evals, evecs, mid, n_occ)
            abs(fm) < tol && return (mid, true)
            fm > 0 ? (hi = mid) : (lo = mid)
        end
        return ((lo+hi)/2, false)
    end

    # Step 2: Newton's method — fast convergence, used as fallback
    function find_mu_newton(evals, evecs, n_occ; tol=1.48e-8, max_iter=50)
        μ = evals[1]                               # initial guess: lowest eigenvalue
        for _ in 1:max_iter
            dn = delta_n(evals, evecs, μ, n_occ)
            abs(dn) < tol && return (μ, true)
            ddn = ddelta_n(evals, evecs, μ)
            abs(ddn) < 1e-14 && return (μ, false)  # flat derivative, cannot continue
            μ -= dn / ddn
        end
        return (μ, false)
    end

    mu_list = Vector{Float64}(undef, length(blocks))
    for (i, (evals, evecs, n_occ)) in enumerate(zip(eigenvalues, eigenvectors, block_occ))
        μ, ok = find_mu_bisection(evals, evecs, n_occ)
        ok || ((μ, ok) = find_mu_newton(evals, evecs, n_occ))
        ok || error("find_chemical_potentials: not converged for block $i (N_occ=$n_occ)")
        mu_list[i] = μ
    end
    return mu_list
end

"""
    calculate_energies(G, U_matrix, eigenvalues, block_occ; mu_list, temperature, ene_cutoff)

Calculate HF total energy components:

**T = 0:**
- `E_band = Σ_{n∈occ} εₙ`

**T > 0** (grand potential formula, mirrors uhfr.py):
- `E_band = Σ_b [μ_b · N_b - T · Σ_n ln(1 + exp(-(εₙ - μ_b)/T))]`
- This reduces to the T=0 formula in the zero-temperature limit.

**Interaction:**
- `E_int = -½ vec(G)ᵀ U vec(G)`

Returns `NamedTuple (band, interaction, total)`.
"""
function calculate_energies(
    G::AbstractMatrix,
    U_matrix::AbstractMatrix,
    eigenvalues::Vector{Vector{Float64}},
    block_occ::Vector{Int};
    mu_list::Union{Nothing,Vector{Float64}} = nothing,
    temperature::Float64 = 0.0,
    ene_cutoff::Float64 = 100.0
)
    G_vec = vec(G)
    if temperature == 0.0 || mu_list === nothing
        E_band = sum(sum(evals[1:n_occ]) for (evals, n_occ) in zip(eigenvalues, block_occ))
    else
        # Finite T: grand potential band contribution (same convention as uhfr.py)
        # Ω_band = Σ_b [μ_b · N_b - T · Σ_n ln(1 + exp(-(εₙ - μ_b)/T))]
        E_band = 0.0
        for (evals, μ, n_occ) in zip(eigenvalues, mu_list, block_occ)
            log_terms = map(ε -> begin
                x = -(ε - μ) / temperature
                x < ene_cutoff ? log1p(exp(x)) : x   # log1p for numerical stability
            end, evals)
            E_band += μ * n_occ - temperature * sum(log_terms)
        end
    end
    E_int = -0.5 * real(dot(G_vec, U_matrix * G_vec))
    return (band=Float64(E_band), interaction=E_int, total=Float64(E_band)+E_int)
end
