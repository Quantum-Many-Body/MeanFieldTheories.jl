"""
Hubbard model on the simple cubic lattice at half-filling: finite-temperature
AFM transition.

Model:
  H = -t Σ_{<ij>,σ} c†_{iσ}c_{jσ} + U Σ_i n_{i↑}n_{i↓}

Magnetic unit cell: 2×2×2 cubic supercell (8 sites, d = 16 with spin).
Real-space grid: 12×12×12 sites → k-grid: 6×6×6 = 216 k-points.
(Matches CellShape=[12,12,12], SubShape=[2,2,2] in the reference.)

This example reproduces Fig. 6 of reference:
  (a) mz vs T/t for U/t = 4, 8, 12, 16
  (b) T_Néel vs U/t

Run:
    julia --project=examples -t 8 examples/Cubic_AFM/run.jl
"""

using Printf
using LinearAlgebra
using MeanFieldTheories

# ── Parameters ────────────────────────────────────────────────────────────────
const t      = 1.0
const U_vals = [4.0, 8.0, 12.0, 16.0]

# Temperature grids: 31 uniform points per U, upper limit matches reference
const T_max = Dict(4.0 => 1.0, 8.0 => 3.0, 12.0 => 5.0, 16.0 => 7.0)
T_scan(U) = collect(range(0.0, T_max[U], length=31))   # low → high

# ── Magnetic unit cell: 2×2×2 cubic supercell ────────────────────────────────
# 8 sites at all corners (i,j,k) with i,j,k ∈ {0,1}
# Ordering: iterate k outer, j middle, i inner
const site_coords_3d = [[Float64(i), Float64(j), Float64(k)]
                        for k in 0:1 for j in 0:1 for i in 0:1]
const n_sub          = 8

# Staggered sign for each site: (-1)^(i+j+k)
const staggered_signs = [(-1)^(i+j+k) for k in 0:1 for j in 0:1 for i in 0:1]

# Magnetic unit cell vectors
const A1 = [2.0, 0.0, 0.0]
const A2 = [0.0, 2.0, 0.0]
const A3 = [0.0, 0.0, 2.0]

unitcell = Lattice(
    [Dof(:cell, 1), Dof(:sub, n_sub)],
    [QN(cell=1, sub=s) for s in 1:n_sub],
    site_coords_3d;
    vectors=[A1, A2, A3]
)

dofs = SystemDofs([Dof(:cell, 1), Dof(:sub, n_sub), Dof(:spin, 2)])

idx = Dict((qn[:cell], qn[:sub], qn[:spin]) => i
           for (i, qn) in enumerate(dofs.valid_states))

# ── Bonds ─────────────────────────────────────────────────────────────────────
nn_bonds     = bonds(unitcell, (:p, :p, :p), 1)
onsite_bonds = bonds(unitcell, (:p, :p, :p), 0)
println("NN bonds: $(length(nn_bonds))  (expected 24)")

onebody = generate_onebody(dofs, nn_bonds,
    (delta, qn1, qn2) -> qn1.spin == qn2.spin ? -t : 0.0)

# ── k-grid: 6×6×6 = 216 k-points ─────────────────────────────────────────────
kpoints     = build_kpoints([A1, A2, A3], (6, 6, 6))
const Nk    = length(kpoints)
const n_elec = n_sub * Nk   # half-filling: 1 electron per site
println("Nk = $Nk,  n_elec = $n_elec  (expected 1728)")

# ── AFM order parameter ───────────────────────────────────────────────────────
function afm_order_parameter(G_k)
    G_loc = dropdims(sum(G_k, dims=3), dims=3) ./ Nk
    m = 0.0
    for s in 1:n_sub
        n_up = real(G_loc[idx[(1,s,1)], idx[(1,s,1)]])
        n_dn = real(G_loc[idx[(1,s,2)], idx[(1,s,2)]])
        m += staggered_signs[s] * (n_up - n_dn) / 2
    end
    return abs(m / n_sub)
end

# ── Main scan ─────────────────────────────────────────────────────────────────
# Each (U, T) point is solved independently with symmetry-breaking restarts.
# field_strength + n_warmup drive the system into the AFM phase when it is
# the true ground state; no warm-start G_init is needed.

results = Dict{Float64, Vector{@NamedTuple{T::Float64, mz::Float64, converged::Bool}}}()

for U in U_vals
    println("\n" * "=" ^ 60)
    println("U/t = $(U)")
    println("=" ^ 60)
    @printf("  %-8s  %-10s  %s\n", "T/t", "mz", "converged")

    twobody = generate_twobody(dofs, onsite_bonds,
        (deltas, qn1, qn2, qn3, qn4) ->
            (qn1.spin, qn2.spin, qn3.spin, qn4.spin) == (1,1,2,2) ? U : 0.0,
        order = (cdag, :i, c, :i, cdag, :i, c, :i))

    # ── AFM initial state: sublattice A fully spin-up, B fully spin-down ────────
    d       = length(dofs.valid_states)
    G_local = zeros(ComplexF64, d, d)
    for s in 1:n_sub
        if staggered_signs[s] == 1   # sublattice A → spin up
            i = idx[(1, s, 1)]
            G_local[i, i] = 1.0
        else                          # sublattice B → spin down
            i = idx[(1, s, 2)]
            G_local[i, i] = 1.0
        end
    end
    G_afm = repeat(reshape(G_local, d, d, 1), 1, 1, Nk)  # same for all k

    rows   = @NamedTuple{T::Float64, mz::Float64, converged::Bool}[]
    prev_G = G_afm   # warm-start initialised to perfect AFM state

    for (i, T) in enumerate(T_scan(U))
        r = solve_hfk(dofs, onebody, twobody, kpoints, n_elec;
            temperature    = T,
            # ── Mode A: warm-start from AFM / previous T ────────────────────────
            # G_init         = prev_G,
            # n_restarts     = 1,
            # field_strength = 0.0,
            # n_warmup       = 0,
            # ── Mode B: independent restarts with symmetry-breaking field ────────
            G_init         = i==1 ? nothing : prev_G,
            n_restarts     = i==1 ? 10 : 1,
            field_strength = i==1 ? 1.5 : 0.0,
            n_warmup       = 20,
            tol            = 1e-10,
            verbose        = false)

        mz     = afm_order_parameter(r.G_k)
        prev_G = r.G_k
        push!(rows, (T=T, mz=mz, converged=r.converged))
        @printf("  %-8.3f  %-10.5f  %s\n", T, mz, r.converged ? "✓" : "!")
    end

    results[U] = rows
end

# ── Save data ─────────────────────────────────────────────────────────────────
open(joinpath(@__DIR__, "res.dat"), "w") do f
    println(f, "# U/t  T/t  mz  converged")
    for U in U_vals, row in results[U]
        println(f, @sprintf("%.1f  %.4f  %.8f  %s",
                            U, row.T, row.mz, row.converged))
    end
end

println("\nDone. Run plot.jl to generate the figure.")
