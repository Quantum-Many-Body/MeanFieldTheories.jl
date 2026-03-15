"""
Hubbard model on the honeycomb lattice (graphene): AFM transition at half-filling.

Model:
  H = -t Σ_{<ij>,σ} (c†_{iσ}c_{jσ} + h.c.) + U Σ_i n_{i↑}n_{i↓}

Honeycomb unit cell: 2 sublattices (A,B). Mean-field predicts an AFM transition
at U/t ≈ 2.2(3), where opposite sublattice moments form and a gap opens.

This example:
  1) U=0 graphene bands along Γ–K–M–Γ.
  2) U=0 zigzag cylinder bands (edge states).
  3) Interacting Hubbard model: AFM transition and gap opening,
     found automatically via symmetry-breaking warmup restarts — no
     prior knowledge of the ordered phase required.

Run:
    julia --project=examples -t 8 examples/SM_AFM/run.jl
"""

using Printf
using LinearAlgebra
using MeanFieldTheories

# ── Parameters ───────────────────────────────────────────────────────────────
const t  = 1.0
const Uc = 2.2

U_sweep = collect(range(0.0, 4.0, length=41))
U_bands = [0.0, 1.0, 2.2, 2.3, 3.0, 4.0]

# ── Lattice geometry ──────────────────────────────────────────────────────────
const a  = 1.0
const a1 = [a, 0.0]
const a2 = [0.5*a, √3/2*a]

# Honeycomb unit cell: A at origin, B shifted by δ = (0, a/√3)
unitcell = Lattice(
    [Dof(:cell, 1), Dof(:sub, 2, [:A, :B])],
    [QN(cell=1, sub=1), QN(cell=1, sub=2)],
    [[0.0, 0.0], [0.0, a/√3]];
    vectors=[a1, a2]
)

# System DOFs: 1 cell × 2 sublattices × 2 spins → d = 4
dofs = SystemDofs([Dof(:cell, 1), Dof(:sub, 2, [:A, :B]), Dof(:spin, 2, [:up, :dn])])

nn_bonds     = bonds(unitcell, (:p, :p), 1)
onsite_bonds = bonds(unitcell, (:p, :p), 0)

# One-body hopping (spin-conserving)
onebody_hop = generate_onebody(dofs, nn_bonds,
    (delta, qn1, qn2) -> qn1.spin == qn2.spin ? -t : 0.0)

# k-grid for SCF
kpoints = build_kpoints([a1, a2], (100, 100))
Nk      = length(kpoints)

# Half-filling: 2 electrons per unit cell ⇒ 2*Nk total
n_elec = 2 * Nk

# ── DOF index map ─────────────────────────────────────────────────────────────
idx = Dict((qn[:cell], qn[:sub], qn[:spin]) => i
           for (i, qn) in enumerate(dofs.valid_states))

# ── AFM order parameter ───────────────────────────────────────────────────────
function afm_order_parameter(G_k)
    Nk_local = size(G_k, 3)
    G_loc    = dropdims(sum(G_k, dims=3), dims=3) ./ Nk_local
    nA_up = real(G_loc[idx[(1,1,1)], idx[(1,1,1)]])
    nA_dn = real(G_loc[idx[(1,1,2)], idx[(1,1,2)]])
    nB_up = real(G_loc[idx[(1,2,1)], idx[(1,2,1)]])
    nB_dn = real(G_loc[idx[(1,2,2)], idx[(1,2,2)]])
    sA = (nA_up - nA_dn) / 2
    sB = (nB_up - nB_dn) / 2
    return abs((sA - sB) / 2)
end

# ── Two-body Hubbard interaction ──────────────────────────────────────────────
function build_U_ops(U)
    return generate_twobody(dofs, onsite_bonds,
        (deltas, qn1, qn2, qn3, qn4) ->
            (qn1.spin, qn2.spin, qn3.spin, qn4.spin) == (1,1,2,2) ? U : 0.0,
        order = (cdag, :i, c, :i, cdag, :i, c, :i))
end

# ── k-path for band structure ─────────────────────────────────────────────────
A_mat = hcat(a1, a2)
B_mat = 2π * inv(A_mat)'
const b1 = B_mat[:, 1]
const b2 = B_mat[:, 2]

const Γ = [0.0, 0.0]
const K = (2b1 + b2) / 3
const M = (b1 + b2) / 2

kpath(p1, p2, n) = [p1 .+ s .* (p2 .- p1) for s in range(0.0, 1.0; length=n)]

nk = 120
k_ΓK = kpath(Γ, K, nk)
k_KM = kpath(K, M, nk)
k_MΓ = kpath(M, Γ, nk)
k_path = [k_ΓK; k_KM[2:end]; k_MΓ[2:end]]

d_ΓK = norm(K - Γ)
d_KM = norm(M - K)
d_MΓ = norm(Γ - M)
arc = [collect(range(0,         d_ΓK;             length=nk));
       collect(range(d_ΓK,      d_ΓK+d_KM;        length=nk))[2:end];
       collect(range(d_ΓK+d_KM, d_ΓK+d_KM+d_MΓ;  length=nk))[2:end]]
xtick_pos = [0.0, d_ΓK, d_ΓK+d_KM, d_ΓK+d_KM+d_MΓ]
xtick_lab = ["Γ", "K", "M", "Γ"]

# ── Part 1: U=0 graphene band structure (2D) ─────────────────────────────────
println("=" ^ 60)
println("Part 1: U=0 graphene band structure (2D)")
println("=" ^ 60)

dofs_2d     = SystemDofs([Dof(:cell, 1), Dof(:sub, 2, [:A, :B])])
nn_bonds_2d = bonds(unitcell, (:p, :p), 1)
onebody_2d  = generate_onebody(dofs_2d, nn_bonds_2d, -t)
T_func_2d   = build_Tk(build_Tr(dofs_2d, onebody_2d.ops, onebody_2d.irvec))

bands_2d_mat = hcat([eigvals(Hermitian(T_func_2d(k))) for k in k_path]...)

open(joinpath(@__DIR__, "bands_2d.dat"), "w") do f
    println(f, "# arc  E1  E2")
    for j in axes(bands_2d_mat, 2)
        println(f, @sprintf("%.6f  %.8f  %.8f",
                            arc[j], bands_2d_mat[1,j], bands_2d_mat[2,j]))
    end
end
println("Saved: examples/SM_AFM/bands_2d.dat")

# ── Part 2: Zigzag cylinder band structure (U=0) ─────────────────────────────
println("\n" * "=" ^ 60)
println("Part 2: Zigzag cylinder band structure (U=0)")
println("=" ^ 60)

Ny        = 50
cylinder  = Lattice(unitcell, (1, Ny))
dofs_cyl  = SystemDofs([Dof(:cell, Ny), Dof(:sub, 2, [:A, :B])])
onebody_cyl = generate_onebody(dofs_cyl, bonds(cylinder, (:p, :o), 1), -t)
T_func_cyl  = build_Tk(build_Tr(dofs_cyl, onebody_cyl.ops, onebody_cyl.irvec))

nkx     = 200
kx_vals = collect(range(-π/a, π/a; length=nkx))
bands_cyl_mat = hcat([eigvals(Hermitian(T_func_cyl([kx, 0.0]))) for kx in kx_vals]...)

cyl_xtick_pos = [-π, -2π/3, 0.0, 2π/3, π]
cyl_xtick_lab = ["-π", "-2π/3", "0", "2π/3", "π"]

nbands_cyl = size(bands_cyl_mat, 1)
open(joinpath(@__DIR__, "bands_cyl.dat"), "w") do f
    println(f, "# kx  E1..E$(nbands_cyl)  (Ny=$Ny)")
    for j in eachindex(kx_vals)
        vals = join([@sprintf("%.8f", bands_cyl_mat[n,j]) for n in 1:nbands_cyl], "  ")
        println(f, @sprintf("%.6f  %s", kx_vals[j], vals))
    end
end
println("Saved: examples/SM_AFM/bands_cyl.dat")

# ── Part 3: U sweep — ground state via symmetry-breaking restarts ─────────────
println("\n" * "=" ^ 60)
println("Part 3: Hubbard model — U sweep (symmetry-breaking restarts)")
println("=" ^ 60)
println(@sprintf("# k-grid: 100×100, Nk=%d,  expected Uc ≈ %.2f", Nk, Uc))
println()
println(@sprintf("# %-6s  %-14s  %-10s  %s", "U", "E_gs", "m_AF", "phase"))

results = Dict{Float64, Any}()

for U in unique(sort([U_sweep; U_bands]))
    U_ops   = build_U_ops(U)
    twobody = (ops=U_ops.ops, delta=U_ops.delta, irvec=U_ops.irvec)

    r = solve_hfk(dofs, onebody_hop, twobody, kpoints, n_elec;
        n_restarts     = 5,
        field_strength = 1.0,
        n_warmup       = 15,
        tol            = 1e-12,
        verbose        = false)

    m_afm = afm_order_parameter(r.G_k)
    phase = m_afm > 0.01 ? "AFM" : "PM"
    results[U] = (r_gs=r, m_afm=m_afm, phase=phase)

    println(@sprintf("  %-6.3f  %+14.8f  %-10.6f  %s",
                     U, r.energies.total, m_afm, phase))
end

open(joinpath(@__DIR__, "res.dat"), "w") do f
    println(f, "# U  m_AF  E_gs  phase")
    for U in U_sweep
        r = results[U]
        println(f, @sprintf("%.4f  %.8f  %+.10f  %s",
                            U, r.m_afm, r.r_gs.energies.total, r.phase))
    end
end

# ── Save AFM band structures for selected U ───────────────────────────────────
open(joinpath(@__DIR__, "afm_bands.dat"), "w") do f
    println(f, "# U  arc  E1  E2  E3  E4")
    for U in U_bands
        U_ops   = build_U_ops(U)
        twobody = (ops=U_ops.ops, delta=U_ops.delta, irvec=U_ops.irvec)
        bands_mat, _ = energy_bands(dofs, onebody_hop, twobody, kpoints,
                                    results[U].r_gs.G_k, k_path)
        for j in axes(bands_mat, 2)
            println(f, @sprintf("%.4f  %.6f  %.8f  %.8f  %.8f  %.8f",
                                U, arc[j],
                                bands_mat[1,j], bands_mat[2,j],
                                bands_mat[3,j], bands_mat[4,j]))
        end
    end
end

println("\nDone. Run plot.jl to generate the figures.")
