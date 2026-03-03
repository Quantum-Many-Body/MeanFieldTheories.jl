"""
Graphene band structure benchmark.

Reproduces the two calculations from TightBindingApproximation.jl docs
(https://quantum-many-body.github.io/TightBindingApproximation.jl/dev/examples/Graphene/)
using build_tk from SingleModeApproximation.jl:

  1. 2D graphene: energy bands along Γ-K-M-Γ (Dirac cones at K)
  2. Zigzag cylinder (1 × Ny, open y): edge states in the flat-band window

Run from the benchmark/ directory:
    julia --project benchmark_graphene.jl
"""

using LinearAlgebra
import Pkg
Pkg.activate(@__DIR__)

using SingleModeApproximation
using Plots

# ── Lattice geometry ──────────────────────────────────────────────────────────

const a  = 1.0                       # lattice constant
const a1 = [a, 0.0]
const a2 = [0.5*a, √3/2*a]

# Honeycomb unit cell: 2 sublattices A (index 1) and B (index 2)
# A at origin, B shifted by δ = (0, a/√3)
unitcell = Lattice(
    [Dof(:cell, 1), Dof(:sub, 2, [:A, :B])],
    [QN(cell=1, sub=1), QN(cell=1, sub=2)],
    [[0.0, 0.0], [0.0, a/√3]];
    supercell_vectors=[a1, a2]
)

# ── Reciprocal lattice ────────────────────────────────────────────────────────

A_mat = hcat(a1, a2)          # columns = lattice vectors
B_mat = 2π * inv(A_mat)'      # columns = reciprocal vectors
const b1 = B_mat[:, 1]        # 2π*[1, -1/√3]
const b2 = B_mat[:, 2]        # 2π*[0,  2/√3]

# High-symmetry points (Cartesian)
const Γ = [0.0, 0.0]
const K = (2b1 + b2) / 3      # [4π/3, 0] — Dirac point
const M = (b1 + b2) / 2       # [π, π/√3]

# Linear interpolation between two k-points
kpath(p1, p2, n) = [p1 .+ t .* (p2 .- p1) for t in range(0.0, 1.0; length=n)]

# ─────────────────────────────────────────────────────────────────────────────
# Part 1: 2D graphene — bands along Γ → K → M → Γ
# ─────────────────────────────────────────────────────────────────────────────

println("=" ^ 60)
println("Part 1: 2D graphene band structure")
println("=" ^ 60)

# SystemDofs for ONE unit cell (= the magnetic unit cell here)
dofs_2d = SystemDofs([Dof(:cell, 1), Dof(:sub, 2, [:A, :B])])

# NN bonds with full PBC
nn_bonds = bonds(unitcell, (:p, :p), 1)
println("Number of NN bonds: ", length(nn_bonds))

# t = -1.0 (nearest-neighbor hopping)
onebody = generate_onebody(dofs_2d, nn_bonds, -1.0)

# Build T(k) closure
T_func = build_tk(dofs_2d, onebody.ops, onebody.irvec)

# Verify Hermiticity at K
Tk_at_K = T_func(K)
@assert norm(Tk_at_K - Tk_at_K') < 1e-10 "T(K) not Hermitian!"

# Dirac cone check: eigenvalues at K should be ≈ 0
evals_K = eigvals(Hermitian(Tk_at_K))
println("\nEigenvalues at K (Dirac point, should be ≈ 0): ", round.(evals_K; digits=8))

# Band structure along Γ-K-M-Γ
nk = 100
k_ΓK = kpath(Γ, K, nk)
k_KM = kpath(K, M, nk)
k_MΓ = kpath(M, Γ, nk)
k_path = [k_ΓK; k_KM[2:end]; k_MΓ[2:end]]

bands_2d = [eigvals(Hermitian(T_func(k))) for k in k_path]

# Sanity checks
E_min = minimum(minimum.(bands_2d))
E_max = maximum(maximum.(bands_2d))
println("Band range: [", round(E_min; digits=4), ", ", round(E_max; digits=4), "]")
println("Expected:   [-3.0, 3.0]  (nearest-neighbor graphene with t=1)")

midpoint = (E_min + E_max) / 2
@assert abs(midpoint) < 1e-6 "Particle-hole symmetry broken! midpoint=$midpoint"
println("Particle-hole symmetry: ✓  (midpoint = $(round(midpoint; digits=10)))")

# Arc-length x-axis for correct segment proportions
d_ΓK = norm(K - Γ)
d_KM = norm(M - K)
d_MΓ = norm(Γ - M)
arc = [collect(range(0,        d_ΓK;              length=nk));
       collect(range(d_ΓK,     d_ΓK+d_KM;         length=nk))[2:end];
       collect(range(d_ΓK+d_KM, d_ΓK+d_KM+d_MΓ;  length=nk))[2:end]]

xtick_pos = [0.0, d_ΓK, d_ΓK+d_KM, d_ΓK+d_KM+d_MΓ]
xtick_lab = ["Γ", "K", "M", "Γ"]

bands_2d_mat = hcat(bands_2d...)   # (nbands × nk)

p1 = plot(; title="Graphene band structure",
            xlabel="k-path", ylabel="E / t",
            xticks=(xtick_pos, xtick_lab),
            xlims=(0, d_ΓK+d_KM+d_MΓ), ylims=(-3.5, 3.5),
            legend=false, framestyle=:box)
hline!(p1, [0.0]; color=:gray, linestyle=:dash, linewidth=0.8)
for i in xtick_pos
    vline!(p1, [i]; color=:gray, linestyle=:dot, linewidth=0.8)
end
for n in axes(bands_2d_mat, 1)
    plot!(p1, arc, bands_2d_mat[n, :]; color=:steelblue, linewidth=1.5)
end
display(p1)
savefig(p1, joinpath(@__DIR__, "graphene_bands_2d.png"))
println("Saved: benchmark/graphene_bands_2d.png")

# ─────────────────────────────────────────────────────────────────────────────
# Part 2: Cylinder with zigzag edges (1 × Ny, open in y)
# ─────────────────────────────────────────────────────────────────────────────

println("\n" * "=" ^ 60)
println("Part 2: Zigzag cylinder band structure")
println("=" ^ 60)

Ny = 50   # number of unit cells along y (open direction)

# Tile: 1 cell in x (periodic), Ny cells in y (open)
cylinder = Lattice(unitcell, [a1, a2], (1, Ny))
println("Cylinder: 1 × $Ny unit cells, $(length(cylinder.position_states)) sites")

# SystemDofs for the full column (= the 1D magnetic unit cell)
dofs_cyl = SystemDofs([Dof(:cell, Ny), Dof(:sub, 2, [:A, :B])])
println("d_int = ", length(dofs_cyl.valid_states), "  (= 2×Ny = $(2Ny))")

# NN bonds: periodic in x, open in y
nn_bonds_cyl = bonds(cylinder, (:p, :o), 1)
println("Number of NN bonds (cylinder): ", length(nn_bonds_cyl))

onebody_cyl = generate_onebody(dofs_cyl, nn_bonds_cyl, -1.0)
T_cyl = build_tk(dofs_cyl, onebody_cyl.ops, onebody_cyl.irvec)

# 1D BZ for cylinder: kx ∈ [-π/a, π/a] = [-π, π]
nkx = 200
kx_vals = range(-π/a, π/a; length=nkx)
k_cyl   = [[kx, 0.0] for kx in kx_vals]

bands_cyl = [eigvals(Hermitian(T_cyl(k))) for k in k_cyl]

# Edge state check: near kx=0, there should be two near-zero eigenvalues
k_mid_idx = nkx ÷ 2
evals_mid = bands_cyl[k_mid_idx]
flat_band_evals = evals_mid[Ny:Ny+1]  # the two mid-gap states
println("\nAt kx=0, two mid-gap edge states (should be ≈ 0):")
println("  E = ", round.(flat_band_evals; digits=6))
@assert all(abs.(flat_band_evals) .< 0.1) "Edge states not near zero at kx=0!"
println("Edge state check: ✓")

bands_cyl_mat = hcat(bands_cyl...)   # (2Ny × nkx)
kx_arr = collect(kx_vals)

p2 = plot(; title="Graphene zigzag cylinder (Ny=$Ny)",
            xlabel="kx", ylabel="E / t",
            xticks=([-π, -2π/3, 0, 2π/3, π], ["-π", "-2π/3", "0", "2π/3", "π"]),
            xlims=(-π, π), ylims=(-3.5, 3.5),
            legend=false, framestyle=:box)
hline!(p2, [0.0]; color=:gray, linestyle=:dash, linewidth=0.8)
vline!(p2, [-2π/3, 2π/3]; color=:gray, linestyle=:dot, linewidth=0.8)
for n in axes(bands_cyl_mat, 1)
    plot!(p2, kx_arr, bands_cyl_mat[n, :]; color=:steelblue, linewidth=0.6, alpha=0.7)
end
display(p2)
savefig(p2, joinpath(@__DIR__, "graphene_bands_cylinder.png"))
println("Saved: benchmark/graphene_bands_cylinder.png")

println("\nAll checks passed ✓")
