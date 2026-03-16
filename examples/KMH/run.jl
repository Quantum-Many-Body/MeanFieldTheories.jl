"""
Kane-Mele-Hubbard (KMH) model – SDW phase boundary (Fig. 6, Rachel & Le Hur)

  H = -t  Σ_{<ij>,σ}    (c†_{iσ}c_{jσ} + h.c.)
    + iλ  Σ_{<<ij>>,σσ'} νᵢⱼ σᶻ_{σσ'} c†_{iσ}c_{jσ'} + h.c.
    + U   Σ_i n_{i↑}n_{i↓}

NNN term with ϕ = π/2:  iλ νᵢⱼ σᶻ  ≡  λ exp(im · σ · νᵢⱼ · π/2)

State ordering (spin slow, sub fast):
  1 = (A,↑),  2 = (B,↑),  3 = (A,↓),  4 = (B,↓)

Scan: λ ∈ {0.2, 0.4, 0.6, 0.8, 1.0}, 20 U-points each.
      λ = 0 is read from examples/SM_AFM/res.dat (pure Hubbard).

Run:
    julia --project=examples -t 8 examples/KMH/run.jl
"""

using MeanFieldTheories
using LinearAlgebra
using Printf

# ── Lattice geometry ──────────────────────────────────────────────────────────
const a1 = [0.0, √3]
const a2 = [3/2, √3/2]

unitcell = Lattice(
    [Dof(:cell, 1), Dof(:sub, 2, [:A, :B])],
    [QN(cell=1, sub=1), QN(cell=1, sub=2)],
    [[0.0, 0.0], [1.0, 0.0]];
    vectors=[a1, a2]
)

dofs = SystemDofs([Dof(:cell, 1), Dof(:sub, 2, [:A, :B]), Dof(:spin, 2, [:up, :dn])])

nn_bonds     = bonds(unitcell, (:p, :p), 1)
nnn_bonds    = bonds(unitcell, (:p, :p), 2)
onsite_bonds = bonds(unitcell, (:p, :p), 0)

# ── k-grid ────────────────────────────────────────────────────────────────────
const Nk1, Nk2 = 100, 100
const kpoints   = build_kpoints([a1, a2], (Nk1, Nk2))
const Nk        = length(kpoints)
const n_elec    = 2 * Nk   # half-filling: 2 electrons per unit cell

# ── DOF index map ─────────────────────────────────────────────────────────────
const idx = Dict((qn[:cell], qn[:sub], qn[:spin]) => i
                 for (i, qn) in enumerate(dofs.valid_states))

# ── One-body operator for given λ ─────────────────────────────────────────────
function build_onebody(λ)
    onebody_nn = generate_onebody(dofs, nn_bonds,
        (delta, qn1, qn2) -> qn1.spin == qn2.spin ? -1.0 : 0.0)

    onebody_nnn = generate_onebody(dofs, nnn_bonds,
        (delta, qn1, qn2) -> begin
            qn1.spin !== qn2.spin && return 0.0im
            sigma = (qn1.spin == 1) ? 1.0 : -1.0
            if qn1.sub == qn2.sub == 1
                nu = any(x -> isapprox(delta, x; atol=1e-6),
                         [[3/2, √3/2], [-3/2, √3/2], [0.0, -√3]]) ?  1.0 :
                     any(x -> isapprox(delta, x; atol=1e-6),
                         [[-3/2, -√3/2], [3/2, -√3/2], [0.0, √3]]) ? -1.0 :
                     throw(ArgumentError("Unexpected NNN bond on sub=1: delta=$delta"))
            elseif qn1.sub == qn2.sub == 2
                nu = any(x -> isapprox(delta, x; atol=1e-6),
                         [[3/2, -√3/2], [0.0, √3], [-3/2, -√3/2]]) ?  1.0 :
                     any(x -> isapprox(delta, x; atol=1e-6),
                         [[-3/2, √3/2], [0.0, -√3], [3/2, √3/2]]) ? -1.0 :
                     throw(ArgumentError("Unexpected NNN bond on sub=2: delta=$delta"))
            else
                return 0.0im
            end
            return λ * exp(im * sigma * nu * π/2)
        end)

    return (ops   = [onebody_nn.ops;   onebody_nnn.ops],
            delta = [onebody_nn.delta; onebody_nnn.delta],
            irvec = [onebody_nn.irvec; onebody_nnn.irvec])
end


# ── Reciprocal lattice vectors ────────────────────────────────────────────────
const A_mat = hcat(a1, a2)
const B_mat = 2π * inv(A_mat)'
const b1 = B_mat[:, 1]   # [-2π/3, 2π/√3]
const b2 = B_mat[:, 2]   # [4π/3,  0]

# High-symmetry points (Fig. 5 path: K→Γ→K'→K→M→Γ)
const Γ  = [0.0, 0.0]
const K  = (2b1 + b2) / 3
const Kp = (b1 + 2b2) / 3   # K'
const M  = (b1 + b2) / 2

kpath_seg(p1, p2, n) = [p1 .+ t .* (p2 .- p1) for t in range(0.0, 1.0; length=n)]

const nk_path = 120
const k_path  = [kpath_seg(K,  Γ,  nk_path);
                 kpath_seg(Γ,  Kp, nk_path)[2:end];
                 kpath_seg(Kp, K,  nk_path)[2:end];
                 kpath_seg(K,  M,  nk_path)[2:end];
                 kpath_seg(M,  Γ,  nk_path)[2:end]]

const arc = let
    segs = [(K,Γ), (Γ,Kp), (Kp,K), (K,M), (M,Γ)]
    a = Float64[]
    pos = 0.0
    for (i, (p1, p2)) in enumerate(segs)
        n   = nk_path
        seg = collect(range(pos, pos + norm(p2-p1); length=n))
        append!(a, i == 1 ? seg : seg[2:end])
        pos += norm(p2 - p1)
    end
    a
end

const xtick_pos = cumsum([0.0, norm(Γ-K), norm(Kp-Γ), norm(K-Kp), norm(M-K), norm(Γ-M)])
const xtick_lab = ["K", "Γ", "K'", "K", "M", "Γ"]

# ── Band structure for λ = 0.05, 0.2, 0.5, 1.0  (Fig. 5) ────────────────────
const λ_bands = [0.05, 0.2, 0.5, 1.0]

open(joinpath(@__DIR__, "bands.dat"), "w") do f
    nbands = 2 * length(dofs.valid_states) ÷ 2   # = 4
    println(f, "# lambda  arc  E1..E$nbands")
    for λ in λ_bands
        onebody = build_onebody(λ)
        T_func  = build_Tk(build_Tr(dofs, onebody.ops, onebody.irvec))
        bands   = hcat([eigvals(Hermitian(T_func(k))) for k in k_path]...)
        for j in eachindex(arc)
            vals = join([@sprintf("%.8f", bands[n, j]) for n in axes(bands, 1)], "  ")
            println(f, @sprintf("%.4f  %.8f  %s", λ, arc[j], vals))
        end
    end
end
println("Saved: examples/KMH/bands.dat  (λ = $(λ_bands))")
println("Run plot.jl to generate the figure.")


# ── Two-body Hubbard U ────────────────────────────────────────────────────────
function build_U_ops(U)
    generate_twobody(dofs, onsite_bonds,
        (deltas, qn1, qn2, qn3, qn4) ->
            (qn1.spin == qn2.spin)&&(qn3.spin==qn4.spin)&&(qn1.spin !== qn3.spin) ? U/2 : 0.0,
        order=(cdag, :i, c, :i, cdag, :i, c, :i))
end

# ── XY-plane Néel order parameter ────────────────────────────────────────────
# m_Neel_xy = ||(mA - mB)_xy|| = sqrt((Sx_A-Sx_B)² + (Sy_A-Sy_B)²)
function sdw_order_parameter(G_k)
    G_loc = dropdims(sum(G_k, dims=3), dims=3) ./ Nk
    iAup, iAdn = idx[(1,1,1)], idx[(1,1,2)]
    iBup, iBdn = idx[(1,2,1)], idx[(1,2,2)]
    Sx_A = real(G_loc[iAup, iAdn]);  Sy_A = imag(G_loc[iAup, iAdn])
    Sx_B = real(G_loc[iBup, iBdn]);  Sy_B = imag(G_loc[iBup, iBdn])
    return sqrt((Sx_A - Sx_B)^2 + (Sy_A - Sy_B)^2)
end

# ── Scan parameters ───────────────────────────────────────────────────────────
const λ_vals   = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
const U_ranges = Dict(
    0.0 => range(0.0, 4.0,  length=20),
    0.2 => range(0.0, 5.0,  length=20),
    0.4 => range(0.0, 7.0,  length=20),
    0.6 => range(0.0, 8.0,  length=20),
    0.8 => range(0.0, 9.0,  length=20),
    1.0 => range(0.0, 10.0, length=20),
)

# ── Main scan ─────────────────────────────────────────────────────────────────
println(@sprintf("%-6s  %-8s  %-10s  %-8s  %-8s  %-8s  %-8s  %-8s  %-8s  %s",
                 "λ", "U", "m_neel_xy",
                 "mx_A", "my_A", "mz_A", "mx_B", "my_B", "mz_B", "conv"))
println("-" ^ 90)

open(joinpath(@__DIR__, "res.dat"), "w") do f
    println(f, "# lambda  U  m_neel_xy  mx_A  my_A  mz_A  mx_B  my_B  mz_B  converged")

    for λ in λ_vals
        onebody = build_onebody(λ)
        prev_G  = nothing
        for (i, U) in enumerate(reverse(collect(U_ranges[λ])))   # high → low
            twobody = build_U_ops(U)
            r = solve_hfk(dofs, onebody, twobody, kpoints, n_elec;
                G_init         = prev_G,
                n_restarts     = i==1 ? 5 : 1,
                field_strength = i==1 ? 1.0 : 0.0,
                n_warmup       = i==1 ? 15 : 0,
                tol            = 1e-10,
                verbose        = false)
            prev_G = r.G_k

            mxy  = sdw_order_parameter(r.G_k)
            mags = local_magnetization(dofs, r.G_k)
            sA, sB = mags[1], mags[2]

            println(@sprintf("%-6.2f  %-8.4f  %-10.6f  %-8.4f  %-8.4f  %-8.4f  %-8.4f  %-8.4f  %-8.4f  %s",
                             λ, U, mxy,
                             sA.mx, sA.my, sA.mz,
                             sB.mx, sB.my, sB.mz,
                             r.converged ? "✓" : "!"))
            println(f, @sprintf("%.4f  %.6f  %.8f  %.6f  %.6f  %.6f  %.6f  %.6f  %.6f  %s",
                                λ, U, mxy,
                                sA.mx, sA.my, sA.mz,
                                sB.mx, sB.my, sB.mz,
                                r.converged))
        end
    end
end

println("\nDone. Results saved to res.dat. Run plot.jl to generate the figure.")

# ── Part 3: Ordering wavevector Q for each ordered (λ, U) ─────────────────────
println()
println("=" ^ 60)
println("Part 3: Magnetic ordering wavevector Q")
println("=" ^ 60)

qpoints = build_kpoints([a1, a2], (200, 200))

rows = NamedTuple[]
open(joinpath(@__DIR__, "res.dat")) do f
    for line in eachline(f)
        startswith(line, "#") && continue
        isempty(strip(line)) && continue
        v = split(strip(line))
        push!(rows, (
            lambda    = parse(Float64, v[1]),
            U         = parse(Float64, v[2]),
            m_neel_xy = parse(Float64, v[3]),
            mx_A = parse(Float64, v[4]), my_A = parse(Float64, v[5]), mz_A = parse(Float64, v[6]),
            mx_B = parse(Float64, v[7]), my_B = parse(Float64, v[8]), mz_B = parse(Float64, v[9]),
        ))
    end
end

println(@sprintf("# %-6s  %-8s  %-10s  %-14s  %-14s  %s",
                 "lambda", "U", "m_neel_xy", "Qx", "Qy", "S(Q)"))

open(joinpath(@__DIR__, "ordering_q.dat"), "w") do f
    println(f, "# lambda  U  m_neel_xy  Qx  Qy  Sq_max")
    for r in rows
        r.m_neel_xy < 0.01 && continue
        mags = [
            (label=(cell=1, sub=1), mx=r.mx_A, my=r.my_A, mz=r.mz_A),
            (label=(cell=1, sub=2), mx=r.mx_B, my=r.my_B, mz=r.mz_B),
        ]
        res = ordering_wavevector(mags, unitcell, qpoints)
        Q   = res.Q
        println(@sprintf("  %-6.2f  %-8.4f  %-10.6f  %-+14.6f  %-+14.6f  %.6f",
                         r.lambda, r.U, r.m_neel_xy, Q[1], Q[2], res.Sq_max))
        println(f, @sprintf("%.4f  %.6f  %.8f  %.8f  %.8f  %.8f",
                            r.lambda, r.U, r.m_neel_xy, Q[1], Q[2], res.Sq_max))
    end
end

println("\nOrdering wavevectors saved to ordering_q.dat.")
