module MakieExt

using MeanFieldTheories
import Makie

# ──────────────── manual arrow (shaft + rotated triangle head) ────────────────
# Draws ONE arrow centered at `center`, pointing in direction `vec` (data coords).
# The total length of the arrow equals norm(vec).
function _arrow!(ax, center, vec, color; shaft_lw=2.0, head_px=12, head_frac=0.35)
    len = sqrt(vec[1]^2 + vec[2]^2)
    len < 1e-8 && return

    tail = Makie.Point2f(center[1] - vec[1]/2, center[2] - vec[2]/2)
    tip  = Makie.Point2f(center[1] + vec[1]/2, center[2] + vec[2]/2)

    # Draw shaft all the way to tip; the triangle head will cover the end,
    # eliminating any visible gap between shaft and head.
    Makie.linesegments!(ax, [tail, tip]; color, linewidth=shaft_lw)

    # Head: utriangle centroid sits 1/3 of head length behind tip so the
    # apex lands approximately at tip (2/3 of head length past the centroid).
    head_center = Makie.Point2f(tip[1] - vec[1]*head_frac/3,
                                 tip[2] - vec[2]*head_frac/3)
    θ = atan(vec[2], vec[1]) - π/2f0
    Makie.scatter!(ax, [head_center]; marker=:utriangle, markersize=head_px,
                   color, rotation=θ, strokewidth=0)
end

# ──────────────── compute minimum inter-site distance ────────────────────────
function _min_dist(positions)
    n = length(positions)
    n < 2 && return 1.0
    dmin = Inf
    for i in 1:n, j in (i+1):n
        d = sqrt(sum((positions[i] .- positions[j]).^2))
        d < dmin && (dmin = d)
    end
    return dmin
end

# ──────────────── single top-view panel ───────────────────────────────────────
# Encodes full 3D spin vector in one 2D panel:
#   • Arrow (centered on site):  direction = (mx, my),
#                                 length    = L × m_xy / m_max
#                                 (L = min_dist × arrow_frac)
#   • ⊙ dot   (mz > threshold):  size ∝ mz / m_max
#   • ⊗ cross (mz < -threshold): size ∝ |mz| / m_max
#
# When all mz ≈ 0 → only arrows.
# When all m_xy ≈ 0 → only ⊙/⊗.
# Mixed → both simultaneously.
function _draw_topview!(ax, mags, positions;
                        arrow_frac,          # L = min_dist * arrow_frac
                        arrow_color,
                        shaft_lw, head_px, head_frac,
                        site_markersize,
                        dot_size_frac,       # inner marker max size = site_markersize * dot_size_frac
                        bonds, bond_color, bond_width,
                        unitcell_vecs, unitcell_origin, padding)

    pts2 = [Makie.Point2f(p[1], p[2]) for p in positions]

    # bonds and unit cell
    _draw_bonds_and_cell!(ax, bonds, bond_color, bond_width,
                          unitcell_vecs, unitcell_origin, positions, (1, 2))
    _set_limits!(ax, positions, (1, 2), padding)

    # outer circle for all sites
    Makie.scatter!(ax, pts2; marker=:circle, markersize=site_markersize,
                   color=:white, strokewidth=1.5, strokecolor=:black)

    # normalisation
    m_max  = maximum(sqrt(m.mx^2 + m.my^2 + m.mz^2) for m in mags)
    m_max  = max(m_max, 1e-8)
    L      = _min_dist(positions) * arrow_frac   # max arrow half-length

    dot_max = site_markersize * dot_size_frac     # max inner marker size

    for (i, mag) in enumerate(mags)
        c    = pts2[i]
        mxy  = sqrt(mag.mx^2 + mag.my^2)
        absz = abs(mag.mz)

        # ── in-plane arrow ──────────────────────────────────────────────────
        if mxy > 1e-6 * m_max
            scale = L * mxy / m_max
            dx = Float32(mag.mx / mxy * scale)
            dy = Float32(mag.my / mxy * scale)
            _arrow!(ax, c, Makie.Vec2f(dx, dy), arrow_color;
                    shaft_lw, head_px, head_frac)
        end

        # ── z-component marker ──────────────────────────────────────────────
        if absz > 1e-6 * m_max
            # Scale linearly with |mz|/m_max, but keep a 40% floor so the
            # marker is always clearly visible even for small z components.
            inner_sz = dot_max * (0.4 + 0.6 * absz / m_max)
            if mag.mz > 0
                # ⊙ — filled dot
                Makie.scatter!(ax, [c]; marker=:circle,
                               markersize=inner_sz, color=:black, strokewidth=0)
            else
                # ⊗ — cross (slightly larger so arms are as prominent as dot)
                Makie.scatter!(ax, [c]; marker=:xcross,
                               markersize=inner_sz * 1.4,
                               color=:black, strokewidth=0)
            end
        end
    end
end

# ──────────────── bond lines and unit-cell outline ────────────────────────────
function _draw_bonds_and_cell!(ax, bonds, bond_color, bond_width,
                                unitcell_vecs, unitcell_origin, positions, proj_pos)
    get_coord(p, dim) = dim <= length(p) ? Float32(p[dim]) : 0f0
    ph, pv = proj_pos

    if bonds !== nothing
        bpts = Makie.Point2f[]
        for b in bonds
            length(b.coordinates) < 2 && continue
            push!(bpts, Makie.Point2f(b.coordinates[1][1], b.coordinates[1][2]))
            push!(bpts, Makie.Point2f(b.coordinates[2][1], b.coordinates[2][2]))
        end
        Makie.linesegments!(ax, bpts; color=bond_color, linewidth=bond_width)
    end

    if unitcell_vecs !== nothing
        orig = unitcell_origin !== nothing ? unitcell_origin : positions[1]
        a = Makie.Vec2f(get_coord(unitcell_vecs[1],ph), get_coord(unitcell_vecs[1],pv))
        b = Makie.Vec2f(get_coord(unitcell_vecs[2],ph), get_coord(unitcell_vecs[2],pv))
        o = Makie.Point2f(get_coord(orig,ph), get_coord(orig,pv))
        Makie.lines!(ax, [o, o+a, o+a+b, o+b, o];
                     color=:black, linestyle=:dash, linewidth=1.0)
    end
end

# ──────────────── axis limits tight around sites ─────────────────────────────
function _set_limits!(ax, positions, proj_pos, padding)
    ph, pv = proj_pos
    get_coord(p, dim) = dim <= length(p) ? Float64(p[dim]) : 0.0
    xs = [get_coord(p, ph) for p in positions]
    ys = [get_coord(p, pv) for p in positions]
    Makie.xlims!(ax, minimum(xs) - padding, maximum(xs) + padding)
    Makie.ylims!(ax, minimum(ys) - padding, maximum(ys) + padding)
end

# ──────────────── main function ───────────────────────────────────────────────

function MeanFieldTheories.plot_magnetization(
    mags::Vector,
    positions::AbstractVector;
    arrow_color              = :crimson,
    arrow_frac::Real         = 0.33,    # arrow max length = min_dist * arrow_frac
    shaft_lw::Real           = 2.0,
    head_px::Real            = 13,
    head_frac::Real          = 0.32,
    site_markersize::Real    = 20,
    dot_size_frac::Real      = 0.45,    # inner ⊙/⊗ max size relative to site_markersize
    bonds                    = nothing,
    bond_color               = :gray60,
    bond_width::Real         = 1.5,
    unitcell_vecs            = nothing,
    unitcell_origin          = nothing,
    axis_padding::Real       = 0.5,
    title::String            = "",
)
    length(mags) == length(positions) ||
        error("length(mags)=$(length(mags)) ≠ length(positions)=$(length(positions))")
    isempty(mags) && error("mags is empty")

    fig = Makie.Figure(size=(440, 420))

    axis_kw = (; xgridvisible=false, ygridvisible=false,
                 aspect=Makie.DataAspect())

    ax = Makie.Axis(fig[1, 1]; title, xlabel="x", ylabel="y", axis_kw...)

    _draw_topview!(ax, mags, positions;
                   arrow_frac, arrow_color,
                   shaft_lw, head_px=Float64(head_px), head_frac,
                   site_markersize=Float64(site_markersize),
                   dot_size_frac,
                   bonds, bond_color, bond_width,
                   unitcell_vecs, unitcell_origin,
                   padding=Float64(axis_padding))

    return fig
end

# ──────────────── plot_lattice ────────────────────────────────────────────────

# Internal helpers for scalar-or-vector style indexing
_cyc(v::AbstractVector, i) = v[mod1(i, length(v))]
_cyc(v, _)                 = v   # scalar → same for all

function MeanFieldTheories.plot_lattice(
    lattice::MeanFieldTheories.Lattice;
    bond_lists      = nothing,
    bond_colors     = [:tomato, :steelblue, :darkorange, :mediumpurple, :teal],
    bond_labels     = String[],
    bond_widths     = 1.5,
    bond_styles     = :solid,
    site_groupby    = nothing,
    site_colors     = [:seagreen, :mediumpurple, :coral, :royalblue, :goldenrod],
    site_labels     = nothing,
    site_markersize::Real = 10,
    title::String         = "",
)
    # ── normalise bond_lists to Vector{Vector{Bond}} ──────────────────────────
    blists = if isnothing(bond_lists)
        []
    elseif bond_lists isa AbstractVector && !isempty(bond_lists) &&
           !(first(bond_lists) isa AbstractVector)
        [bond_lists]   # single Vector{Bond} → wrap
    else
        collect(bond_lists)
    end

    fig = Makie.Figure(size = (600, 560))
    ax  = Makie.Axis(fig[1, 1];
                     title,
                     xlabel         = "x",
                     ylabel         = "y",
                     aspect         = Makie.DataAspect(),
                     xgridvisible   = false,
                     ygridvisible   = false)

    # ── bonds ─────────────────────────────────────────────────────────────────
    for (k, blist) in enumerate(blists)
        pts = Makie.Point2f[]
        for b in blist
            length(b.coordinates) < 2 && continue
            push!(pts, Makie.Point2f(b.coordinates[1][1], b.coordinates[1][2]))
            push!(pts, Makie.Point2f(b.coordinates[2][1], b.coordinates[2][2]))
        end
        isempty(pts) && continue
        lbl = k <= length(bond_labels) ? bond_labels[k] : ""
        Makie.linesegments!(ax, pts;
            color     = _cyc(bond_colors, k),
            linewidth = _cyc(bond_widths, k),
            linestyle = _cyc(bond_styles, k),
            label     = lbl)
    end

    # ── sites ─────────────────────────────────────────────────────────────────
    coords = lattice.coordinates
    states = lattice.position_states

    if isnothing(site_groupby)
        # All sites in one group
        pts = [Makie.Point2f(c[1], c[2]) for c in coords]
        lbl = isnothing(site_labels) ? "" :
              (site_labels isa AbstractVector ? first(site_labels) : string(site_labels))
        Makie.scatter!(ax, pts;
            color      = _cyc(site_colors, 1),
            markersize = site_markersize,
            label      = lbl)
    else
        # Group by the requested DOF value
        groups = Dict{Int, Vector{Int}}()
        for (i, s) in enumerate(states)
            val = s[site_groupby]
            push!(get!(groups, val, Int[]), i)
        end
        for (gi, (val, idxs)) in enumerate(sort!(collect(groups); by = first))
            pts = [Makie.Point2f(coords[i][1], coords[i][2]) for i in idxs]
            lbl = if !isnothing(site_labels)
                site_labels isa AbstractVector ? site_labels[gi] : string(site_labels)
            else
                "$site_groupby=$val"
            end
            Makie.scatter!(ax, pts;
                color      = _cyc(site_colors, gi),
                markersize = site_markersize,
                label      = lbl)
        end
    end

    # ── legend in a narrow right column ──────────────────────────────────────
    has_legend = !isempty(bond_labels) || !isnothing(site_groupby) ||
                 !isnothing(site_labels)
    if has_legend
        Makie.Legend(fig[1, 2], ax)
        Makie.colsize!(fig.layout, 2, Makie.Fixed(120))
    end

    return fig
end

end # module MakieExt
