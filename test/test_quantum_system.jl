"""
Tests for Quantum System (quantum_system.jl)
"""

using Test
using LinearAlgebra
using SingleModeApproximation

@testset "QuantumNumber" begin
    qn = QN(site=1, spin=2)
    @test qn.site == 1
    @test qn.spin == 2
    @test qn[:site] == 1
    @test keys(qn) == (:site, :spin)
    @test values(qn) == (1, 2)

    # Equality
    @test QN(a=1, b=2) == QN(a=1, b=2)
    @test QN(a=1, b=2) != QN(a=1, b=3)

    # From NamedTuple
    qn2 = QuantumNumber((x=3, y=4))
    @test qn2.x == 3
end

@testset "Dof" begin
    dof = Dof(:site, 4)
    @test dof.name == :site
    @test dof.size == 4
    @test dof.labels === nothing

    dof2 = Dof(:spin, 2, [:up, :down])
    @test dof2.labels == [:up, :down]

    @test_throws ErrorException Dof(:spin, 2, [:up, :down, :mid])
end

@testset "SystemDofs" begin
    # Default ordering (reverse): spin varies slowest, site varies fastest
    dofs = SystemDofs([Dof(:site, 4), Dof(:spin, 2)])
    @test ndofs(dofs) == 2
    @test total_dim(dofs) == 8
    @test length(dofs.valid_states) == 8
    # Default sortrule = [2, 1]: spin slow, site fast
    @test dofs.valid_states[1] == QN(site=1, spin=1)
    @test dofs.valid_states[2] == QN(site=2, spin=1)
    @test dofs.valid_states[3] == QN(site=3, spin=1)
    @test dofs.valid_states[5] == QN(site=1, spin=2)

    # With constraint
    constrained = SystemDofs([Dof(:a, 2), Dof(:b, 2)], constraint = qn -> qn.a == qn.b)
    @test total_dim(constrained) == 2
    @test constrained.valid_states[1] == QN(a=1, b=1)
    @test constrained.valid_states[2] == QN(a=2, b=2)

    # Custom sortrule (forward order: site slow, spin fast)
    forward = SystemDofs([Dof(:site, 3), Dof(:spin, 2)], sortrule = [1, 2])
    @test forward.valid_states[1] == QN(site=1, spin=1)
    @test forward.valid_states[2] == QN(site=1, spin=2)
    @test forward.valid_states[3] == QN(site=2, spin=1)

    # Test qn_to_idx dictionary
    @test haskey(dofs.qn_to_idx, QN(site=1, spin=1))
    @test dofs.qn_to_idx[QN(site=1, spin=1)] == 1
    @test dofs.qn_to_idx[QN(site=2, spin=1)] == 2

    # Convenience constructors
    ss = site_spin_system(3)
    @test ndofs(ss) == 2
    @test total_dim(ss) == 6

    # Test blocking with nested sortrule
    # No explicit blocks (single block containing all states)
    no_blocks = SystemDofs([Dof(:site, 3), Dof(:spin, 2)], sortrule = [2, 1])
    @test length(no_blocks.blocks) == 1
    @test no_blocks.blocks[1] == 1:6  # Single block with all 6 states

    # Single-level blocking by spin
    spin_blocked = SystemDofs([Dof(:site, 3), Dof(:spin, 2)], sortrule = [[2], 1])
    @test length(spin_blocked.blocks) == 2
    @test spin_blocked.blocks[1] == 1:3  # spin=1, all sites
    @test spin_blocked.blocks[2] == 4:6  # spin=2, all sites

    # Multi-level blocking (spin and valley)
    multi_blocked = SystemDofs([
        Dof(:site, 2),
        Dof(:spin, 2),
        Dof(:valley, 2)
    ], sortrule = [[3, 2], 1])
    @test length(multi_blocked.blocks) == 4
    @test multi_blocked.blocks[1] == 1:2   # valley=1, spin=1
    @test multi_blocked.blocks[2] == 3:4   # valley=1, spin=2
    @test multi_blocked.blocks[3] == 5:6   # valley=2, spin=1
    @test multi_blocked.blocks[4] == 7:8   # valley=2, spin=2
end

@testset "qn2linear and linear2qn" begin
    dofs = site_spin_system(4)
    # site_spin_system uses default reverse ordering: spin slow, site fast

    @test qn2linear(dofs, QN(site=1, spin=1)) == 1
    @test qn2linear(dofs, QN(site=2, spin=1)) == 2
    @test qn2linear(dofs, QN(site=3, spin=1)) == 3
    @test qn2linear(dofs, QN(site=4, spin=1)) == 4
    @test qn2linear(dofs, QN(site=1, spin=2)) == 5

    # Accept NamedTuple
    @test qn2linear(dofs, (site=1, spin=2)) == 5

    # Round trip
    for i in 1:total_dim(dofs)
        @test qn2linear(dofs, linear2qn(dofs, i)) == i
    end

    @test_throws ErrorException qn2linear(dofs, QN(site=5, spin=1))
end

@testset "Lattice" begin
    # Direct construction
    states = [QN(cell=1, sub=1), QN(cell=1, sub=2)]
    coords = [[0.0, 0.0], [0.5, 0.3]]
    lattice = Lattice([Dof(:cell, 1), Dof(:sub, 2)], states, coords)

    @test length(lattice.position_states) == 2
    @test lattice.supercell_vectors === nothing

    # get_coordinate
    @test get_coordinate(lattice, QN(cell=1, sub=1)) == [0.0, 0.0]
    @test get_coordinate(lattice, (cell=1, sub=2)) == [0.5, 0.3]

    # Tiling construction
    unitcell = Lattice(
        [Dof(:cell, 1), Dof(:sub, 2)],
        [QN(cell=1, sub=1), QN(cell=1, sub=2)],
        [[0.0, 0.0], [0.5, 0.0]]
    )
    a1, a2 = [1.0, 0.0], [0.0, 1.0]
    tiled = Lattice(unitcell, [a1, a2], (2, 2))

    @test length(tiled.position_states) == 8
    @test tiled.position_dofs[1].size == 4
    @test tiled.supercell_vectors == [[2.0, 0.0], [0.0, 2.0]]
end

@testset "Bond and bonds()" begin
    unitcell = Lattice(
        [Dof(:cell, 1), Dof(:sub, 2)],
        [QN(cell=1, sub=1), QN(cell=1, sub=2)],
        [[0.0, 0.0], [0.5, 0.0]]
    )
    lattice = Lattice(unitcell, [[1.0, 0.0], [0.0, 1.0]], (2, 2))

    # Onsite bonds
    onsite = bonds(lattice, (:p, :p), 0)
    @test length(onsite) == 8
    @test length(onsite[1].states) == 1

    # Nearest neighbor (unidirectional - positive direction only)
    nn = bonds(lattice, (:p, :p), 1)
    @test length(nn[1].states) == 2
    @test all(length(b.states) == 2 for b in nn)
    # All bonds should have positive direction delta
    for b in nn
        delta = b.coordinates[2] .- b.coordinates[1]
        @test is_positive_direction(delta)
    end

    # Multiple neighbors
    multi = bonds(lattice, (:p, :p), [0, 1])
    @test length(multi) > length(onsite)
end

@testset "Operators" begin
    qn1, qn2 = QN(site=1, spin=1), QN(site=2, spin=1)

    # FermionOp constructors
    @test c(qn1).dag == false
    @test cdag(qn1).dag == true
    @test c(qn1).qn == qn1

    # Basic constructor with FermionOp vector
    op = Operators(-1.0, [cdag(qn1), c(qn2)])
    @test op.value == -1.0
    @test length(op.ops) == 2

    # Complex coefficient
    op_c = Operators(1.0 + 0.5im, [cdag(qn1), c(qn2)])
    @test op_c.value == 1.0 + 0.5im

    # Constructor with varargs
    op1 = Operators(-1.0, cdag(qn1), c(qn2))
    @test op1.value == -1.0
    @test length(op1.ops) == 2

    # Two-body operators (stored as-is, no reordering at construction)
    qn_up = QN(site=1, spin=1)
    qn_dn = QN(site=1, spin=2)
    op2 = Operators(4.0, cdag(qn_up), c(qn_up), cdag(qn_dn), c(qn_dn))
    @test op2.value == 4.0
    @test length(op2.ops) == 4

    # Pair hopping: stored as c†c†cc (reordering happens in build_interaction_tensor)
    i_up, i_dn = QN(site=1, spin=1), QN(site=1, spin=2)
    j_up, j_dn = QN(site=2, spin=1), QN(site=2, spin=2)
    op3 = Operators(1.0, cdag(i_up), cdag(i_dn), c(j_dn), c(j_up))
    @test op3.value == 1.0  # No sign change at construction
    @test op3.ops[1].dag == true  # First is c†
    @test op3.ops[2].dag == true  # Second is c†
end

@testset "Operators with different input formats" begin
    # One-body with QN
    qn1, qn2 = QN(site=1, spin=1), QN(site=2, spin=1)
    t1 = Operators(-1.0, cdag(qn1), c(qn2))
    @test t1.value == -1.0
    @test length(t1.ops) == 2

    # One-body with NamedTuple (c/cdag accept NamedTuple)
    t2 = Operators(-1.0, cdag((site=1, spin=1)), c((site=2, spin=1)))
    @test t2.ops[1].qn == qn1
    @test t2.ops[2].qn == qn2

    # Two-body
    qn_up = QN(site=1, spin=1)
    qn_dn = QN(site=1, spin=2)
    t4 = Operators(4.0, cdag(qn_up), c(qn_up), cdag(qn_dn), c(qn_dn))
    @test t4.value == 4.0
    @test length(t4.ops) == 4
end

@testset "build_onebody_matrix" begin
    dofs = site_spin_system(4)
    # Default ordering: spin slow, site fast
    # Index mapping: site=1,spin=1 -> 1, site=2,spin=1 -> 2, ..., site=1,spin=2 -> 5, ...
    qn1, qn2 = QN(site=1, spin=1), QN(site=2, spin=1)

    ops = [Operators(-1.0, cdag(qn1), c(qn2)), Operators(-1.0, cdag(qn2), c(qn1))]
    H = build_onebody_matrix(dofs, ops)

    @test size(H) == (8, 8)
    @test H[1, 2] == -1.0  # qn1->1, qn2->2
    @test H[2, 1] == -1.0  # H.c.
    @test ishermitian(H)

    # Complex value test
    ops_c = [Operators(1.0 + 0.5im, cdag(qn1), c(qn2)), Operators(conj(1.0 + 0.5im), cdag(qn2), c(qn1))]
    H_c = build_onebody_matrix(dofs, ops_c)
    @test H_c[1, 2] == 1.0 + 0.5im
    @test H_c[2, 1] == 1.0 - 0.5im  # conjugate
    @test ishermitian(H_c)
end

@testset "build_interaction_tensor" begin
    dofs = site_spin_system(4)
    # Default ordering: spin slow, site fast
    # c†_1↑ c_2↑ c†_3↓ c_4↓  (in InterAll c†c c†c format)
    qn1 = QN(site=1, spin=1)  # site 1, spin up -> index 1
    qn2 = QN(site=2, spin=1)  # site 2, spin up -> index 2
    qn3 = QN(site=3, spin=2)  # site 3, spin down -> index 7
    qn4 = QN(site=4, spin=2)  # site 4, spin down -> index 8
    ops = [Operators(2.5, cdag(qn1), c(qn2), cdag(qn3), c(qn4))]
    V = build_interaction_tensor(dofs, ops)

    @test size(V) == (8, 8, 8, 8)
    # New indices: (1,1)->1, (2,1)->2, (3,2)->7, (4,2)->8
    @test V[1, 2, 7, 8] == 2.5
    @test count(!iszero, V) == 1
end

@testset "generate_onebody" begin
    # Create a simple 2x2 lattice with spin
    unitcell = Lattice(
        [Dof(:cell, 1), Dof(:sub, 2)],
        [QN(cell=1, sub=1), QN(cell=1, sub=2)],
        [[0.0, 0.0], [0.5, 0.0]]
    )
    lattice = Lattice(unitcell, [[1.0, 0.0], [0.0, 1.0]], (2, 2))

    # Full DOFs include spin
    dofs = SystemDofs([Dof(:cell, 4), Dof(:sub, 2), Dof(:spin, 2)])

    # Get nearest neighbor bonds
    nn_bonds = bonds(lattice, (:p, :p), 1)

    # Test 1: Constant hopping (all spin combinations)
    ops = generate_onebody(dofs, nn_bonds, -1.0).ops
    @test length(ops) > 0
    # Each bond generates 4 terms (2 spins × 2 spins) and their H.C. terms.
    @test length(ops) == length(nn_bonds) * 4 * 2

    H = build_onebody_matrix(dofs, ops)
    @test ishermitian(H)

    # Test 2: Spin-diagonal hopping
    ops_diag = generate_onebody(dofs, nn_bonds, (delta, qn1, qn2) ->
        qn1.spin == qn2.spin ? -1.0 : 0.0).ops
    # Each bond generates 2 terms (only same-spin) and their H.C. terms.
    @test length(ops_diag) == length(nn_bonds) * 2 * 2

    H_diag = build_onebody_matrix(dofs, ops_diag)
    @test ishermitian(H_diag)

    # Test 3: Complex hopping
    ops_complex = generate_onebody(dofs, nn_bonds, (delta, qn1, qn2) ->
        qn1.spin == qn2.spin ? (1.0 + 0.5im) : 0.0).ops
    H_complex = build_onebody_matrix(dofs, ops_complex)
    @test ishermitian(H_complex)

    # Test 4: Direction-dependent value
    ops_dir = generate_onebody(dofs, nn_bonds, (delta, qn1, qn2) -> begin
        qn1.spin != qn2.spin && return 0.0
        # Value depends on delta direction
        return delta[1] > 0 ? 1.0 : -1.0
    end).ops
    @test length(ops_dir) > 0
end