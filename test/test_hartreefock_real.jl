"""
Tests for Hartree-Fock module (groundstate/hartreefock_real.jl)
"""

using Test
using LinearAlgebra
using SparseArrays
using MeanFieldTheories

@testset "build_T" begin
    # Setup a simple 2-site, 2-spin system
    dofs = SystemDofs([Dof(:site, 2), Dof(:spin, 2)], sortrule = [[2], 1])
    N = total_dim(dofs)  # Should be 4

    # Create a 2D lattice for hopping
    unitcell = Lattice([Dof(:site, 1)], [QN(site=1)], [[0.0, 0.0]])
    lattice = Lattice(unitcell, [[1.0, 0.0], [0.0, 1.0]], (4, 1))
    nn_bonds = bonds(lattice, (:p, :o), 1)

    # Generate hopping operators: -t c†_i c_j
    t_ops = generate_onebody(dofs, nn_bonds, -1.0).ops

    @testset "Basic functionality" begin
        t_matrix = build_T(dofs, t_ops)

        @test size(t_matrix) == (N, N)
        @test t_matrix isa SparseMatrixCSC
        @test nnz(t_matrix) > 0
        @test ishermitian(t_matrix)
    end

    @testset "Block optimization" begin
        # Create dofs with explicit blocks (spin-conserved)
        dofs_blocked = SystemDofs([Dof(:site, 2), Dof(:spin, 2)], sortrule = [[2], 1])
        # Create dofs without explicit blocks (single block)
        dofs_no_block = SystemDofs([Dof(:site, 2), Dof(:spin, 2)])

        t_with_blocks = build_T(dofs_blocked, t_ops)
        t_without_blocks = build_T(dofs_no_block, t_ops)

        # With explicit blocks should have same or fewer non-zeros (t is block-diagonal)
        @test nnz(t_with_blocks) <= nnz(t_without_blocks)

        # Both should be Hermitian
        @test ishermitian(t_with_blocks)
        @test ishermitian(t_without_blocks)

        # Note: matrices are NOT directly comparable — dofs_blocked and dofs_no_block
        # have different linear orderings (spin-first vs site-first), and block
        # filtering further removes cross-spin elements in the blocked case.
    end

    @testset "Consistency with dense version" begin
        # Use dofs without blocks so no filtering occurs; both functions must agree.
        dofs_no_block = SystemDofs([Dof(:site, 2), Dof(:spin, 2)])
        t_sparse = build_T(dofs_no_block, t_ops)
        t_dense = build_onebody_matrix(dofs_no_block, t_ops)

        @test Matrix(t_sparse) ≈ t_dense
    end

    @testset "Error handling" begin
        @test_throws ErrorException build_T(dofs, Operators[])
    end
end

@testset "build_U" begin
    # Setup a simple 2-site, 2-spin system
    dofs = SystemDofs([Dof(:site, 2), Dof(:spin, 2)], sortrule = [[2], 1])
    N = total_dim(dofs)  # Should be 4

    # Create a simple 2D lattice
    unitcell = Lattice([Dof(:site, 1)], [QN(site=1)], [[0.0, 0.0]])
    lattice = Lattice(unitcell, [[1.0, 0.0], [0.0, 1.0]], (2, 1))
    onsite_bonds = bonds(lattice, (:o, :o), 0)

    # Simple Hubbard U interaction: n_↑ n_↓
    U_ops = generate_twobody(dofs, onsite_bonds,
        (deltas, qn1, qn2, qn3, qn4) ->
            (qn1.site == qn2.site == qn3.site == qn4.site) &&
            (qn1.spin, qn2.spin, qn3.spin, qn4.spin) == (1, 1, 2, 2) ? 1.0 : 0.0,
        order = (cdag, 1, c, 1, cdag, 1, c, 1)).ops

    @testset "Basic functionality" begin
        U_matrix = build_U(dofs, U_ops)

        @test size(U_matrix) == (N^2, N^2)
        @test U_matrix isa SparseMatrixCSC
        @test nnz(U_matrix) > 0
        @test all(isreal, nonzeros(U_matrix))
    end

    @testset "Block optimization" begin
        # Create dofs with explicit blocks (spin-conserved)
        dofs_blocked = SystemDofs([Dof(:site, 2), Dof(:spin, 2)], sortrule = [[2], 1])
        # Create dofs without explicit blocks (single block)
        dofs_no_block = SystemDofs([Dof(:site, 2), Dof(:spin, 2)])

        U_with_blocks = build_U(dofs_blocked, U_ops)
        U_without_blocks = build_U(dofs_no_block, U_ops)

        # With explicit blocks should have fewer or equal non-zeros
        @test nnz(U_with_blocks) <= nnz(U_without_blocks)
    end

    @testset "Consistency with V tensor method" begin
        # Use dofs without blocks so all Hartree+Fock contributions are included;
        # the reference formula and build_U must then agree exactly.
        dofs_no_block = SystemDofs([Dof(:site, 2), Dof(:spin, 2)])
        V = build_interaction_tensor(dofs_no_block, U_ops)

        # Apply 4-term formula manually
        U_ref = zeros(ComplexF64, N, N, N, N)
        for i in 1:N, j in 1:N, k in 1:N, l in 1:N
            U_ref[i,j,k,l] = V[i,j,k,l] + V[k,l,i,j] - V[k,j,i,l] - V[i,l,k,j]
        end

        U_matrix = build_U(dofs_no_block, U_ops)

        @test Matrix(U_matrix) ≈ reshape(U_ref, N^2, N^2)
    end

    @testset "Error handling" begin
        @test_throws ErrorException build_U(dofs, Operators[])
    end
end

@testset "solve_hf — 2×4 Hubbard benchmark" begin
    # Replicate the uhfr.py benchmark: 2×4 Hubbard model at half-filling
    # Expected: total energy ≈ -3.408, converged within ~50 iterations
    Nsite = 8
    Ncond = 8
    U_strength = 8.0
    t_hopping = -1.0

    dofs = SystemDofs([Dof(:site, Nsite), Dof(:spin, 2)], sortrule = [[2], 1])

    # Nearest-neighbour bonds from UHFr/trans.def
    nn_bonds = [
        Bond([QN(site=1), QN(site=3)], [[0.0, 0.0], [0.0, 0.0]]),
        Bond([QN(site=1), QN(site=5)], [[0.0, 0.0], [0.0, 0.0]]),
        Bond([QN(site=1), QN(site=7)], [[0.0, 0.0], [0.0, 0.0]]),
        Bond([QN(site=1), QN(site=8)], [[0.0, 0.0], [0.0, 0.0]]),
        Bond([QN(site=2), QN(site=3)], [[0.0, 0.0], [0.0, 0.0]]),
        Bond([QN(site=2), QN(site=5)], [[0.0, 0.0], [0.0, 0.0]]),
        Bond([QN(site=2), QN(site=7)], [[0.0, 0.0], [0.0, 0.0]]),
        Bond([QN(site=2), QN(site=8)], [[0.0, 0.0], [0.0, 0.0]]),
        Bond([QN(site=3), QN(site=4)], [[0.0, 0.0], [0.0, 0.0]]),
        Bond([QN(site=3), QN(site=6)], [[0.0, 0.0], [0.0, 0.0]]),
        Bond([QN(site=4), QN(site=5)], [[0.0, 0.0], [0.0, 0.0]]),
        Bond([QN(site=4), QN(site=7)], [[0.0, 0.0], [0.0, 0.0]]),
        Bond([QN(site=4), QN(site=8)], [[0.0, 0.0], [0.0, 0.0]]),
        Bond([QN(site=5), QN(site=6)], [[0.0, 0.0], [0.0, 0.0]]),
        Bond([QN(site=6), QN(site=7)], [[0.0, 0.0], [0.0, 0.0]]),
        Bond([QN(site=6), QN(site=8)], [[0.0, 0.0], [0.0, 0.0]]),
    ]
    onsite_bonds = [Bond([QN(site=i)], [[0.0, 0.0]]) for i in 1:Nsite]

    t_ops = generate_onebody(dofs, nn_bonds,
        (delta, qn1, qn2) -> qn1.spin == qn2.spin ? t_hopping : 0.0,
        hc = true).ops

    U_ops = generate_twobody(dofs, onsite_bonds,
        (deltas, qn1, qn2, qn3, qn4) ->
            (qn1.site == qn2.site == qn3.site == qn4.site) &&
            (qn1.spin, qn2.spin, qn3.spin, qn4.spin) == (1, 1, 2, 2) ? U_strength : 0.0,
        order = (cdag, 1, c, 1, cdag, 1, c, 1)).ops

    result = solve_hf(
        dofs,
        vcat(t_ops, U_ops),
        [Ncond ÷ 2, Ncond ÷ 2],
        temperature = 0.0,
        max_iter = 1000,
        n_restarts = 1,
        tol = 1e-8,
        mix_alpha = 0.5,
        verbose = false,
        seed = 123456789,
    )

    @test result.converged
    @test result.energies.total ≈ -3.408 atol=1e-2
    @test result.residual < 1e-6
end
