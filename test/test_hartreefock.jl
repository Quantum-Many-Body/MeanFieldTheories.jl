"""
Tests for Hartree-Fock module (groundstate/hartreefock_real.jl)
"""

using Test
using LinearAlgebra
using SparseArrays
using SingleModeApproximation

@testset "build_t_matrix" begin
    # Setup a simple 2-site, 2-spin system
    dofs = SystemDofs([Dof(:site, 2), Dof(:spin, 2)], sortrule = [[2], 1])
    N = total_dim(dofs)  # Should be 4

    # Create a 2D lattice for hopping
    unitcell = Lattice([Dof(:site, 1)], [QN(site=1)], [[0.0, 0.0]])
    lattice = Lattice(unitcell, [[1.0, 0.0], [0.0, 1.0]], (4, 1))
    nn_bonds = bonds(lattice, (:p, :o), 1)

    # Generate hopping operators: -t c†_i c_j
    t_ops = generate_onebody(dofs, nn_bonds, -1.0)

    @testset "Basic functionality" begin
        t_matrix = build_t_matrix(dofs, t_ops)

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

        t_with_blocks = build_t_matrix(dofs_blocked, t_ops)
        t_without_blocks = build_t_matrix(dofs_no_block, t_ops)

        # With explicit blocks should have same or fewer non-zeros (t is block-diagonal)
        @test nnz(t_with_blocks) <= nnz(t_without_blocks)

        # Both should be Hermitian
        @test ishermitian(t_with_blocks)
        @test ishermitian(t_without_blocks)

        # Both should produce same matrix (blocks is just optimization)
        @test Matrix(t_with_blocks) ≈ Matrix(t_without_blocks)
    end

    @testset "Consistency with dense version" begin
        t_sparse = build_t_matrix(dofs, t_ops)
        t_dense = build_onebody_matrix(dofs, t_ops)

        @test Matrix(t_sparse) ≈ t_dense
    end

    @testset "Error handling" begin
        @test_throws ErrorException build_t_matrix(dofs, Operators[])
    end
end

@testset "build_U_matrix" begin
    # Setup a simple 2-site, 2-spin system
    dofs = SystemDofs([Dof(:site, 2), Dof(:spin, 2)], sortrule = [[2], 1])
    N = total_dim(dofs)  # Should be 4

    # Create a simple 2D lattice
    unitcell = Lattice([Dof(:site, 1)], [QN(site=1)], [[0.0, 0.0]])
    lattice = Lattice(unitcell, [[1.0, 0.0], [0.0, 1.0]], (2, 1))
    onsite_bonds = bonds(lattice, (:o, :o), 0)

    # Simple Hubbard U interaction: n_↑ n_↓
    U_ops = generate_twobody(dofs, onsite_bonds,
        (delta, qn1, qn2, qn3, qn4) ->
            (qn1.site == qn2.site == qn3.site == qn4.site) &&
            (qn1.spin, qn2.spin, qn3.spin, qn4.spin) == (1, 1, 2, 2) ? 1.0 : 0.0,
        order = (cdag, 1, c, 1, cdag, 1, c, 1))

    @testset "Basic functionality" begin
        U_matrix = build_U_matrix(dofs, U_ops)

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

        U_with_blocks = build_U_matrix(dofs_blocked, U_ops)
        U_without_blocks = build_U_matrix(dofs_no_block, U_ops)

        # With explicit blocks should have fewer or equal non-zeros
        @test nnz(U_with_blocks) <= nnz(U_without_blocks)
    end

    @testset "Consistency with V tensor method" begin
        # Compare with reference implementation
        V = build_interaction_tensor(dofs, U_ops)

        # Apply 4-term formula manually
        U_ref = zeros(ComplexF64, N, N, N, N)
        for i in 1:N, j in 1:N, k in 1:N, l in 1:N
            U_ref[i,j,k,l] = V[i,j,k,l] + V[k,l,i,j] - V[k,j,i,l] - V[i,l,k,j]
        end

        U_matrix = build_U_matrix(dofs, U_ops)

        @test Matrix(U_matrix) ≈ reshape(U_ref, N^2, N^2)
    end

    @testset "Error handling" begin
        @test_throws ErrorException build_U_matrix(dofs, Operators[], dofs.blocks)
    end
end
