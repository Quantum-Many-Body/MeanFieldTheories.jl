# Quantum System Tutorial

## Introduction

The `quantumsystem` module provides a flexible framework for constructing quantum many-body Hamiltonians in Julia. It is designed to:

- Define quantum systems with arbitrary degrees of freedom (spin, orbital, sublattice, valley, etc.)
- Construct lattice structures with various geometries (square, honeycomb, triangular, etc.)
- Generate operator terms using symbolic notation with flexible orderings
- Build matrix and tensor representations for numerical calculations
- Automatically handle fermionic anticommutation relations and operator reordering

This tutorial demonstrates how to use the module through two cutting-edge examples from recent literature:

1. **Twisted bilayer MoTe₂ Kane-Mele model** (2025): A tight-binding model with topological bands and valley-dependent complex hopping
2. **Multi-orbital extended Hubbard model**: Including Hund's coupling, pair hopping, and orbital-dependent interactions

---

## Core Concepts

### 1. Quantum Numbers and Degrees of Freedom

Quantum numbers are represented as named tuples that index the Hilbert space. The `SystemDofs` structure defines all quantum degrees of freedom:

```julia
using SingleModeApproximation

# Example: System with site, sublattice, and valley
dofs = SystemDofs(
    site = 1:16,             # 16 sites
    sublattice = [:A, :B],   # Two sublattices (honeycomb)
    valley = [:K_plus, :K_minus]  # ±K valleys (or spin up/down)
)

# Create a quantum number
qn = QN(site=1, sublattice=:A, valley=:K_plus)

# Linear indexing
idx = qn2linear(dofs, qn)      # Convert to linear index
qn_back = linear2qn(dofs, idx)  # Convert back
```

**Key features:**
- Named tuple representation: natural mathematical notation
- Automatic linear indexing for matrix/tensor construction
- Arbitrary quantum numbers supported (spin, orbital, layer, sublattice, valley, etc.)

### 2. Lattice Structure

Lattices define spatial geometry and neighbor relationships:

```julia
# Honeycomb lattice (for MoTe₂, graphene, etc.)
lattice = Lattice(:Honeycomb, 4, 4, pbc=true)

# Square lattice (for cuprates, iron-based superconductors, etc.)
lattice = Lattice(:Square, 6, 6, pbc=true)

# Get neighbor bonds
onsite_bonds = bonds(lattice, 0)   # On-site (delta=0)
nn_bonds = bonds(lattice, 1)       # Nearest neighbors
nnn_bonds = bonds(lattice, 2)      # Next-nearest neighbors

# Each bond contains: site indices, displacement vector
for bond in nn_bonds[1:3]
    println("Bond: sites $(bond.i) -> $(bond.j), delta=$(bond.delta)")
    println("  Direction: ", is_positive_direction(lattice, bond.delta))
end
```

### 3. Operators

The module provides symbolic operator construction:

- **`c(qn)`**: Annihilation operator at quantum number `qn`
- **`cdag(qn)`**: Creation operator at quantum number `qn`
- **`Operators(value, op1, op2, ...)`**: Operator product with coefficient

```julia
qn1 = QN(site=1, valley=:K_plus)
qn2 = QN(site=2, valley=:K_plus)

# Hopping term: -t c†₁ c₂
term = Operators(-1.0, cdag(qn1), c(qn2))

# Complex hopping: (0.81 - 1.32i) c†₁ c₂
term_complex = Operators(0.81 - 1.32im, cdag(qn1), c(qn2))
```

**Display in natural notation:**
```julia
terms = [
    Operators(-3.49, cdag(QN(site=1, sublattice=:A)), c(QN(site=2, sublattice=:B))),
    Operators(0.81-1.32im, cdag(QN(site=1, sublattice=:A)), c(QN(site=3, sublattice=:A)))
]
display(terms)
# Output:
# Operators with 2 terms:
#   -3.49 c†_{1,A}c_{2,B}
#   + (0.81 - 1.32im) c†_{1,A}c_{3,A}
```

### 4. Term Generators

High-level functions generate common operator terms:

**One-body generators:**
```julia
generate_onebody(dofs, bonds, value; order=(cdag, 1, c, 2))
```
- `dofs`: Degrees of freedom
- `bonds`: Spatial bonds (from lattice)
- `value`: Coefficient (Number) or function `(delta, qn1, qn2) -> Number`
- `order`: Operator pattern, e.g., `(cdag, 1, c, 2)` means c†₁c₂

**Two-body generators:**
```julia
generate_twobody(dofs, bonds, value; order=(cdag, 1, c, 1, cdag, 2, c, 2))
```

**Convenience functions:**
- `generate_coulomb_intra`, `generate_hund`, `generate_ising`, etc.

### 5. Matrix/Tensor Construction

```julia
# One-body Hamiltonian matrix
H_matrix = build_onebody_matrix(dofs, onebody_terms)

# Two-body interaction tensor
V_tensor = build_interaction_tensor(dofs, twobody_terms)
```

---

## Example 1: Twisted Bilayer MoTe₂ Kane-Mele Model

**Reference:** Qiu & Wu, *Topological magnons and domain walls in twisted bilayer MoTe₂*, Phys. Rev. B **112**, 085132 (2025)

### Model Description

The tight-binding Hamiltonian for twisted bilayer MoTe₂ projected onto Wannier states is:

$$
\hat{H}_{KM} = \sum_{\tau, \alpha\beta, RR'} t^\tau_{\alpha\beta}(R-R') b^\dagger_{R\alpha\tau} b_{R'\beta\tau}
$$

where:
- $b^\dagger_{R\alpha\tau}$ creates a hole at moiré unit cell $R$, sublattice $\alpha$ (A or B), valley $\tau$ (± or equivalently spin ↑↓)
- $t^\tau_{\alpha\beta}(R-R')$ are hopping parameters up to 5th nearest neighbors

**Key features:**
1. **Nearest-neighbor hopping** $t_1$: Real, connects A-B sublattices
2. **Next-nearest-neighbor hopping** $t_2$: **Complex** with phase factor $e^{i\phi_t \tau \nu_{ij}}$
   - $\tau = \pm$ for valley (spin-valley locking)
   - $\nu_{ij} = +1$ if hopping follows dashed arrows in Fig. 1, $-1$ if against
3. **Extended hoppings** $t_3, t_4, t_5$: Connect further neighbors

This is a **generalized Kane-Mele model** - two copies of time-reversal partner Haldane models - leading to topological Chern bands.

**Parameters at twist angle θ = 3.5°** (from Fig. 1):
- $t_1 = -3.49$ meV (nearest neighbor, real)
- $t_2 = 0.81 - 1.32i$ meV (next-nearest neighbor, complex with direction-dependent phase)
- $t_3 = 0.72$ meV
- $t_4 = -0.26$ meV
- $t_5 = 0.08$ meV

### Implementation

```julia
using SingleModeApproximation
using LinearAlgebra

# ========================================
# System setup
# ========================================
Lx, Ly = 4, 4
lattice = Lattice(:Honeycomb, Lx, Ly, pbc=true)
nsites = lattice.nsites

# Degrees of freedom: site, sublattice (A/B), valley (± or ↑↓)
# Valley ± is locked to spin ↑↓ in MoTe₂
dofs = SystemDofs(
    site = 1:nsites,
    sublattice = [:A, :B],
    valley = [:plus, :minus]  # τ = ± (or equivalently spin ↑↓)
)

println("Twisted MoTe₂ Kane-Mele model")
println("System size: $nsites sites")
println("Hilbert space dimension: ", length(dofs))

# ========================================
# Parameters (from Qiu & Wu, PRB 2025, Fig. 1, θ = 3.5°)
# ========================================
t1 = -3.49          # meV, nearest-neighbor (real)
t2_mag = 0.81       # meV, magnitude of next-nearest-neighbor
t2_phase = -1.32    # meV, imaginary part (encodes φ_t and direction)
t3 = 0.72           # meV
t4 = -0.26          # meV
t5 = 0.08           # meV

# ========================================
# 1. Nearest-neighbor hopping: t1 (A-B)
# ========================================
nn_bonds = bonds(lattice, 1)

function nn_hopping(delta, qn1, qn2)
    # t1 connects A to B sublattices only
    # Same valley (no valley mixing)
    if qn1.sublattice != qn2.sublattice && qn1.valley == qn2.valley
        return t1
    end
    return 0.0
end

t1_terms = generate_onebody(
    dofs, nn_bonds, nn_hopping;
    order = (cdag, 1, c, 2)
).ops

# ========================================
# 2. Next-nearest-neighbor hopping: t2 (A-A or B-B)
#    Complex with valley and direction dependent phase
# ========================================
nnn_bonds = bonds(lattice, 2)

function nnn_hopping(delta, qn1, qn2)
    # t2 connects same sublattice (A-A or B-B)
    # Has phase factor: exp(i φ_t τ ν_ij)
    if qn1.sublattice == qn2.sublattice && qn1.valley == qn2.valley
        # Determine τ: +1 for valley=plus, -1 for valley=minus
        tau = (qn1.valley == :plus) ? 1 : -1

        # Determine ν_ij based on hopping direction
        # On honeycomb lattice, next-nearest neighbors have specific directions
        # ν_ij = +1 if following clockwise path around hexagon, -1 if counterclockwise
        # This is encoded in delta and sublattice
        # For honeycomb lattice, use displacement vector to determine circulation

        # Get coordinates to determine direction
        coord1 = get_coordinate(lattice, qn1.site)
        coord2 = get_coordinate(lattice, qn2.site)
        displacement = coord2 - coord1

        # Determine circulation based on displacement and sublattice
        # This follows the dashed arrows in Fig. 1 of the paper
        # Simplified version: use cross product with reference direction
        # For sublattice A: counterclockwise is positive
        # For sublattice B: clockwise is positive
        if qn1.sublattice == :A
            # Example criterion (adjust based on actual lattice geometry)
            nu_ij = (displacement[1] * 0.5 + displacement[2] > 0) ? 1 : -1
        else  # sublattice B
            nu_ij = (displacement[1] * 0.5 + displacement[2] > 0) ? -1 : 1
        end

        # Phase factor: exp(i φ_t τ ν_ij)
        # From t2 = 0.81 - 1.32i, we extract the phase
        # t2 = |t2| * exp(i * arg(t2)) = |t2| * exp(i φ_t τ ν_ij)
        # For simplicity, use the full complex t2 with direction dependence
        phase_factor = tau * nu_ij
        t2_complex = (t2_mag + im * t2_phase)

        # Apply phase: if phase_factor is negative, take complex conjugate
        if phase_factor > 0
            return t2_complex
        else
            return conj(t2_complex)
        end
    end
    return 0.0
end

t2_terms = generate_onebody(
    dofs, nnn_bonds, nnn_hopping;
    order = (cdag, 1, c, 2)
).ops

# ========================================
# 3. Third-nearest-neighbor hopping: t3
# ========================================
tn3_bonds = bonds(lattice, 3)

function t3_hopping(delta, qn1, qn2)
    # Real hopping, same valley
    if qn1.valley == qn2.valley
        return t3
    end
    return 0.0
end

t3_terms = generate_onebody(
    dofs, tn3_bonds, t3_hopping;
    order = (cdag, 1, c, 2)
).ops

# ========================================
# 4. Fourth-nearest-neighbor hopping: t4
# ========================================
tn4_bonds = bonds(lattice, 4)

function t4_hopping(delta, qn1, qn2)
    if qn1.valley == qn2.valley
        return t4
    end
    return 0.0
end

t4_terms = generate_onebody(
    dofs, tn4_bonds, t4_hopping;
    order = (cdag, 1, c, 2)
).ops

# ========================================
# 5. Fifth-nearest-neighbor hopping: t5
# ========================================
tn5_bonds = bonds(lattice, 5)

function t5_hopping(delta, qn1, qn2)
    if qn1.valley == qn2.valley
        return t5
    end
    return 0.0
end

t5_terms = generate_onebody(
    dofs, tn5_bonds, t5_hopping;
    order = (cdag, 1, c, 2)
).ops

# ========================================
# 6. Coulomb interactions (optional)
# ========================================
onsite_bonds = bonds(lattice, 0)
U = 10.0  # On-site Coulomb repulsion (meV)

coulomb_terms = generate_coulomb_intra(dofs, onsite_bonds, U)

# ========================================
# Combine all hopping terms
# ========================================
all_hopping_terms = vcat(
    t1_terms,
    t2_terms,
    t3_terms,
    t4_terms,
    t5_terms
)

# Build one-body Hamiltonian
H_KM = build_onebody_matrix(dofs, all_hopping_terms)
V_int = build_interaction_tensor(dofs, coulomb_terms)

# ========================================
# Analysis
# ========================================
println("\n" * "="^70)
println("Twisted Bilayer MoTe₂: Kane-Mele Model (Eq. 2 from PRB 112, 085132)")
println("="^70)
println("Parameters at θ = 3.5°:")
println("  t1 = $t1 meV (NN, real)")
println("  t2 = $(t2_mag) + $(t2_phase)i meV (NNN, complex)")
println("  t3 = $t3 meV")
println("  t4 = $t4 meV")
println("  t5 = $t5 meV")
println("  U = $U meV (on-site Coulomb)")

println("\nSystem properties:")
println("  Lattice: Honeycomb $(Lx)×$(Ly), $nsites sites")
println("  Hilbert space: ", size(H_KM, 1), " states")
println("  Hopping terms: ", length(all_hopping_terms))
println("    - t1 (NN): ", length(t1_terms))
println("    - t2 (NNN): ", length(t2_terms))
println("    - t3: ", length(t3_terms))
println("    - t4: ", length(t4_terms))
println("    - t5: ", length(t5_terms))
println("  Interaction terms: ", length(coulomb_terms))

# Diagonalize and analyze spectrum
eigenvalues = eigvals(Hermitian(H_KM))
println("\nOne-body band structure:")
println("  Bandwidth: $(rd(maximum(real(eigenvalues)) - minimum(real(eigenvalues)))) meV")
println("  Energy range: [$(rd(minimum(real(eigenvalues)))), $(rd(maximum(real(eigenvalues))))] meV")

# Check if there's a gap (topological bands have gaps)
sorted_eigs = sort(real(eigenvalues))
mid_idx = length(sorted_eigs) ÷ 2
if mid_idx > 0 && mid_idx < length(sorted_eigs)
    gap = sorted_eigs[mid_idx+1] - sorted_eigs[mid_idx]
    println("  Gap around Fermi level: $(rd(gap)) meV")
end

# Check Hermiticity
hermiticity_error = norm(H_KM - H_KM')
println("\nHermiticity check: ||H - H†|| = $(hermiticity_error)")

# Display sample terms
println("\n" * "-"^70)
println("Sample t1 (nearest-neighbor) terms:")
println("-"^70)
display(t1_terms[1:min(4, length(t1_terms))])

println("\n" * "-"^70)
println("Sample t2 (next-nearest-neighbor, complex) terms:")
println("-"^70)
display(t2_terms[1:min(4, length(t2_terms))])
```

### Key Features Explained

1. **Valley-dependent phase (Kane-Mele mechanism)**:
   - Next-nearest-neighbor hopping $t_2$ has phase factor $e^{i\phi_t \tau \nu_{ij}}$
   - $\tau = \pm$ distinguishes ±K valleys (or spin ↑↓)
   - $\nu_{ij} = \pm 1$ depends on circulation direction around hexagon
   - This breaks time-reversal within each valley but preserves combined $\mathcal{T}$ symmetry
   - Leads to opposite Chern numbers ±1 in the two valleys

2. **Topological properties**:
   - Two moiré valence bands with Chern numbers ±1 per valley
   - Quantum anomalous Hall effect at ν = 1 filling
   - Topological magnon excitations with chiral edge states

3. **Wannier basis**:
   - A sublattice: polarized to top layer (t)
   - B sublattice: polarized to bottom layer (b)
   - Reflects layer-sublattice locking in twisted system

### Extensions and Applications

This model serves as the foundation for studying:
- **Topological magnons** (intervalley spin-flip excitations with Chern numbers)
- **Domain walls** between regions with opposite Chern numbers
- **Fractional quantum anomalous Hall states** at other fillings
- **Magnetic ordering temperature** via effective spin models

---

## Example 2: Multi-Orbital Extended Hubbard Model

### Model Description

The multi-orbital extended Hubbard model describes strongly correlated systems with multiple orbitals per site. The Hamiltonian includes:

$$
H = \sum_{ii', mm', \sigma} t^{ii'}_{mm'} c^\dagger_{im\sigma} c_{i'm'\sigma} + H_{\text{int}}
$$

The interaction part can be decomposed as:

$$
H_{\text{int}} = \sum_i \left[ \sum_m U n_{m\uparrow} n_{m\downarrow} + \sum_{m \neq m', \sigma\sigma'} U' n_{m\sigma} n_{m'\sigma'} + \sum_{m \neq m'} J c^\dagger_{m\uparrow} c^\dagger_{m\downarrow} c_{m'\downarrow} c_{m'\uparrow} - \sum_{m \neq m'} J c^\dagger_{m\uparrow} c_{m\downarrow} c^\dagger_{m'\downarrow} c_{m'\uparrow} \right]
$$

where:
- **U**: Intra-orbital Coulomb repulsion
- **U'**: Inter-orbital Coulomb repulsion
- **J**: Hund's coupling (favors parallel spins) and spin-exchange term

### Implementation

```julia
using SingleModeApproximation
using LinearAlgebra

# ========================================
# System setup
# ========================================
Lx, Ly = 4, 4
lattice = Lattice(:Square, Lx, Ly, pbc=true)
nsites = lattice.nsites

# Parameters
t1 = 1.0      # Hopping for orbital 1 (eV)
t2 = 0.8      # Hopping for orbital 2 (eV)
t12 = 0.2     # Inter-orbital hopping (eV)
U1 = 4.0      # Intra-orbital Coulomb for orbital 1 (eV)
U2 = 3.5      # Intra-orbital Coulomb for orbital 2 (eV)
U_prime = 2.0 # Inter-orbital Coulomb (eV)
J_H = 0.6     # Hund's coupling (eV)

# Degrees of freedom
dofs = SystemDofs(
    site = 1:nsites,
    orbital = 1:2,
    spin = [:up, :dn]
)

println("Multi-Orbital Extended Hubbard Model on $(Lx)×$(Ly) square lattice")
println("Hilbert space dimension: ", length(dofs))

# ========================================
# ONE-BODY TERMS
# ========================================
nn_bonds = bonds(lattice, 1)
onsite_bonds = bonds(lattice, 0)

# Intra-orbital hopping
function intra_hopping(delta, qn1, qn2)
    if qn1.orbital == qn2.orbital
        return (qn1.orbital == 1) ? -t1 : -t2
    end
    return 0.0
end

hopping_intra = generate_onebody(dofs, nn_bonds, intra_hopping;
    order = (cdag, 1, c, 2)).ops

# Inter-orbital hopping
function inter_hopping(delta, qn1, qn2)
    return (qn1.orbital != qn2.orbital) ? -t12 : 0.0
end

hopping_inter = generate_onebody(dofs, nn_bonds, inter_hopping;
    order = (cdag, 1, c, 2)).ops

# ========================================
# TWO-BODY INTERACTION TERMS
# ========================================

# Intra-orbital Coulomb: U n_{m↑} n_{m↓}
function intra_coulomb(delta, qn1, qn2, qn3, qn4)
    if (qn1.orbital == qn2.orbital == qn3.orbital == qn4.orbital &&
        qn1.spin != qn2.spin)
        return (qn1.orbital == 1) ? U1 : U2
    end
    return 0.0
end

coulomb_intra = generate_twobody(dofs, onsite_bonds, intra_coulomb;
    order = (cdag, 1, c, 1, cdag, 2, c, 2))

# Inter-orbital Coulomb: U' n_{mσ} n_{m'σ'}
function inter_coulomb(delta, qn1, qn2, qn3, qn4)
    return (qn1.orbital != qn3.orbital) ? U_prime : 0.0
end

coulomb_inter = generate_twobody(dofs, onsite_bonds, inter_coulomb;
    order = (cdag, 1, c, 1, cdag, 2, c, 2))

# Hund's coupling and spin exchange
hund_terms = generate_hund(dofs, onsite_bonds, J_H)

# ========================================
# Build Hamiltonians
# ========================================
hopping_terms = vcat(hopping_intra, hopping_inter)
interaction_terms = vcat(coulomb_intra, coulomb_inter, hund_terms)

H_onebody = build_onebody_matrix(dofs, hopping_terms)
V_twobody = build_interaction_tensor(dofs, interaction_terms)

# ========================================
# Analysis
# ========================================
println("\n" * "="^70)
println("Multi-Orbital Extended Hubbard Model")
println("="^70)
println("Parameters:")
println("  t1 = $t1 eV, t2 = $t2 eV, t12 = $t12 eV")
println("  U1 = $U1 eV, U2 = $U2 eV, U' = $U_prime eV, J = $J_H eV")

println("\nSystem properties:")
println("  One-body terms: ", length(hopping_terms))
println("  Two-body terms: ", length(interaction_terms))
println("  Matrix size: ", size(H_onebody))
println("  Tensor size: ", size(V_twobody))

eigenvalues = eigvals(Hermitian(H_onebody))
println("\nOne-body spectrum:")
println("  Bandwidth: $(rd(maximum(real(eigenvalues)) - minimum(real(eigenvalues)))) eV")

println("\nSample Hund's coupling terms:")
display(hund_terms[1:min(4, length(hund_terms))])
```

### Key Features

1. **Orbital differentiation**: Each orbital has distinct hopping and Coulomb U
2. **Hund's coupling**: Favors high-spin configurations (parallel spins in different orbitals)
3. **Rich phase diagram**: Competition between U, U', J drives various orders

---

## Summary

The `quantumsystem` module enables rapid construction of quantum many-body Hamiltonians:

✅ **Flexible degrees of freedom** - valleys, orbitals, sublattices, spins
✅ **Complex parameters** - direction-dependent phases, valley-dependent couplings
✅ **Automatic sign handling** - fermionic anticommutation built-in
✅ **Topological models** - Kane-Mele, Haldane, Chern insulators
✅ **Strong correlations** - Extended Hubbard, Hund's coupling

### Typical Workflow

1. Define `SystemDofs` and `Lattice`
2. Use `generate_onebody`/`generate_twobody` with value functions
3. Build matrices with `build_onebody_matrix`/`build_interaction_tensor`
4. Solve using diagonalization, DMFT, ED, DMRG, etc.

This framework accelerates research in twisted moiré materials, topological insulators, and strongly correlated systems!

## References

- Qiu, W.-X. & Wu, F., *Topological magnons and domain walls in twisted bilayer MoTe₂*, Phys. Rev. B **112**, 085132 (2025)
