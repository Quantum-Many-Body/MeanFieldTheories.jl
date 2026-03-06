# Quantum System

The `quantumsystem` module provides a flexible, general-purpose framework for constructing quantum many-body Hamiltonians. It is organized around three levels that follow a natural physical hierarchy:

**Degrees of freedom → Lattice → Operators**

Each level is deliberately general: no restrictions are imposed on the number or nature of quantum numbers, the geometry of the lattice, or the form of operator terms. Physical constraints are enforced entirely through user-supplied functions, without modifying the library.

---

## 1. Degrees of Freedom

### Quantum numbers

A single-particle state is labelled by a `QuantumNumber` — a named-tuple wrapper that provides clean field access and display:

```julia
using MeanFieldTheories

qn = QN(site=2, spin=1, orbital=2)
qn.site      # 2
qn.spin      # 1
```

The `QN` alias is equivalent to `QuantumNumber`.

### Defining the Hilbert space

`SystemDofs` is built from a list of `Dof` objects, one per quantum-number component:

```julia
dofs = SystemDofs([
    Dof(:site, 4),
    Dof(:spin, 2),
    Dof(:orbital, 3)
])
```

`Dof(:name, n)` declares that this component takes integer values `1:n`. An optional label vector can be supplied for readability:

```julia
Dof(:spin, 2, [:up, :down])
```

The full set of valid single-particle states is the Cartesian product of all components:

```julia
dofs.valid_states   # Vector of QuantumNumber, length = 4 × 2 × 3 = 24
```

Any combination of components is supported — site, spin, orbital, sublattice, valley, layer, and so on — with no upper limit.

A `constraint` function can restrict the state space to a physically relevant subspace:

```julia
# Only keep states with site ≤ 2
dofs = SystemDofs([Dof(:site, 4), Dof(:spin, 2)],
    constraint = qn -> qn.site <= 2)
```

### Sorting and block structure

The order in which states appear in `valid_states` (and thus the row/column ordering of matrices) is controlled by `sortrule`, a permutation of `1:ndofs`:

```julia
# Default: reverse order, i.e. last DOF varies slowest
# [Dof 2 slow, Dof 1 fast] for a 2-DOF system
dofs = SystemDofs([Dof(:site, 3), Dof(:spin, 2)])
# valid_states: QN(site=1,spin=1), QN(site=2,spin=1), QN(site=3,spin=1),
#               QN(site=1,spin=2), QN(site=2,spin=2), QN(site=3,spin=2)
```

When a conserved quantum number makes the Hamiltonian block-diagonal, wrapping its index in a nested vector declares it as the block label:

```julia
# Spin is conserved → two blocks: spin=1 (indices 1:3) and spin=2 (indices 4:6)
dofs = SystemDofs([Dof(:site, 3), Dof(:spin, 2)], sortrule = [[2], 1])
dofs.blocks   # [1:3, 4:6]
```

Blocks are exploited automatically by the mean-field solvers to reduce memory and computation: the interaction matrix only needs to store elements within each block, giving a factor of $B^2$ reduction for $B$ symmetry blocks (e.g., 75% reduction for two spin blocks).

Multiple nested levels can be specified for simultaneous conservation of several quantum numbers:

```julia
# Block by valley first, then by spin within each valley block
dofs = SystemDofs([
    Dof(:site, 4), Dof(:orbital, 2), Dof(:spin, 2), Dof(:valley, 2)
], sortrule = [[4, 3], 2, 1])
# → 4 blocks: (K,↑), (K,↓), (K′,↑), (K′,↓)
```

### Index conversion

Linear indices map quantum numbers to matrix row/column positions. The lookup is O(1) via an internal hash table:

```julia
idx = qn2linear(dofs, QN(site=2, spin=1))   # Int
qn  = linear2qn(dofs, idx)                  # QuantumNumber
```

---

## 2. Lattice

### Unit cell

A `Lattice` maps position states to real-space coordinates. It is constructed from position degrees of freedom, the list of position states (one per site in the unit cell), and their Cartesian coordinates:

```julia
# Single-site square-lattice unit cell
unitcell = Lattice(
    [Dof(:cell, 1)],    # first DOF must have size=1 for tiling
    [QN(cell=1)],
    [[0.0, 0.0]]
)

# Two-site (honeycomb) unit cell
unitcell = Lattice(
    [Dof(:cell, 1), Dof(:sublattice, 2, [:A, :B])],
    [QN(cell=1, sublattice=1), QN(cell=1, sublattice=2)],
    [[0.0, 0.0], [0.5, 0.289]]
)
```

The position DOFs of a `Lattice` can be a subset of the full system DOFs: `SystemDofs` may include additional internal DOFs (spin, orbital, ...) that the lattice does not know about. This separation means spatial and internal structure can be defined independently.

### Supercell construction

The physical simulation cell is built by tiling the unit cell along primitive lattice vectors. The first DOF of the unit cell (which must have `size=1`) is expanded to the total number of unit cells:

```julia
a1, a2 = [1.0, 0.0], [0.0, 1.0]
lattice = Lattice(unitcell, [a1, a2], (4, 4))
# First DOF now has size = 16 (= 4×4 unit cells)
# supercell_vectors = [4*a1, 4*a2] are set automatically
```

The `supercell_vectors` field is used for periodic boundary conditions in bond generation and for the reciprocal-space grid in momentum-space HF.

Site coordinates can be queried by quantum number, even when the QN contains extra components beyond the position DOFs:

```julia
get_coordinate(lattice, QN(cell=3, sublattice=2))
get_coordinate(lattice, QN(cell=3, sublattice=2, spin=1))  # extra keys are ignored
```

### Bonds

A `Bond` connects one or more sites and stores, for each site:

- `states`: the position QN (site label within the unit cell),
- `coordinates`: absolute (unwrapped) Cartesian coordinate,
- `icoordinates`: lattice vector of the unit cell containing that site.

The `icoordinates` field is essential for momentum-space calculations: the difference `icoordinates[2] - icoordinates[1]` gives the integer lattice vector crossed by the bond, which becomes the displacement $\mathbf{R}$ in $T_{\mathbf{R}}$. For bonds that cross a periodic boundary, the periodic image site carries a non-zero `icoordinate` shift (e.g., `[-1.0, 0.0]`), while the originating site has `[0.0, 0.0]`.

`bonds(lattice, boundary, neighbors)` generates all bonds of a given neighbor shell. The boundary tuple specifies periodic (`:p`) or open (`:o`) conditions per spatial direction:

```julia
# Onsite bonds (one site per bond)
onsite = bonds(lattice, (:p, :p), 0)

# Nearest-neighbor bonds (two sites per bond), periodic in both directions
nn = bonds(lattice, (:p, :p), 1)

# Next-nearest-neighbor, open in y
nnn = bonds(lattice, (:p, :o), 2)

# Multiple shells at once
nn_and_nnn = bonds(lattice, (:p, :p), [1, 2])

# Specific distances
specific = bonds(lattice, (:p, :p), [1.0, 1.732])
```

Bonds are not restricted to two sites. Three-site and four-site bonds can be constructed manually (`Bond(states, coordinates, icoordinates)`) and passed to `generate_twobody` with the appropriate `order` to model ring-exchange and other multi-site interactions.

---

## 3. Operators

### Primitive operators and operator terms

Single creation and annihilation operators are constructed from quantum numbers:

```julia
i = QN(site=1, spin=1)
j = QN(site=2, spin=1)

cdag(i)   # creation  c†_{site=1, spin=1}
c(j)      # annihilation c_{site=2, spin=1}
```

An `Operators` object is a product of primitive operators with a (possibly complex) coefficient:

```julia
# Hopping term: -t c†_i c_j
Operators(-1.0, [cdag(i), c(j)])

# Complex hopping
Operators(0.81 - 1.32im, [cdag(i), c(j)])

# Four-operator interaction: U c†_↑ c_↑ c†_↓ c_↓
Operators(U, [cdag(QN(site=1,spin=1)), c(QN(site=1,spin=1)),
              cdag(QN(site=1,spin=2)), c(QN(site=1,spin=2))])
```

Operators are stored in the user-supplied order; reordering to canonical InterAll form (c†c c†c …) happens automatically during matrix construction.

### Generating one-body terms

`generate_onebody` iterates over all bonds and all combinations of internal DOFs, calling a user-supplied `value` function to assign the coefficient for each term:

```julia
onebody = generate_onebody(dofs, bonds, value; order=(cdag, 1, c, 2), hc=true)
```

**Arguments:**
- `value`: a `Number` (uniform coefficient) or a function `(delta, qn1, qn2) -> Number`.
  `delta` is generated as `bond.coordinates[1] - bond.coordinates[2]` — the vector
  from site 2 to site 1, encoding the full bond direction (magnitude and angle).
- `order`: `(op_type1, site1, op_type2, site2)` — which site of the bond each operator
  acts on. Default `(cdag, 1, c, 2)` places the creation operator on site 1 and the
  annihilation operator on site 2, giving the standard hopping $c^\dagger_1 c_2$.
- `hc=true`: automatically appends the Hermitian conjugate of each term with the
  displacement sign flipped. Together with a `value` function that returns real
  coefficients, this guarantees a Hermitian Hamiltonian without manual bookkeeping.

**Return value:** a NamedTuple `(ops, delta, irvec)` — three parallel vectors.
`irvec[n] = icoordinates[2] - icoordinates[1]` for term `n` is the unit-cell
displacement, which is passed directly to `build_Tr` for momentum-space HF.

**Example — spin-conserving nearest-neighbor hopping:**

```julia
t_onebody = generate_onebody(dofs, nn_bonds,
    (delta, qn1, qn2) -> qn1.spin == qn2.spin ? -1.0 : 0.0)
# t_onebody is a NamedTuple with three parallel vectors:
#   t_onebody.ops   → Vector{Operators}       (one entry per generated term; hc=true doubles the count)
#   t_onebody.delta → Vector{Vector{Float64}} (physical bond vector for each term)
#   t_onebody.irvec → Vector{Vector{Float64}} (unit-cell displacement R for each term; used by build_Tr)
t_ops = t_onebody.ops
```

**Example — direction-dependent complex hopping:**

The `delta` vector carries the full bond direction, enabling valley- or direction-dependent phases without any separate bookkeeping:

```julia
t2_onebody = generate_onebody(dofs, nnn_bonds,
    (delta, qn1, qn2) -> begin
        qn1.spin != qn2.spin && return 0.0          # spin-conserving
        tau = qn1.spin == 1 ? +1 : -1               # valley sign
        nu  = sign(delta[1] + 0.5*delta[2]) |> Int  # circulation direction
        return (0.81 + 1.32im) * (tau * nu)
    end)
# t2_onebody.ops   → Vector{Operators} with complex coefficients
# t2_onebody.delta → Vector{Vector{Float64}} encoding bond direction (used above for nu)
# t2_onebody.irvec → Vector{Vector{Float64}} encoding lattice vector R
t2_ops = t2_onebody.ops
```

**Example — orbital-selective intra- and inter-orbital hopping:**

```julia
# Intra-orbital: different hopping per orbital, same spin
t_intra = generate_onebody(dofs, nn_bonds,
    (delta, qn1, qn2) -> begin
        qn1.spin != qn2.spin && return 0.0
        qn1.orbital != qn2.orbital && return 0.0
        qn1.orbital == 1 ? -1.0 : -0.8
    end).ops
# .ops → Vector{Operators}: one term per (bond, spin, orbital) combination passing the filter

# Inter-orbital: mixing between orbitals
t_inter = generate_onebody(dofs, nn_bonds,
    (delta, qn1, qn2) ->
        qn1.spin == qn2.spin && qn1.orbital != qn2.orbital ? -0.2 : 0.0).ops
```

### Generating two-body terms

`generate_twobody` follows the same pattern for four-operator interaction terms, but offers additional generality:

```julia
twobody = generate_twobody(dofs, bonds, value; order=(cdag, 1, c, 1, cdag, 2, c, 2))
```

**Arguments:**
- `value`: a `Number` or a function `(deltas, qn1, qn2, qn3, qn4) -> Number`.
  `deltas` is generated as `deltas[i] = bond.coordinates[i] - bond.coordinates[nb]`
  for `i = 1 … nb-1`, where `nb = length(bond.states)` is the number of bond sites
  and the **last site is the reference (origin)**. This convention is consistent with
  the hopping picture $c^\dagger_i c_j$ where the bond direction points from $j$ to $i$:
  `deltas[1] = coord[1] - coord[nb]`.
  For 1-site bonds `deltas` is empty; for 2-site bonds `deltas[1]` is the single
  bond vector (site 1 minus site 2); for 3- or 4-site bonds all displacements from
  the last site are provided, giving complete cluster geometry.
- `order`: `(type1, site1, type2, site2, type3, site3, type4, site4)`.
  Each site index refers to `bond.states[i]`; it can be 1, 2, 3, or 4 (the bond must
  have at least that many sites). This allows all four operators to be placed on
  any combination of sites, not just one or two.

**Return value:** a NamedTuple `(ops, irvec)`.
- `ops`: operators already reordered to InterAll format (c†c c†c) with the fermionic
  sign absorbed into the coefficient.
- `irvec`: a `Vector{NTuple{3, Vector{Float64}}}`, where each tuple `(τ1, τ2, τ3)`
  gives the three unit-cell displacements that characterize the four-site interaction
  under translational invariance: $\tau_n = \mathrm{icoord}(\mathrm{op}_n) - \mathrm{icoord}(\mathrm{op}_4)$.
  For onsite or density-density interactions $\tau_1 = \tau_3 = 0$.

**Example — onsite Hubbard $U n_\uparrow n_\downarrow$:**

```julia
hubbard = generate_twobody(dofs, onsite_bonds,
    (deltas, qn1, qn2, qn3, qn4) ->
        qn1.orbital == qn2.orbital == qn3.orbital == qn4.orbital &&
        (qn1.spin, qn2.spin, qn3.spin, qn4.spin) == (1, 1, 2, 2) ? U : 0.0,
    order = (cdag, 1, c, 1, cdag, 1, c, 1))
# hubbard is a NamedTuple:
#   hubbard.ops   → Vector{Operators} in InterAll (c†c c†c) format with fermionic sign absorbed
#   hubbard.irvec → Vector{NTuple{3,Vector{Float64}}}; each tuple (τ1,τ2,τ3) gives
#                   unit-cell displacements of operators 1,2,3 relative to operator 4.
#                   For onsite Hubbard: all τ = [0.0, 0.0] (all on same site).
```

**Example — nearest-neighbor density-density interaction along x only:**

```julia
nn_coulomb = generate_twobody(dofs, nn_bonds,
    (deltas, qn1, qn2, qn3, qn4) -> deltas[1] ≈ [1.0, 0.0] ? V : 0.0)
# For a 2-site bond: deltas = [bond.coordinates[1] - bond.coordinates[2]]
# i.e. the vector from the second site to the first — only +x bonds (deltas[1]=[1,0]) pass the filter
# nn_coulomb.ops   → Vector{Operators} for +x direction bonds only
# nn_coulomb.irvec → (τ1,τ2,τ3) where τ1=τ3=[0,0] and τ2 encodes the unit-cell shift between the two sites
```

**Example — Hund's pair hopping $J c^\dagger_{m\uparrow} c^\dagger_{m\downarrow} c_{m'\downarrow} c_{m'\uparrow}$:**

```julia
hund = generate_twobody(dofs, onsite_bonds,
    (deltas, qn1, qn2, qn3, qn4) ->
        (qn1.orbital, qn3.orbital) == (qn2.orbital, qn4.orbital) &&
        qn1.orbital != qn3.orbital &&
        (qn1.spin, qn2.spin, qn3.spin, qn4.spin) == (1, 2, 2, 1) ? J : 0.0,
    order = (cdag, 1, cdag, 1, c, 1, c, 1))
# hund.ops   → Vector{Operators} with reordering sign from c†c†cc → c†c c†c InterAll form
# hund.irvec → all ([0,0],[0,0],[0,0]) since all operators are on the same site
```

**Example — four-site ring exchange $K c^\dagger_1 c_4 c^\dagger_3 c_2$ (distinct sites):**

For interactions where all four operators sit on *different* lattice sites a 4-site bond is needed. The site indices in `order` map directly to `bond.states[i]`, so any permutation of sites 1–4 is valid:

```julia
# Build 4-site plaquette bonds (e.g., corners of a square unit cell)
plaquette_bonds = bonds(lattice, (:p, :p), [1, 2, 3])  # collect all up to 3rd-neighbor to form plaquettes
# (or construct Bond objects manually for exact 4-site clusters)

ring = generate_twobody(dofs, plaquette_bonds,
    (deltas, qn1, qn2, qn3, qn4) ->
        qn1.spin == qn2.spin == qn3.spin == qn4.spin ? K : 0.0,
    order = (cdag, 1, c, 4, cdag, 3, c, 2))
# order = (cdag,1, c,4, cdag,3, c,2) means:
#   operator 1: c†  at bond.states[1], coord = bond.coordinates[1], icoord = bond.icoordinates[1]
#   operator 2: c   at bond.states[4], coord = bond.coordinates[4], icoord = bond.icoordinates[4]
#   operator 3: c†  at bond.states[3], coord = bond.coordinates[3], icoord = bond.icoordinates[3]
#   operator 4: c   at bond.states[2], coord = bond.coordinates[2], icoord = bond.icoordinates[2]
# After InterAll reordering (c†c c†c), τ1,τ2,τ3 are computed from the
# icoordinates of the reordered operators relative to the 4th one.
# ring.ops   → Vector{Operators} with all four operators on distinct lattice sites
# ring.irvec → non-trivial (τ1,τ2,τ3) encoding the full 4-site cluster geometry
```

The `value` function mechanism makes all these variants — and arbitrary combinations of them — expressible without any modifications to the library. Physical constraints (spin conservation, orbital selection, direction filtering, multi-site geometry) are encoded entirely in user-supplied anonymous functions.

### Building matrices

Once operator lists are assembled, matrix representations are built by:

```julia
# Full one-body Hamiltonian H[i,j]
H = build_onebody_matrix(dofs, onebody.ops)

# Full two-body interaction tensor V[i,j,k,l] in InterAll format
V = build_interaction_tensor(dofs, twobody.ops)
```

For mean-field calculations the interaction is more efficiently accessed through `build_U` (real-space HF) or `build_Vr` / `build_Vk` (momentum-space HF), which bypass the $O(N^4)$ dense tensor entirely. See the respective HF solver documentation for details.

---

## Summary

| Component | Key design choices |
|-----------|-------------------|
| `SystemDofs` | Arbitrary DOF names and sizes; `constraint` prunes the state space; nested `sortrule` creates block-diagonal structure for conserved quantum numbers |
| `Lattice` / `Bond` | Any unit-cell geometry; tiling constructor handles PBC automatically; `icoordinates` carries unit-cell origin information needed for Bloch-space calculations |
| `generate_onebody` | `delta`-aware value function for direction-dependent coefficients; `hc=true` for automatic Hermitian conjugate; returns `(ops, delta, irvec)` |
| `generate_twobody` | 1–4 site bonds; `deltas` vector for full cluster geometry; arbitrary operator ordering; returns InterAll operators with `(τ1, τ2, τ3)` displacement metadata |
| `build_*` | Sparse matrix/tensor construction; specialized paths in HF solvers avoid dense $N^4$ tensors |
