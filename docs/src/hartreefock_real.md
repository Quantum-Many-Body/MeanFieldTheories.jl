# Real-Space Hartree-Fock Approximation

## Theory

### 1. Physical Hamiltonian

Consider a general lattice Hamiltonian consisting of one-body and two-body terms:

$$H = H_0 + H_{\text{int}}$$

**One-body term**:

$$H_0 = \sum_{ij}\sum_{ab} t^{ab}_{ij}\, c^\dagger_{ia} c_{jb}$$

**Two-body term**:

$$H_{\text{int}} = \sum_{ijkl}\sum_{abcd} V^{abcd}_{ijkl}\, c^\dagger_{ia} c_{jb} c^\dagger_{kc} c_{ld}$$

The interaction matrix element is defined as

$$V^{abcd}_{ijkl} = \iint d\mathbf{r}_1\,d\mathbf{r}_2\;
w^*_{ia}(\mathbf{r}_1)\,w_{jb}(\mathbf{r}_1)\,
\frac{e^2}{|\mathbf{r}_1-\mathbf{r}_2|}\,
w^*_{kc}(\mathbf{r}_2)\,w_{ld}(\mathbf{r}_2)$$

where $i, j, k, l$ are site indices and $a, b, c, d$ label all internal degrees of freedom (orbital, spin, etc.). The physical picture of the operator ordering is: $c^\dagger_{ia} c_{jb}$ contributes to the charge density at $\mathbf{r}_1$, and $c^\dagger_{kc} c_{ld}$ contributes to the charge density at $\mathbf{r}_2$.

---

### 2. Wick Decomposition

Applying Wick's theorem to each four-operator product, the two non-trivial single-contraction channels give:

$$c^\dagger_{ia} c_{jb} c^\dagger_{kc} c_{ld} \;\approx\;
\underbrace{G^{ab}_{ij}\,c^\dagger_{kc} c_{ld} + c^\dagger_{ia} c_{jb}\,G^{cd}_{kl}}_{\text{Hartree (direct)}}\;-\;
\underbrace{G^{ad}_{il}\,c^\dagger_{kc} c_{jb} + c^\dagger_{ia} c_{ld}\,G^{cb}_{kj}}_{\text{Fock (exchange)}}$$

where the **one-body Green's function** (density matrix) is defined as

$$G^{ab}_{ij} = \langle c^\dagger_{ia} c_{jb} \rangle$$

The fully contracted constant is dropped here and accounted for in the double-counting correction to the total energy (Theory §5).

The **Hartree terms** contract same-side pairs ($c^\dagger_{ia} c_{jb}$ or $c^\dagger_{kc} c_{ld}$), representing direct Coulomb repulsion. The **Fock terms** contract cross-side pairs ($c^\dagger_{ia} c_{ld}$ or $c^\dagger_{kc} c_{jb}$) and carry a minus sign from fermionic anticommutation, representing exchange interactions.

---

### 3. Effective Interaction Tensor

Substituting the Wick decomposition into $H_{\text{int}}$ and reindexing each term to isolate the canonical bilinear $c^\dagger_{ia} c_{jb}$:

| Term | After relabeling |
|:---|:---|
| Hartree: $V^{abcd}_{ijkl}\,G^{ab}_{ij}\,c^\dagger_{kc} c_{ld}$ | $c^\dagger_{ia} c_{jb} \displaystyle\sum_{kl}\sum_{cd} V^{cdab}_{klij}\,G^{cd}_{kl}$ |
| Hartree: $V^{abcd}_{ijkl}\,c^\dagger_{ia} c_{jb}\,G^{cd}_{kl}$ | $c^\dagger_{ia} c_{jb} \displaystyle\sum_{kl}\sum_{cd} V^{abcd}_{ijkl}\,G^{cd}_{kl}$ |
| Fock: $-V^{abcd}_{ijkl}\,G^{ad}_{il}\,c^\dagger_{kc} c_{jb}$ | $-c^\dagger_{ia} c_{jb} \displaystyle\sum_{kl}\sum_{cd} V^{cbad}_{kjil}\,G^{cd}_{kl}$ |
| Fock: $-V^{abcd}_{ijkl}\,c^\dagger_{ia} c_{ld}\,G^{cb}_{kj}$ | $-c^\dagger_{ia} c_{jb} \displaystyle\sum_{kl}\sum_{cd} V^{adcb}_{ilkj}\,G^{cd}_{kl}$ |

Collecting all four contributions defines the **effective interaction tensor**:

$$\boxed{U^{abcd}_{ijkl} = V^{abcd}_{ijkl} + V^{cdab}_{klij} - V^{cbad}_{kjil} - V^{adcb}_{ilkj}}$$

The mean-field interaction Hamiltonian then takes the compact form

$$H_{\text{int}}^{\text{MF}} = \sum_{ij}\sum_{ab} c^\dagger_{ia} c_{jb} \sum_{kl}\sum_{cd} U^{abcd}_{ijkl}\, G^{cd}_{kl}$$

---

### 4. Effective Single-Particle Hamiltonian

Adding the one-body term, the full mean-field Hamiltonian becomes

$$\boxed{H^{\text{eff},ab}_{ij} = t^{ab}_{ij} + \sum_{kl}\sum_{cd} U^{abcd}_{ijkl}\,G^{cd}_{kl}}$$

Using the composite index $ia \equiv (i, a)$, the effective Hamiltonian takes the matrix form

$$h^{\text{eff}}_{ia,\,jb} = T_{ia,\,jb} + \sum_{kc,\,ld} U_{ia,\,jb;\,kc,\,ld}\; G_{kc,\,ld}$$

where $U$ is a sparse $N_\text{tot}^2 \times N_\text{tot}^2$ matrix ($N_\text{tot} = N_\text{sites} \times N_\text{int}$) and $G$ is vectorized to length $N_\text{tot}^2$, so the $N_\text{tot} \times N_\text{tot}$ effective Hamiltonian is obtained by a single sparse matrix–vector product:

$$h^{\text{eff}} = T + U \cdot \mathrm{vec}(G)$$

---

### 5. Self-Consistent Field Iteration

1. Initialize $G^{ab}_{ij}$ (random Hermitian perturbation or a provided initial guess).
2. Build the effective Hamiltonian: $h^{\text{eff}} = T + U \cdot \mathrm{vec}(G)$.
3. Diagonalize: $h^{\text{eff}}\,|\psi_n\rangle = \varepsilon_n\,|\psi_n\rangle$.
4. Update $G$ from the new occupation:
   - **$T = 0$**: occupy the lowest $N_e$ states,
     $$G^{ab}_{ij} = \sum_{n=1}^{N_e} \psi^*_{n,\,ia}\,\psi_{n,\,jb}$$
   - **$T > 0$**: use Fermi-Dirac with $\mu$ fixed by $\sum_n f(\varepsilon_n - \mu) = N_e$,
     $$G^{ab}_{ij} = \sum_n f(\varepsilon_n - \mu)\,\psi^*_{n,\,ia}\,\psi_{n,\,jb}, \qquad f(\varepsilon) = \frac{1}{e^{\varepsilon/T}+1}$$
5. Mix: $G \leftarrow (1-\alpha)\,G_{\text{old}} + \alpha\,G_{\text{new}}$. If $\|\mathrm{vec}(G_{\text{new}}) - \mathrm{vec}(G_{\text{old}})\| / N_\text{tot}^2 < \epsilon$, stop; otherwise return to step 2.

**Energies at convergence**:

$$E_{\text{band}} = \begin{cases} \displaystyle\sum_{n \in \text{occ}} \varepsilon_n & T = 0 \\[8pt] \displaystyle\mu N_e - T \sum_n \ln\!\left(1 + e^{-(\varepsilon_n-\mu)/T}\right) & T > 0 \end{cases}$$

$$E_{\text{int}} = -\frac{1}{2}\,\mathrm{vec}(G)^T \cdot U \cdot \mathrm{vec}(G), \qquad E_{\text{total}} = E_{\text{band}} + E_{\text{int}}$$

The interaction energy $E_{\text{int}}$ is the double-counting correction ensuring each electron–electron interaction is counted exactly once in $E_{\text{total}}$.

---

## Code Implementation

### 1. One-Body Hamiltonian Matrix

**`build_T(dofs, ops)`** — Builds the sparse $N_\text{tot} \times N_\text{tot}$ one-body Hamiltonian matrix $T$ from one-body operators. Each operator $t^{ab}_{ij} \cdot c^\dagger_{ia} c_{jb}$ contributes one off-diagonal entry using the composite index.

### 2. Effective Interaction Tensor

**`build_U(dofs, ops, blocks)`** — Builds the sparse $N_\text{tot}^2 \times N_\text{tot}^2$ effective interaction matrix $U_{ia,\,jb;\,kc,\,ld}$ from two-body operators, applying the four-term formula

$$U^{abcd}_{ijkl} = V^{abcd}_{ijkl} + V^{cdab}_{klij} - V^{cbad}_{kjil} - V^{adcb}_{ilkj}$$

directly during construction — without ever forming the intermediate $V$ tensor (which would require $O(N_\text{tot}^4)$ storage, e.g., 1.3 TB for a $30 \times 30$ system). Each operator $V^{abcd}_{ijkl}$ contributes to four entries simultaneously:

$$U_{ia,\,jb;\,kc,\,ld} \mathrel{+}= V, \quad U_{kc,\,ld;\,ia,\,jb} \mathrel{+}= V, \quad U_{kc,\,jb;\,ia,\,ld} \mathrel{-}= V, \quad U_{ia,\,ld;\,kc,\,jb} \mathrel{-}= V$$

Set `include_fock=false` to retain only the Hartree terms (first two contributions).

**Block sparsification**: If `dofs` is constructed with `sortrule` to define symmetry blocks (e.g., spin-up / spin-down), $G$ is block-diagonal at convergence — $G^{cd}_{kl} = 0$ when $kc$ and $ld$ belong to different blocks. Since only the terms with non-zero $G^{cd}_{kl}$ contribute to $h^{\text{eff},ab}_{ij}$, it suffices to store only the $U$ entries where $(kc, ld)$ fall in the same block. For $B$ equal-sized blocks this reduces storage by $(1-1/B^2)$ — 75% for $B = 2$ spin blocks.

### 3. Self-Consistent Field Iteration

**`solve_hf(dofs, ops, block_occupations; kwargs...)`** — Public entry point. Assembles the static matrices once, then runs one or more SCF restarts and returns the lowest-energy converged result.

**Step 1 — Preprocessing.**
Before the SCF loop, `solve_hf` calls `build_T` and `build_U` once to construct the static sparse matrices $T$ and $U$. Both are reused unchanged across all restarts and iterations.

**Step 2 — Initialization.**
The Green's function $G$ is initialized block-by-block. By default, each symmetry block is filled with a small random Hermitian perturbation (entries drawn uniformly from $[-0.005,\,0.005]$, then symmetrized). A user-provided `G_init` seeds the first restart; all subsequent restarts draw fresh random initializations.

**Step 3 — Build effective Hamiltonian.**
At each SCF iteration, $h^{\text{eff}}$ is assembled via a single sparse matrix–vector product:

$$h^{\text{eff}} = T + \mathrm{reshape}(U \cdot \mathrm{vec}(G),\; N \times N)$$

**Step 4 — Block diagonalization.**
$h^{\text{eff}}$ is diagonalized block by block. Since $G$ is block-diagonal by symmetry, $h^{\text{eff}}$ inherits the same block structure, and each $d_b \times d_b$ sub-block is solved independently by full dense diagonalization.

**Step 5 — Update $G$.**
The new Green's function $G_{\text{new}}$ is constructed from the eigenstates weighted by occupation numbers $f_n$:

$$G^{ab}_{ij} = \sum_n f_n\,\psi^*_{n,\,ia}\,\psi_{n,\,jb}$$

- **$T = 0$**: $f_n = 1$ for the lowest $N_e$ states, zero otherwise. The chemical potential $\mu$ is set to the midpoint of the HOMO-LUMO gap.
- **$T > 0$**: $f_n = [e^{(\varepsilon_n-\mu)/T}+1]^{-1}$ (Fermi-Dirac). The chemical potential $\mu$ is determined independently per block by enforcing $\sum_n f_n = N_e^{(b)}$, using bisection (robust, guaranteed convergence when the root is bracketed) followed by Newton's method as a fallback. An overflow guard sets $f_n = 0$ when $(\varepsilon_n - \mu)/T > \varepsilon_{\text{cut}}$ (default 100) to avoid numerical overflow at low temperatures.

**Step 6 — Convergence check.**
The normalized Frobenius residual

$$r = \frac{\|\mathrm{vec}(G_{\text{new}}) - \mathrm{vec}(G_{\text{old}})\|}{B \cdot N_{\text{site}}^2}$$

is compared against `tol` (default $10^{-6}$). If $r < \varepsilon_{\text{tol}}$, the SCF cycle is declared converged.

**Step 7 — Mixing.**
When convergence is not yet reached, $G$ is updated by blending $G_{\text{old}}$ and $G_{\text{new}}$. Two strategies are available:

- **Linear mixing**: $G \leftarrow (1-\alpha_{\text{eff}})\,G_{\text{old}} + \alpha_{\text{eff}}\,G_{\text{new}}$. The effective mixing parameter $\alpha_{\text{eff}}$ is scheduled conservatively at early iterations ($0.3\alpha$ for iter $\leq 5$, $0.6\alpha$ for iter $\leq 15$, full $\alpha$ thereafter) to suppress oscillations near the start.
- **DIIS acceleration** (Pulay, default `diis_m = 8`): Once at least two iterates are stored, the optimal $G$ is extrapolated from the last `diis_m` iterates by minimizing the residual norm in their span, solving a small $(m+1) \times (m+1)$ linear system. Falls back to linear mixing when the DIIS matrix is (near-)singular. Disable with `diis_m = 0`.

**Step 8 — Multiple restarts.**
When `n_restarts > 1`, the full SCF procedure is repeated from independent random initializations. The result with the lowest total energy among all converged runs is returned; if none converge, the lowest-energy unconverged run is returned instead.

**Output.**
Returns a `NamedTuple` with fields:
- `G`: converged density matrix ($N \times N$)
- `eigenvalues`, `eigenvectors`: per-block arrays from the last diagonalization
- `energies`: `(band, interaction, total)` — see Theory §5 for energy formulas
- `mu_list`: chemical potential per symmetry block
- `converged`, `iterations`, `residual`: SCF diagnostics
- `ncond`: total particle number $\mathrm{Tr}(G)$
- `sz`: total $S_z$ (if `dofs` contains a `:spin` Dof with size 2, otherwise `nothing`)
