# Momentum-Space Hartree-Fock: Derivation

## Step 1: General Two-Body Hamiltonian

The most general two-body interaction Hamiltonian in InterAll format is:

$$H_{\text{int}} = \frac{1}{2}\sum_{ijkl}\sum_{abcd} V_{ijkl}^{abcd} \, c^\dagger_{ia} c_{jb} c^\dagger_{kc} c_{ld}$$

Here $i,j,k,l$ are lattice site indices, and $a,b,c,d$ are **intra-site degrees of freedom** (a collective label for spin, orbital, or any other internal quantum numbers).

---

## Step 2: Reduction via Translational Invariance

### Vertex Locality

For physical Coulomb-type interactions, charge is conserved at each interaction vertex — the interaction does not directly mediate inter-site hopping. This means each $c^\dagger c$ pair acts on the **same site**:

$$j = i, \qquad l = k$$

The first operator pair $c^\dagger_{ia} c_{ib}$ sits at site $i$; the second $c^\dagger_{kc} c_{kd}$ sits at site $k$. This encompasses all common interaction types:

| Interaction | Orbital structure | Site structure |
|-------------|-------------------|----------------|
| Hubbard $U n_{a\uparrow}n_{a\downarrow}$ | $a=b=c=d$ | $i=k$ (on-site) |
| Inter-site Coulomb $V n_i n_j$ | $a=b$, $c=d$ | $i \neq k$ |
| Hund, Pair-Lift, Exchange | general $a,b,c,d$ | $i=k$ or $i \neq k$ |

Note that $a,b,c,d$ are **four independent internal indices** — the structure can be $aaaa$, $aabb$, $abab$, or fully general $abcd$ depending on the interaction type.

### Translational Invariance

For a translationally invariant system, $V_{ii,kk}^{abcd}$ (after imposing vertex locality $j=i$, $l=k$) depends only on the **relative displacement** $\mathbf{r} = \mathbf{R}_k - \mathbf{R}_i$:

$$V_{ii,kk}^{abcd} = W^{abcd}(\mathbf{R}_k - \mathbf{R}_i) \equiv W^{abcd}(\mathbf{r})$$

Renaming $k \to j$ for clarity, the reduced Hamiltonian is:

$$\boxed{H_{\text{int}} = \frac{1}{2}\sum_{i,j}\sum_{abcd} W^{abcd}(\mathbf{R}_j - \mathbf{R}_i) \, c^\dagger_{ia} c_{ib} c^\dagger_{jc} c_{jd}}$$

where $W^{abcd}(\mathbf{r})$ depends only on the inter-site distance. For $i=j$, $W^{abcd}(\mathbf{0})$ reduces to the local multi-orbital interaction matrix.

---

## Step 3: Fourier Transform to Momentum Space

For a lattice with $N$ sites and periodic boundary conditions, define Bloch creation/annihilation operators:

$$c^\dagger_{ia} = \frac{1}{\sqrt{N}}\sum_{\mathbf{k}} e^{-i\mathbf{k}\cdot\mathbf{R}_i} c^\dagger_{\mathbf{k}a}, \qquad c_{ia} = \frac{1}{\sqrt{N}}\sum_{\mathbf{k}} e^{+i\mathbf{k}\cdot\mathbf{R}_i} c_{\mathbf{k}a}$$

Substituting all four operators into $H_{\text{int}}$ and introducing momentum labels $\mathbf{k}_1,\mathbf{k}_2,\mathbf{k}_3,\mathbf{k}_4$:

$$H_{\text{int}} = \frac{1}{2N^2}\sum_{ij,abcd}\sum_{\mathbf{k}_1\mathbf{k}_2\mathbf{k}_3\mathbf{k}_4} W^{abcd}(\mathbf{R}_j - \mathbf{R}_i)\, e^{-i\mathbf{k}_1\cdot\mathbf{R}_i} e^{+i\mathbf{k}_2\cdot\mathbf{R}_i} e^{-i\mathbf{k}_3\cdot\mathbf{R}_j} e^{+i\mathbf{k}_4\cdot\mathbf{R}_j} \, c^\dagger_{\mathbf{k}_1 a} c_{\mathbf{k}_2 b} c^\dagger_{\mathbf{k}_3 c} c_{\mathbf{k}_4 d}$$

**Summing over site $i$**, with $\mathbf{r} = \mathbf{R}_j - \mathbf{R}_i$:

$$\sum_i e^{i(-\mathbf{k}_1+\mathbf{k}_2-\mathbf{k}_3+\mathbf{k}_4)\cdot\mathbf{R}_i} = N\,\delta_{\mathbf{k}_1-\mathbf{k}_2+\mathbf{k}_3-\mathbf{k}_4,\,\mathbf{0}}$$

This enforces **momentum conservation**:

$$\mathbf{k}_1 - \mathbf{k}_2 + \mathbf{k}_3 - \mathbf{k}_4 = \mathbf{0} \pmod{\text{reciprocal lattice vector}}$$

**Summing over the relative displacement $\mathbf{r}$** defines the Fourier transform of the interaction:

$$\tilde{W}^{abcd}(\mathbf{q}) \equiv \sum_\mathbf{r} W^{abcd}(\mathbf{r})\,e^{i\mathbf{q}\cdot\mathbf{r}}$$

The overall prefactor becomes $\frac{1}{2N^2} \cdot N = \frac{1}{2N}$.

### Introducing the Momentum Transfer

Define the momentum transfer $\mathbf{q} = \mathbf{k}_1 - \mathbf{k}_2 = \mathbf{k}_4 - \mathbf{k}_3$ (momentum conservation guarantees equality). Taking $\mathbf{k}_2$ and $\mathbf{k}_4$ as the two independent free momenta:

$$\mathbf{k}_1 = \mathbf{k}_2 + \mathbf{q}, \qquad \mathbf{k}_3 = \mathbf{k}_4 - \mathbf{q}$$

The interaction Hamiltonian in momentum space becomes:

$$\boxed{H_{\text{int}} = \frac{1}{2N}\sum_{abcd}\sum_{\mathbf{k}_2,\mathbf{k}_4,\mathbf{q}} \tilde{W}^{abcd}(\mathbf{q})\, c^\dagger_{\mathbf{k}_2+\mathbf{q},a}\; c_{\mathbf{k}_2,b}\; c^\dagger_{\mathbf{k}_4-\mathbf{q},c}\; c_{\mathbf{k}_4,d}}$$

The four operators carry momenta $\mathbf{k}_2+\mathbf{q}$, $\mathbf{k}_2$, $\mathbf{k}_4-\mathbf{q}$, $\mathbf{k}_4$ respectively. The Hamiltonian still depends on **three independent momenta** and cannot yet be cast as a single-particle problem.

---

## Step 4: Hartree-Fock Mean-Field Decoupling

### Wick's Theorem

Applying Wick's theorem to the four-operator product (dropping the constant fully-contracted term):

$$c^\dagger_{\mathbf{k}_2+\mathbf{q},a}\; c_{\mathbf{k}_2,b}\; c^\dagger_{\mathbf{k}_4-\mathbf{q},c}\; c_{\mathbf{k}_4,d}$$
$$\approx \underbrace{\langle c^\dagger_{\mathbf{k}_2+\mathbf{q},a} c_{\mathbf{k}_2,b}\rangle}_{\mathcal{G}_1} c^\dagger_{\mathbf{k}_4-\mathbf{q},c} c_{\mathbf{k}_4,d}
  + c^\dagger_{\mathbf{k}_2+\mathbf{q},a} c_{\mathbf{k}_2,b}\underbrace{\langle c^\dagger_{\mathbf{k}_4-\mathbf{q},c} c_{\mathbf{k}_4,d}\rangle}_{\mathcal{G}_2}$$
$$- \underbrace{\langle c^\dagger_{\mathbf{k}_2+\mathbf{q},a} c_{\mathbf{k}_4,d}\rangle}_{\mathcal{G}_3} c^\dagger_{\mathbf{k}_4-\mathbf{q},c} c_{\mathbf{k}_2,b}
  - c^\dagger_{\mathbf{k}_2+\mathbf{q},a} c_{\mathbf{k}_4,d}\underbrace{\langle c^\dagger_{\mathbf{k}_4-\mathbf{q},c} c_{\mathbf{k}_2,b}\rangle}_{\mathcal{G}_4}$$

The first two terms (direct contractions) give the **Hartree** contributions; the last two (cross contractions) give the **Fock** contributions, with minus signs from fermionic statistics.

### Translational Invariance of the Green's Function

For a ground state with discrete translational symmetry, the single-particle Green's function is **diagonal in $\mathbf{k}$**:

$$\langle c^\dagger_{\mathbf{k},a} c_{\mathbf{k}',b}\rangle = G_{ab}(\mathbf{k})\,\delta_{\mathbf{k},\mathbf{k}'}$$

States at different momenta are uncorrelated, a direct consequence of the commutation of the translation operator with the Hamiltonian.

**Important note on symmetry breaking**: Although the Hamiltonian has translational symmetry, the HF ground state may spontaneously break this symmetry (e.g., antiferromagnetic order, charge density waves). In such cases, one must redefine the **unit cell** to the enlarged magnetic/modulated cell that restores periodicity. From this point forward, $\mathbf{k}$ refers to momenta in the reciprocal lattice of this (possibly enlarged) cell, and indices $a,b,c,d$ label all internal degrees of freedom within the chosen cell (dimension $d \to nd$ for an $n$-fold enlargement, with correspondingly fewer $\mathbf{k}$ points).

Substituting into each contraction:

$$\mathcal{G}_1 = G_{ab}(\mathbf{k}_2)\,\delta_{\mathbf{q},\mathbf{0}}, \qquad \mathcal{G}_2 = G_{cd}(\mathbf{k}_4)\,\delta_{\mathbf{q},\mathbf{0}}$$

$$\mathcal{G}_3 = G_{ad}(\mathbf{k}_4)\,\delta_{\mathbf{k}_2+\mathbf{q},\mathbf{k}_4}, \qquad \mathcal{G}_4 = G_{cb}(\mathbf{k}_2)\,\delta_{\mathbf{k}_4-\mathbf{q},\mathbf{k}_2}$$

---

## Step 5: Hartree Self-Energy — Only $\mathbf{q}=0$ Contributes

The factors $\delta_{\mathbf{q},\mathbf{0}}$ in $\mathcal{G}_1$ and $\mathcal{G}_2$ show that **all $\mathbf{q} \neq 0$ Hartree contributions vanish**.

**From $\mathcal{G}_1$** (setting $\mathbf{q}=0$; remaining operator: $c^\dagger_{\mathbf{k}_4,c}c_{\mathbf{k}_4,d}$):

$$H^{H1} = \frac{1}{2N}\sum_{abcd}\sum_{\mathbf{k}_2,\mathbf{k}_4} \tilde{W}^{abcd}(0)\, G_{ab}(\mathbf{k}_2)\; c^\dagger_{\mathbf{k}_4,c} c_{\mathbf{k}_4,d}$$

Summing over $\mathbf{k}_2$ yields the **on-site density matrix**:

$$G_{ab}^{(0)} \equiv G_{ab}(\mathbf{r}=0) = \frac{1}{N}\sum_{\mathbf{k}_2} G_{ab}(\mathbf{k}_2)$$

so:

$$H^{H1} = \frac{1}{2}\sum_{\mathbf{k}}\sum_{cd} \left(\sum_{ab}\tilde{W}^{abcd}(0)\,G_{ab}^{(0)}\right) c^\dagger_{\mathbf{k},c} c_{\mathbf{k},d}$$

**From $\mathcal{G}_2$** (setting $\mathbf{q}=0$; remaining operator: $c^\dagger_{\mathbf{k}_2,a}c_{\mathbf{k}_2,b}$):

$$H^{H2} = \frac{1}{2}\sum_{\mathbf{k}}\sum_{ab} \left(\sum_{cd}\tilde{W}^{abcd}(0)\,G_{cd}^{(0)}\right) c^\dagger_{\mathbf{k},a} c_{\mathbf{k},b}$$

Both contributions are $\mathbf{k}$-independent. Collecting into the $c^\dagger_{\mathbf{k},a}c_{\mathbf{k},b}$ channel, the **Hartree self-energy matrix** is:

$$\boxed{\Sigma^H_{ab} = \frac{1}{2}\sum_{cd}\left[\tilde{W}^{ab,cd}(0) + \tilde{W}^{cd,ab}(0)\right] G_{cd}^{(0)}}$$

**Physical picture**: $\Sigma^H$ is a constant matrix, identical for all $\mathbf{k}$ points, determined entirely by the local density matrix $G^{(0)}$. Here $\tilde{W}^{abcd}(0) = \sum_\mathbf{r} W^{abcd}(\mathbf{r})$ is the interaction summed over all inter-site distances.

---

## Step 6: Fock Self-Energy — Convolution in $\mathbf{k}$ Space

The constraint $\delta_{\mathbf{k}_2+\mathbf{q},\mathbf{k}_4}$ from $\mathcal{G}_3$ fixes $\mathbf{k}_4 = \mathbf{k}_2+\mathbf{q}$, so $\mathbf{k}_4-\mathbf{q} = \mathbf{k}_2$: Op 3 and Op 2 acquire the same momentum, and the remaining operator product becomes $-c^\dagger_{\mathbf{k}_2,c}\,c_{\mathbf{k}_2,b}$ (the minus sign from the Fock cross-contraction).

**From $\mathcal{G}_3$**:

$$H^{F1} = -\frac{1}{2N}\sum_{abcd}\sum_{\mathbf{k}_2,\mathbf{q}} \tilde{W}^{abcd}(\mathbf{q})\,G_{ad}(\mathbf{k}_2+\mathbf{q})\; c^\dagger_{\mathbf{k}_2,c}\, c_{\mathbf{k}_2,b}$$

**From $\mathcal{G}_4$** (same constraint $\mathbf{k}_4 = \mathbf{k}_2+\mathbf{q}$; remaining operator: $-c^\dagger_{\mathbf{k}_2+\mathbf{q},a}\,c_{\mathbf{k}_2+\mathbf{q},d}$; relabeling $\mathbf{k}_2+\mathbf{q} \to \mathbf{k}$):

$$H^{F2} = -\frac{1}{2N}\sum_{abcd}\sum_{\mathbf{k},\mathbf{q}} \tilde{W}^{abcd}(-\mathbf{q})\,G_{cb}(\mathbf{k}-\mathbf{q})\; c^\dagger_{\mathbf{k},a}\, c_{\mathbf{k},d}$$

Collecting both Fock terms into the $c^\dagger_{\mathbf{k},a}c_{\mathbf{k},b}$ channel:

**From $H^{F1}$**: the remaining operator is $c^\dagger_{\mathbf{k},c}\,c_{\mathbf{k},b}$, so the channel indices are $a\leftarrow c$ (3rd position in $\tilde{W}$) and $b\leftarrow b$ (2nd position), while $a,d$ are summed via $G_{ad}$. Relabelling the internal summation $a\to c,\,d\to d$:

$$\Sigma^{F1}_{ab}(\mathbf{k}) = -\frac{1}{2N}\sum_\mathbf{q}\sum_{cd}\tilde{W}^{cbad}(\mathbf{q})\,G_{cd}(\mathbf{k}+\mathbf{q})$$

**From $H^{F2}$**: the remaining operator is $c^\dagger_{\mathbf{k},a}\,c_{\mathbf{k},d}$, so $a\leftarrow a$ (1st position) and $b\leftarrow d$ (4th position), while $b,c$ are summed via $G_{cb}$. Substituting $\mathbf{q}\to-\mathbf{q}$ to bring $G$ to $G(\mathbf{k}+\mathbf{q})$ form and swapping the dummy pair $b\leftrightarrow c$:

$$\Sigma^{F2}_{ab}(\mathbf{k}) = -\frac{1}{2N}\sum_\mathbf{q}\sum_{cd}\tilde{W}^{adcb}(\mathbf{q})\,G_{cd}(\mathbf{k}+\mathbf{q})$$

The **Fock self-energy** is:

$$\boxed{\Sigma^F_{ab}(\mathbf{k}) = -\frac{1}{N}\sum_\mathbf{q}\sum_{cd}\,\frac{1}{2}\!\left[\tilde{W}^{cbad}(\mathbf{q}) + \tilde{W}^{adcb}(\mathbf{q})\right] G_{cd}(\mathbf{k}+\mathbf{q})}$$

This is the exact analogue of the Hartree formula (Step 5): two symmetry-related permutations of $\tilde{W}$ appear, one from each Wick contraction.

### Convolution Structure and FFT Acceleration

The Fock self-energy is a **convolution in $\mathbf{k}$ space**: $\Sigma^F(\mathbf{k}) \propto \sum_\mathbf{q} \tilde{W}(\mathbf{q})\,G(\mathbf{k}+\mathbf{q})$. By the convolution theorem:

$$\text{convolution in } \mathbf{k}\text{-space} \;\Longleftrightarrow\; \text{pointwise product in } \mathbf{r}\text{-space}$$

$$[\Sigma^F(\mathbf{k})]_{ab} = -\mathcal{F}_{\mathbf{r}\to\mathbf{k}}\!\left[\sum_{cd}\frac{1}{2}\!\left[W^{cbad}(\mathbf{r})+W^{adcb}(\mathbf{r})\right] G_{cd}(\mathbf{r})\right]$$

Concretely:
1. Compute the combined interaction kernel $\frac{1}{2}[W^{cbad}(\mathbf{r})+W^{adcb}(\mathbf{r})]$ and contract with $G_{cd}(\mathbf{r})$ pointwise in real space (summing over $c,d$)
2. Apply a single FFT to obtain $[\Sigma^F(\mathbf{k})]_{ab}$

This reduces the naive $O(N^2)$ convolution to $O(N\log N)$.

**Physical picture**: $\Sigma^F(\mathbf{k})$ depends on the full real-space Green's function $G(\mathbf{r})$ and captures non-local quantum exchange effects.

---

## Step 7: Effective Single-Particle Hamiltonian

Combining the one-body term (block-diagonal in $\mathbf{k}$ after Fourier transform) with the mean-field interaction, the effective Hamiltonian at each $\mathbf{k}$ point is:

$$\boxed{H^{\text{eff}}(\mathbf{k}) = T(\mathbf{k}) + \Sigma^H + \Sigma^F(\mathbf{k})}$$

| Term | Expression | $\mathbf{k}$-dependence | Physical meaning |
|------|-----------|-------------------------|------------------|
| $T(\mathbf{k})$ | $\sum_\mathbf{r} T^{ab}(\mathbf{r})\,e^{i\mathbf{k}\cdot\mathbf{r}}$ | $\mathbf{k}$-dependent | Single-particle hopping (band structure) |
| $\Sigma^H$ | $\sum_{cd}\bar{W}^{ab,cd}(0)\,G_{cd}^{(0)}$ | **$\mathbf{k}$-independent** | Hartree: mean-field Coulomb repulsion |
| $\Sigma^F(\mathbf{k})$ | $-\mathcal{F}[W(\mathbf{r})\cdot G(\mathbf{r})](\mathbf{k})$ | $\mathbf{k}$-dependent | Fock: quantum exchange interaction |

The original four-body interaction depending on three independent momenta has been reduced, via HF decoupling and translational symmetry, to $N$ independent $d\times d$ eigenvalue problems (where $d$ is the number of intra-site degrees of freedom). This is the fundamental efficiency gain of momentum-space Hartree-Fock.

---

## Self-Consistent Field (SCF) Iteration

1. **Initialize** $G_{ab}(\mathbf{k})$ (random, zero, or from a previous calculation)
2. **Build** $H^{\text{eff}}(\mathbf{k}) = T(\mathbf{k}) + \Sigma^H[G^{(0)}] + \Sigma^F[G(\mathbf{r})]$
3. **Diagonalize** $H^{\text{eff}}(\mathbf{k})$ at each $\mathbf{k}$ point: $H^{\text{eff}}(\mathbf{k})\,v_{n\mathbf{k}} = \varepsilon_{n\mathbf{k}}\,v_{n\mathbf{k}}$
4. **Determine occupation** $f_{n\mathbf{k}}$: step function at $T=0$, or Fermi-Dirac at $T>0$
5. **Update Green's function**: $G_{ab}(\mathbf{k}) = \sum_n v^*_{an}(\mathbf{k})\,f_{n\mathbf{k}}\,v_{bn}(\mathbf{k})$, then FFT to real space
6. **Check convergence**: $\|G_{\text{new}} - G_{\text{old}}\| < \varepsilon$
7. **Mix**: $G \leftarrow (1-\alpha)G_{\text{old}} + \alpha G_{\text{new}}$, go to step 2
