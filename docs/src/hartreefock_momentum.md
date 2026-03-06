# Momentum-Space Hartree-Fock Approximation

## Theory

### 1. Real-Space Starting Point

Consider the Coulomb interaction Hamiltonian projected onto a Wannier basis, written in **creation-annihilation alternating order**:

$$H_{\text{int}} = \frac{1}{2}\sum_{ijkl}\sum_{abcd} V^{abcd}_{ijkl}\, c^\dagger_{ia}\,c_{jb}\,c^\dagger_{kc}\,c_{ld}$$

where $i,j,k,l$ are site indices (each summed independently) and $a,b,c,d$ label all internal degrees of freedom (orbital, spin, etc.). The interaction matrix element is defined as

$$V^{abcd}_{ijkl} = \iint d\mathbf{r}_1\,d\mathbf{r}_2\;
w^*_{ia}(\mathbf{r}_1)\,w_{jb}(\mathbf{r}_1)\,
\frac{e^2}{|\mathbf{r}_1-\mathbf{r}_2|}\,
w^*_{kc}(\mathbf{r}_2)\,w_{ld}(\mathbf{r}_2)$$

where the Wannier functions satisfy $w_{i\alpha}(\mathbf{r}) = w_\alpha(\mathbf{r}-\mathbf{R}_i)$. The physical picture of the operator ordering is: $c^\dagger_{ia}c_{jb}$ contributes to the charge density at $\mathbf{r}_1$, and $c^\dagger_{kc}c_{ld}$ contributes to the charge density at $\mathbf{r}_2$.

---

### 2. Fourier Transform to $k$-Space

#### 2.1 Fourier Transform Convention

$$c_{i\alpha} = \frac{1}{\sqrt{N}}\sum_{\mathbf{k}} e^{i\mathbf{k}\cdot\mathbf{R}_i}\,c_{\mathbf{k}\alpha}, \qquad
c^\dagger_{i\alpha} = \frac{1}{\sqrt{N}}\sum_{\mathbf{k}} e^{-i\mathbf{k}\cdot\mathbf{R}_i}\,c^\dagger_{\mathbf{k}\alpha}$$

#### 2.2 Translational Invariance

Substituting $w_{i\alpha}(\mathbf{r}) = w_\alpha(\mathbf{r}-\mathbf{R}_i)$ into the matrix element gives

$$V^{abcd}_{ijkl} = \iint d\mathbf{r}_1\,d\mathbf{r}_2\;
w^*_a(\mathbf{r}_1-\mathbf{R}_i)\,w_b(\mathbf{r}_1-\mathbf{R}_j)\,
\frac{e^2}{|\mathbf{r}_1-\mathbf{r}_2|}\,
w^*_c(\mathbf{r}_2-\mathbf{R}_k)\,w_d(\mathbf{r}_2-\mathbf{R}_l)$$

Taking $\mathbf{R}_l$ as the reference site, perform the change of integration variables $\mathbf{r}_1 \to \mathbf{r}_1 + \mathbf{R}_l$, $\mathbf{r}_2 \to \mathbf{r}_2 + \mathbf{R}_l$. The Coulomb kernel $|\mathbf{r}_1 - \mathbf{r}_2|^{-1}$ is invariant under simultaneous translation, and the Jacobian is unity, so

$$V^{abcd}_{ijkl} = \iint d\mathbf{r}_1\,d\mathbf{r}_2\;
w^*_a\!\left(\mathbf{r}_1-(\mathbf{R}_i-\mathbf{R}_l)\right)\,
w_b\!\left(\mathbf{r}_1-(\mathbf{R}_j-\mathbf{R}_l)\right)\,
\frac{e^2}{|\mathbf{r}_1-\mathbf{r}_2|}\,
w^*_c\!\left(\mathbf{r}_2-(\mathbf{R}_k-\mathbf{R}_l)\right)\,
w_d(\mathbf{r}_2)$$

The result depends on $i,j,k,l$ only through the three relative displacements

$$\boldsymbol{\tau}_1 = \mathbf{R}_i - \mathbf{R}_l, \quad
\boldsymbol{\tau}_2 = \mathbf{R}_j - \mathbf{R}_l, \quad
\boldsymbol{\tau}_3 = \mathbf{R}_k - \mathbf{R}_l$$

so we define

$$\bar{V}^{abcd}(\boldsymbol{\tau}_1,\boldsymbol{\tau}_2,\boldsymbol{\tau}_3)
\equiv \iint d\mathbf{r}_1\,d\mathbf{r}_2\;
w^*_a(\mathbf{r}_1-\boldsymbol{\tau}_1)\,w_b(\mathbf{r}_1-\boldsymbol{\tau}_2)\,
\frac{e^2}{|\mathbf{r}_1-\mathbf{r}_2|}\,
w^*_c(\mathbf{r}_2-\boldsymbol{\tau}_3)\,w_d(\mathbf{r}_2)
= V^{abcd}_{ijkl}$$

The Hamiltonian becomes

$$H_{\text{int}} = \frac{1}{2}\sum_l\sum_{\boldsymbol{\tau}_1\boldsymbol{\tau}_2\boldsymbol{\tau}_3}\sum_{abcd}
\bar{V}^{abcd}(\boldsymbol{\tau}_1,\boldsymbol{\tau}_2,\boldsymbol{\tau}_3)\,
c^\dagger_{l+\tau_1,\,a}\,c_{l+\tau_2,\,b}\,c^\dagger_{l+\tau_3,\,c}\,c_{l,d}$$

#### 2.3 Substituting the Fourier Expansion

Expanding each of the four operators, the four phase factors are

$$e^{-i\mathbf{k}_1\cdot\mathbf{R}_i}\cdot e^{+i\mathbf{k}_2\cdot\mathbf{R}_j}\cdot e^{-i\mathbf{k}_3\cdot\mathbf{R}_k}\cdot e^{+i\mathbf{k}_4\cdot\mathbf{R}_l}$$

Rewriting in relative coordinates and separating out the part depending on the reference site $\mathbf{R}_l$:

$$= e^{-i\mathbf{k}_1\cdot\boldsymbol{\tau}_1 +i\mathbf{k}_2\cdot\boldsymbol{\tau}_2 -i\mathbf{k}_3\cdot\boldsymbol{\tau}_3}
\cdot e^{\,i(-\mathbf{k}_1+\mathbf{k}_2-\mathbf{k}_3+\mathbf{k}_4)\cdot\mathbf{R}_l}$$

Summing over all $\mathbf{R}_l$ imposes **momentum conservation**:

$$\sum_l e^{\,i(-\mathbf{k}_1+\mathbf{k}_2-\mathbf{k}_3+\mathbf{k}_4)\cdot\mathbf{R}_l}
= N\,\delta_{\mathbf{k}_1+\mathbf{k}_3,\,\mathbf{k}_2+\mathbf{k}_4}$$

#### 2.4 Three-Momentum Interaction Kernel

Summing over the three relative displacements defines the **three-momentum Fourier transform**:

$$\widetilde{V}^{abcd}(\mathbf{k}_1,\mathbf{k}_2,\mathbf{k}_3)
\equiv \sum_{\boldsymbol{\tau}_1\boldsymbol{\tau}_2\boldsymbol{\tau}_3}
\bar{V}^{abcd}(\boldsymbol{\tau}_1,\boldsymbol{\tau}_2,\boldsymbol{\tau}_3)\,
e^{-i\mathbf{k}_1\cdot\boldsymbol{\tau}_1 +i\mathbf{k}_2\cdot\boldsymbol{\tau}_2 -i\mathbf{k}_3\cdot\boldsymbol{\tau}_3}$$

The overall prefactor becomes $\frac{1}{2N^2}\cdot N = \frac{1}{2N}$, giving the $k$-space Hamiltonian:

$$\boxed{H_{\text{int}} = \frac{1}{2N}\sum_{\mathbf{k}_1\mathbf{k}_2\mathbf{k}_3}\sum_{abcd}
\widetilde{V}^{abcd}(\mathbf{k}_1,\mathbf{k}_2,\mathbf{k}_3)\,
c^\dagger_{\mathbf{k}_1 a}\,c_{\mathbf{k}_2 b}\,c^\dagger_{\mathbf{k}_3 c}\,c_{\mathbf{k}_4 d}}$$

where $\mathbf{k}_4 = \mathbf{k}_1 + \mathbf{k}_3 - \mathbf{k}_2$ is fixed by momentum conservation, leaving only three independent momenta.

**Index correspondence** in $\widetilde{V}^{abcd}$:

- Index $a$: $c^\dagger_{\mathbf{k}_1 a}$ (site $i$, displacement $\boldsymbol{\tau}_1$, negative phase)
- Index $b$: $c_{\mathbf{k}_2 b}$ (site $j$, displacement $\boldsymbol{\tau}_2$, positive phase)
- Index $c$: $c^\dagger_{\mathbf{k}_3 c}$ (site $k$, displacement $\boldsymbol{\tau}_3$, negative phase)
- Index $d$: $c_{\mathbf{k}_4 d}$ (reference site $l$, no phase)

---

### 3. Hartree-Fock Decoupling

#### 3.1 One-Body Green's Function (Density Matrix)

For a ground state that preserves discrete translational symmetry, the single-particle one-body Green's function (density matrix) is diagonal in momentum space:

$$\langle c^\dagger_{\mathbf{k}\alpha}\,c_{\mathbf{k}'\beta}\rangle
= \delta_{\mathbf{k},\mathbf{k}'}\,G^{\alpha\beta}(\mathbf{k})$$

States at different $\mathbf{k}$ points are uncorrelated. If translational symmetry is spontaneously broken (e.g., antiferromagnetic order, charge density wave), the magnetic unit cell must be adopted as the new unit cell and $\mathbf{k}$ redefined in the corresponding Brillouin zone.

#### 3.2 Wick Decomposition

Applying Wick's theorem to the four-operator product (dropping fully contracted constants), there are four single-contraction channels:

$$c^\dagger_{\mathbf{k}_1 a}\,c_{\mathbf{k}_2 b}\,c^\dagger_{\mathbf{k}_3 c}\,c_{\mathbf{k}_4 d}
\approx
\underbrace{
+\langle c^\dagger_{\mathbf{k}_1 a} c_{\mathbf{k}_2 b}\rangle\, c^\dagger_{\mathbf{k}_3 c} c_{\mathbf{k}_4 d}
+\langle c^\dagger_{\mathbf{k}_3 c} c_{\mathbf{k}_4 d}\rangle\, c^\dagger_{\mathbf{k}_1 a} c_{\mathbf{k}_2 b}
}_{\text{Hartree (direct)}}
\underbrace{
-\langle c^\dagger_{\mathbf{k}_1 a} c_{\mathbf{k}_4 d}\rangle\, c^\dagger_{\mathbf{k}_3 c} c_{\mathbf{k}_2 b}
-\langle c^\dagger_{\mathbf{k}_3 c} c_{\mathbf{k}_2 b}\rangle\, c^\dagger_{\mathbf{k}_1 a} c_{\mathbf{k}_4 d}
}_{\text{Fock (exchange)}}$$

The Hartree terms contract "same-side" operator pairs ($\mathbf{r}_1$ side: $1,2$; $\mathbf{r}_2$ side: $3,4$), while the Fock terms contract "cross-side" pairs, picking up a minus sign from fermionic anticommutation.

#### 3.3 Momentum Constraints

Using $\langle c^\dagger_{\mathbf{k}\alpha}c_{\mathbf{k}'\beta}\rangle = \delta_{\mathbf{k},\mathbf{k}'}G^{\alpha\beta}(\mathbf{k})$, each contraction under the momentum conservation constraint $\mathbf{k}_1+\mathbf{k}_3 = \mathbf{k}_2+\mathbf{k}_4$ yields the following:

| Contracted pair | Channel | Constraint | Remaining bilinear |
|:------:|:--:|:--------:|:----------:|
| $(c^\dagger_{\mathbf{k}_1 a},\,c_{\mathbf{k}_2 b})$ | Hartree | $\mathbf{k}_1=\mathbf{k}_2=\mathbf{k}$, $\mathbf{k}_3=\mathbf{k}_4=\mathbf{q}$ | $c^\dagger_{\mathbf{q}c}c_{\mathbf{q}d}$ |
| $(c^\dagger_{\mathbf{k}_3 c},\,c_{\mathbf{k}_4 d})$ | Hartree | $\mathbf{k}_3=\mathbf{k}_4=\mathbf{k}$, $\mathbf{k}_1=\mathbf{k}_2=\mathbf{q}$ | $c^\dagger_{\mathbf{q}a}c_{\mathbf{q}b}$ |
| $(c^\dagger_{\mathbf{k}_1 a},\,c_{\mathbf{k}_4 d})$ | Fock | $\mathbf{k}_1=\mathbf{k}_4=\mathbf{k}$, $\mathbf{k}_2=\mathbf{k}_3=\mathbf{q}$ | $-c^\dagger_{\mathbf{q}c}c_{\mathbf{q}b}$ |
| $(c^\dagger_{\mathbf{k}_3 c},\,c_{\mathbf{k}_2 b})$ | Fock | $\mathbf{k}_2=\mathbf{k}_3=\mathbf{k}$, $\mathbf{k}_1=\mathbf{k}_4=\mathbf{q}$ | $-c^\dagger_{\mathbf{q}a}c_{\mathbf{q}d}$ |

Each contraction reduces the four-operator product to a bilinear $c^\dagger_{\mathbf{q}\alpha}c_{\mathbf{q}\beta}$ at momentum $\mathbf{q}$, reflecting the $k$-diagonal structure of the effective Hamiltonian in a translationally symmetric ground state.

---

### 4. Hartree-Fock Self-Energy

Substituting all contractions back into $H_{\text{int}}$, the mean-field interaction takes the form

$$H_{\text{MF}} = \sum_{\mathbf{q}}\sum_{\alpha\beta} \Sigma^{\alpha\beta}(\mathbf{q})\,c^\dagger_{\mathbf{q}\alpha}c_{\mathbf{q}\beta}$$

Collecting contributions from all four contraction channels (relabeling dummy indices), the **Hartree-Fock self-energy** is

$$\boxed{\Sigma^{\alpha\beta}(\mathbf{q}) = \frac{1}{2N}\sum_{\mathbf{k}}\sum_{\mu\nu}
\Bigl[
\underbrace{
\widetilde{V}^{\mu\nu\alpha\beta}(\mathbf{k},\mathbf{k},\mathbf{q})
+\widetilde{V}^{\alpha\beta\mu\nu}(\mathbf{q},\mathbf{q},\mathbf{k})
}_{\text{Hartree}}
\underbrace{
-\widetilde{V}^{\mu\beta\alpha\nu}(\mathbf{k},\mathbf{q},\mathbf{q})
-\widetilde{V}^{\alpha\nu\mu\beta}(\mathbf{q},\mathbf{k},\mathbf{k})
}_{\text{Fock}}
\Bigr]G^{\mu\nu}(\mathbf{k})}$$

where:
- **Hartree terms**: from same-side contractions; the three-momentum kernel is evaluated with the first two or last two momentum slots equal ($\mathbf{k},\mathbf{k}$ or $\mathbf{q},\mathbf{q}$), and summed over the one-body Green's function $G^{\mu\nu}(\mathbf{k})$;
- **Fock terms**: from cross-side contractions; the kernel is evaluated with the last two or first and third momentum slots equal, carrying a minus sign.

The two Hartree terms and two Fock terms are each related by the particle-exchange symmetry of the Coulomb integral $V^{abcd}_{ijkl} = V^{cdab}_{klij}$ (and exist independently in the fully general case).

---

### 5. Effective Single-Particle Hamiltonian and Self-Consistent Iteration

The single-body hopping term (diagonal in $\mathbf{k}$ after Fourier transform)

$$T^{\alpha\beta}(\mathbf{k}) = \sum_{\mathbf{r}} T^{\alpha\beta}(\mathbf{r})\,e^{i\mathbf{k}\cdot\mathbf{r}}$$

is combined with the mean-field self-energy to give the effective Hamiltonian at each $\mathbf{k}$ point:

$$\boxed{H^{\text{eff}}(\mathbf{k}) = T(\mathbf{k}) + \Sigma(\mathbf{k})}$$

The original four-body problem depending on three independent momenta is reduced, via HF decoupling and translational symmetry, to $N_k$ independent $d\times d$ eigenvalue problems (where $d$ is the number of degrees of freedom per unit cell).

Self-consistent iteration procedure:

1. Initialize $G^{\alpha\beta}(\mathbf{k})$ (from the occupied states of $T(\mathbf{k})$ or randomly)
2. Compute the self-energy $\Sigma(\mathbf{k})$ and construct $H^{\text{eff}}(\mathbf{k})$
3. Diagonalize $H^{\text{eff}}(\mathbf{k})$: $H^{\text{eff}}(\mathbf{k})\,|\psi_{n\mathbf{k}}\rangle = \varepsilon_{n\mathbf{k}}\,|\psi_{n\mathbf{k}}\rangle$
4. Update the one-body Green's function according to occupation numbers (step function at $T=0$ or Fermi-Dirac at finite temperature): $G^{\alpha\beta}(\mathbf{k}) = \sum_n f_{n\mathbf{k}}\,\psi^*_{n\alpha}(\mathbf{k})\,\psi_{n\beta}(\mathbf{k})$
5. Mix old and new $G(\mathbf{k})$, return to step 2, and repeat until convergence

---

### 6. Single-Variable Interactions and FFT Acceleration

The naive evaluation of the HF self-energy (§4) requires a double loop over $\mathbf{k}$ and $\mathbf{q}$, costing $O(N_k^2 d^4)$. A significant reduction to $O(N_k\log N_k)$ is possible whenever $\bar{V}^{abcd}(\boldsymbol{\tau}_1,\boldsymbol{\tau}_2,\boldsymbol{\tau}_3)$ depends on only a **single displacement variable**, i.e., two of the three $\boldsymbol{\tau}$'s are related by equality or vanish. There are three distinct structural cases.

#### 6.1 Case A — Density-Density ($\boldsymbol{\tau}_1=\boldsymbol{\tau}_2=\boldsymbol{\tau},\;\boldsymbol{\tau}_3=\mathbf{0}$)

This is the structure of Hubbard $U$, nearest-neighbor Coulomb $V$, and Hund's coupling: the $\mathbf{r}_1$-side pair $c^\dagger_i c_j$ has $i=j$, and the $\mathbf{r}_2$-side pair $c^\dagger_k c_l$ has $k=l$. In creation-annihilation alternating order this corresponds to operators $(c^\dagger_{\mathbf{R}_1},\, c_{\mathbf{R}_1},\, c^\dagger_{\mathbf{R}_2},\, c_{\mathbf{R}_2})$, giving

$$\bar{V}^{abcd}(\boldsymbol{\tau}_1,\boldsymbol{\tau}_2,\boldsymbol{\tau}_3)
= W^{abcd}(\boldsymbol{\tau})\,\delta_{\boldsymbol{\tau}_1,\boldsymbol{\tau}_2}\,\delta_{\boldsymbol{\tau}_3,\mathbf{0}}, \qquad
\boldsymbol{\tau} = \mathbf{R}_1 - \mathbf{R}_2$$

The three-momentum kernel collapses to a single-momentum function:

$$\widetilde{V}^{abcd}(\mathbf{k}_1,\mathbf{k}_2,\mathbf{k}_3) = \widetilde{W}^{abcd}(\mathbf{k}_2-\mathbf{k}_1)$$

Substituting into the four self-energy channels:

| Channel | $\widetilde{V}$ at that channel | Result |
|---|---|---|
| Hartree 1: $\widetilde{V}^{\mu\nu\alpha\beta}(k,k,q)$ | $\widetilde{W}^{\mu\nu\alpha\beta}(\mathbf{0})$ | **k-independent** |
| Hartree 2: $\widetilde{V}^{\alpha\beta\mu\nu}(q,q,k)$ | $\widetilde{W}^{\alpha\beta\mu\nu}(\mathbf{0})$ | **k-independent** |
| Fock 1: $\widetilde{V}^{\mu\beta\alpha\nu}(k,q,q)$ | $\widetilde{W}^{\mu\beta\alpha\nu}(\mathbf{q}-\mathbf{k})$ | **convolution** in $\mathbf{k}$ |
| Fock 2: $\widetilde{V}^{\alpha\nu\mu\beta}(q,k,k)$ | $\widetilde{W}^{\alpha\nu\mu\beta}(\mathbf{k}-\mathbf{q})$ | **convolution** in $\mathbf{k}$ |

**Hartree** ($\mathbf{q}$-independent):

$$\Sigma_H^{\alpha\beta} = \frac{1}{2}\sum_{\mu\nu}\left[\widetilde{W}^{\mu\nu\alpha\beta}(\mathbf{0})+\widetilde{W}^{\alpha\beta\mu\nu}(\mathbf{0})\right]\bar{G}^{\mu\nu}$$

where $\bar{G}^{\mu\nu} = \frac{1}{N}\sum_\mathbf{k}G^{\mu\nu}(\mathbf{k}) = G^{\mu\nu}(\mathbf{r}=\mathbf{0})$ is the on-site Green's function.

**Fock** (FFT-acceleratable via convolution theorem):

$$\Sigma_F^{\alpha\beta}(\mathbf{q}) = -\mathcal{F}_{\mathbf{r}\to\mathbf{q}}\!\left[\frac{1}{2}\sum_{\mu\nu}\left[W^{\mu\beta\alpha\nu}(\mathbf{r})+W^{\alpha\nu\mu\beta}(\mathbf{r})\right]G^{\mu\nu}(\mathbf{r})\right]$$

Real-space kernel: $\boldsymbol{W}(\mathbf{r})\cdot\boldsymbol{G}(\mathbf{r})$ (pointwise product, then FFT).

---

#### 6.2 Case B — Exchange-Type ($\boldsymbol{\tau}_1=\mathbf{0},\;\boldsymbol{\tau}_2=\boldsymbol{\tau}_3=\boldsymbol{\tau}$)

This structure arises from operators in creation-annihilation alternating order of the form $(c^\dagger_{\mathbf{R}_1},\, c_{\mathbf{R}_2},\, c^\dagger_{\mathbf{R}_2},\, c_{\mathbf{R}_1})$, i.e., $\boldsymbol{\tau}_1 = \mathbf{R}_1-\mathbf{R}_1=\mathbf{0}$ and $\boldsymbol{\tau}_2=\boldsymbol{\tau}_3=\mathbf{R}_2-\mathbf{R}_1$:

$$\bar{V}^{abcd}(\boldsymbol{\tau}_1,\boldsymbol{\tau}_2,\boldsymbol{\tau}_3)
= W^{abcd}(\boldsymbol{\tau})\,\delta_{\boldsymbol{\tau}_1,\mathbf{0}}\,\delta_{\boldsymbol{\tau}_2,\boldsymbol{\tau}_3}$$

$$\widetilde{V}^{abcd}(\mathbf{k}_1,\mathbf{k}_2,\mathbf{k}_3) = \widetilde{W}^{abcd}(\mathbf{k}_2-\mathbf{k}_3)$$

The channel structure is the complement of Case A:

| Channel | $\widetilde{V}$ at that channel | Result |
|---|---|---|
| Hartree 1: $\widetilde{V}^{\mu\nu\alpha\beta}(k,k,q)$ | $\widetilde{W}^{\mu\nu\alpha\beta}(\mathbf{k}-\mathbf{q})$ | **convolution** in $\mathbf{k}$ |
| Hartree 2: $\widetilde{V}^{\alpha\beta\mu\nu}(q,q,k)$ | $\widetilde{W}^{\alpha\beta\mu\nu}(\mathbf{q}-\mathbf{k})$ | **convolution** in $\mathbf{k}$ |
| Fock 1: $\widetilde{V}^{\mu\beta\alpha\nu}(k,q,q)$ | $\widetilde{W}^{\mu\beta\alpha\nu}(\mathbf{0})$ | **k-independent** |
| Fock 2: $\widetilde{V}^{\alpha\nu\mu\beta}(q,k,k)$ | $\widetilde{W}^{\alpha\nu\mu\beta}(\mathbf{0})$ | **k-independent** |

**Hartree** (FFT-acceleratable):

$$\Sigma_H^{\alpha\beta}(\mathbf{q}) = \frac{1}{2N}\sum_{\mathbf{k}}\sum_{\mu\nu}\left[\widetilde{W}^{\mu\nu\alpha\beta}(\mathbf{k}-\mathbf{q})+\widetilde{W}^{\alpha\beta\mu\nu}(\mathbf{q}-\mathbf{k})\right]G^{\mu\nu}(\mathbf{k})$$

$$= \mathcal{F}_{\mathbf{r}\to\mathbf{q}}\!\left[\frac{1}{2}\sum_{\mu\nu}\left[W^{\mu\nu\alpha\beta}(-\mathbf{r})+W^{\alpha\beta\mu\nu}(\mathbf{r})\right]G^{\mu\nu}(\mathbf{r})\right]$$

Real-space kernel: $\boldsymbol{W}(-\mathbf{r})\cdot\boldsymbol{G}(\mathbf{r})$.

**Fock** ($\mathbf{q}$-independent):

$$\Sigma_F^{\alpha\beta} = -\frac{1}{2}\sum_{\mu\nu}\left[\widetilde{W}^{\mu\beta\alpha\nu}(\mathbf{0})+\widetilde{W}^{\alpha\nu\mu\beta}(\mathbf{0})\right]\bar{G}^{\mu\nu}$$

---

#### 6.3 Case C — Pair-Hopping ($\boldsymbol{\tau}_1=\boldsymbol{\tau}_3=\boldsymbol{\tau},\;\boldsymbol{\tau}_2=\mathbf{0}$)

This structure arises for pair-hopping interactions, e.g., operators in creation-annihilation alternating order of the form $(c^\dagger_{\mathbf{R}_1},\, c_{\mathbf{R}_2},\, c^\dagger_{\mathbf{R}_1},\, c_{\mathbf{R}_2})$, giving $\boldsymbol{\tau}_1=\boldsymbol{\tau}_3=\mathbf{R}_1-\mathbf{R}_2$ and $\boldsymbol{\tau}_2=\mathbf{0}$:

$$\bar{V}^{abcd}(\boldsymbol{\tau}_1,\boldsymbol{\tau}_2,\boldsymbol{\tau}_3)
= W^{abcd}(\boldsymbol{\tau})\,\delta_{\boldsymbol{\tau}_1,\boldsymbol{\tau}_3}\,\delta_{\boldsymbol{\tau}_2,\mathbf{0}}$$

$$\widetilde{V}^{abcd}(\mathbf{k}_1,\mathbf{k}_2,\mathbf{k}_3) = \widetilde{W}^{abcd}(-(\mathbf{k}_1+\mathbf{k}_3)) = \widetilde{W}^{abcd}(\mathbf{k}_2-\mathbf{k}_4)$$

where the second equality uses momentum conservation $\mathbf{k}_1+\mathbf{k}_3=\mathbf{k}_2+\mathbf{k}_4$. All four channels now depend on $\mathbf{k}+\mathbf{q}$:

| Channel | $\widetilde{V}$ at that channel | Result |
|---|---|---|
| Hartree 1: $\widetilde{V}^{\mu\nu\alpha\beta}(k,k,q)$ | $\widetilde{W}^{\mu\nu\alpha\beta}(-(\mathbf{k}+\mathbf{q}))$ | **cross-correlation** in $\mathbf{k}$ |
| Hartree 2: $\widetilde{V}^{\alpha\beta\mu\nu}(q,q,k)$ | $\widetilde{W}^{\alpha\beta\mu\nu}(-(\mathbf{q}+\mathbf{k}))$ | **cross-correlation** in $\mathbf{k}$ |
| Fock 1: $\widetilde{V}^{\mu\beta\alpha\nu}(k,q,q)$ | $\widetilde{W}^{\mu\beta\alpha\nu}(-(\mathbf{k}+\mathbf{q}))$ | **cross-correlation** in $\mathbf{k}$ |
| Fock 2: $\widetilde{V}^{\alpha\nu\mu\beta}(q,k,k)$ | $\widetilde{W}^{\alpha\nu\mu\beta}(-(\mathbf{q}+\mathbf{k}))$ | **cross-correlation** in $\mathbf{k}$ |

The full self-energy is:

$$\Sigma^{\alpha\beta}(\mathbf{q}) = \frac{1}{2N}\sum_{\mathbf{k}}\sum_{\mu\nu}
\left[\widetilde{W}^{\mu\nu\alpha\beta}+\widetilde{W}^{\alpha\beta\mu\nu}
-\widetilde{W}^{\mu\beta\alpha\nu}-\widetilde{W}^{\alpha\nu\mu\beta}\right](-(\mathbf{k}+\mathbf{q}))\,G^{\mu\nu}(\mathbf{k})$$

The sum $\sum_\mathbf{k}\widetilde{W}(\mathbf{k}+\mathbf{q})\,G(\mathbf{k})$ is a **cross-correlation**. By substituting $\mathbf{k}'=\mathbf{k}+\mathbf{q}$ and applying the convolution theorem:

$$\frac{1}{N}\sum_{\mathbf{k}}\widetilde{W}(\mathbf{k}+\mathbf{q})\,G(\mathbf{k}) = \mathcal{F}_{\mathbf{r}\to\mathbf{q}}\!\left[W(\mathbf{r})\cdot G(-\mathbf{r})\right]$$

Therefore:

$$\Sigma^{\alpha\beta}(\mathbf{q}) = \frac{1}{2}\mathcal{F}_{\mathbf{r}\to\mathbf{q}}\!\left[\sum_{\mu\nu}
\left[W^{\mu\nu\alpha\beta}(\mathbf{r})+W^{\alpha\beta\mu\nu}(\mathbf{r})
-W^{\mu\beta\alpha\nu}(\mathbf{r})-W^{\alpha\nu\mu\beta}(\mathbf{r})\right]
G^{\mu\nu}(-\mathbf{r})\right]$$

Real-space kernel: $\boldsymbol{W}(\mathbf{r})\cdot\boldsymbol{G}(-\mathbf{r})$ (pointwise product with time-reversed Green's function, then FFT).

---

#### 6.4 Summary

The three FFT-acceleratable cases are distinguished by which $\boldsymbol{\tau}$ indices coincide:

| Case | $\boldsymbol{\tau}$ structure | $\widetilde{V}$ argument | Hartree | Fock | Real-space |
|---|---|---|---|---|---|
| **A** density-density | $\tau_1=\tau_2=\tau,\;\tau_3=0$ | $\mathbf{k}_2-\mathbf{k}_1$ | $\widetilde{W}(\mathbf{0})\bar{G}$ | $W(\mathbf{r})\cdot G(\mathbf{r})$ via FFT | $W\cdot G$ |
| **B** exchange-type | $\tau_1=0,\;\tau_2=\tau_3=\tau$ | $\mathbf{k}_2-\mathbf{k}_3$ | $W(-\mathbf{r})\cdot G(\mathbf{r})$ via FFT | $\widetilde{W}(\mathbf{0})\bar{G}$ | $W(-\cdot)\cdot G$ |
| **C** pair-hopping | $\tau_1=\tau_3=\tau,\;\tau_2=0$ | $-(\mathbf{k}_1+\mathbf{k}_3)$ | $W(\mathbf{r})\cdot G(-\mathbf{r})$ via FFT | $W(\mathbf{r})\cdot G(-\mathbf{r})$ via FFT | $W\cdot G(-\cdot)$ |

All three cases reduce the $O(N_k^2 d^4)$ direct summation to $O(N_k\log N_k)$. Note that Case A and Case B are related by swapping $\mathbf{r}\leftrightarrow-\mathbf{r}$ (i.e., the kernel $W(-\mathbf{r})$ is just the interaction evaluated at reversed displacement), and Case C uses the time-reversed Green's function $G(-\mathbf{r})$ instead. Any interaction that does not fall into one of these three cases requires the full three-momentum evaluation via `build_Uk` at $O(N_k^2 d^4)$ cost.

---

## Code Implementation

### 1. Preprocessing: Kinetic Term

**`build_Tr(dofs, ops, irvec)`** — Parses one-body operators and builds the real-space hopping table $T_{ab}(\mathbf{r})$ as a sparse entry list. Returns `(entries, d_int)` where each entry carries the displacement $\mathbf{r}$, orbital indices $(a, b)$, and hopping amplitude.

**`build_Tk(T_r, kgrid)`** — Fourier-transforms the $T(\mathbf{r})$ table to $T_{ab}(\mathbf{k})$ at every $\mathbf{k}$-point:

$$T_{ab}(\mathbf{k}) = \sum_{\mathbf{r}} T_{ab}(\mathbf{r})\,e^{i\mathbf{k}\cdot\mathbf{r}}$$

Returns a precomputed `Array{ComplexF64, 3}` of shape $(N_k, d, d)$.

### 2. Preprocessing: Interaction Term

**`build_Vr(dofs, ops, irvec)`** — Parses two-body operators and builds the real-space interaction table $\bar{V}^{abcd}(\boldsymbol{\tau}_1, \boldsymbol{\tau}_2, \boldsymbol{\tau}_3)$. Returns one $d\times d\times d\times d$ array per unique $(\boldsymbol{\tau}_1,\boldsymbol{\tau}_2,\boldsymbol{\tau}_3)$ triple.

**`build_Wr(V_r)`** — Extracts the single-displacement kernel $W^{abcd}(\boldsymbol{\tau})$ for Case A (density-density). Validates $\boldsymbol{\tau}_1=\boldsymbol{\tau}_2$ and $\boldsymbol{\tau}_3=\mathbf{0}$; returns `(mats, delta)` with the same convention as `build_Tr`.

**`build_Vk(V_r)`** — Returns a closure `(k1, k2, k3) -> Array{ComplexF64,4}` for the three-momentum kernel $\widetilde{V}(\mathbf{k}_1,\mathbf{k}_2,\mathbf{k}_3)$. Only needed for the general path (Case not A/B/C).

**`build_Uk(V_k)`** — Returns a closure `(k, q) -> Matrix{ComplexF64}` of shape $(d^2\times d^2)$ assembling the antisymmetrized HF matrix from all four Wick channels (Theory §4). Used in the general path.

### 3. Self-Consistent Field Iteration

**`build_heff_k!(H_k, T_k, V_r, G_k, G_r, kgrid)`** — Builds $H^\text{eff}(\mathbf{q}) = T(\mathbf{q}) + \Sigma(\mathbf{q})$ in-place. Automatically detects the $\boldsymbol{\tau}$ structure of `V_r` and selects the appropriate path:
- **Case A** (Theory §6.1): $O(N_k \log N_k)$, FFT with $W(\mathbf{r})\cdot G(\mathbf{r})$.
- **Case B** (Theory §6.2): $O(N_k \log N_k)$, FFT with $W(-\mathbf{r})\cdot G(\mathbf{r})$.
- **Case C** (Theory §6.3): $O(N_k \log N_k)$, FFT with $W(\mathbf{r})\cdot G(-\mathbf{r})$.
- **General path** (Theory §4): $O(N_k^2 d^4)$, direct summation via `build_Uk`.

**`solve_hfk(dofs, lattice, onebody, twobody, n_electrons)`** — Public entry point. Runs the full SCF loop described in Theory §5 and returns a `NamedTuple` with fields `G_k`, `G_r`, `eigenvalues`, `eigenvectors`, `energies` (band / interaction / total), `mu`, `ncond`, `kgrid`, `converged`, `iterations`, and `residual`.

Supports linear mixing and DIIS acceleration, multiple random restarts with lowest-energy selection, and finite-temperature Fermi-Dirac occupation.

> **To be added**: detailed step-by-step description of the SCF loop (initialization, $k$-point Hamiltonian construction, diagonalization, Green's function update, mixing, convergence criterion, and output fields) — pending completion of the code implementation.
