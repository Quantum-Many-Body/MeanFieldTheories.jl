# Momentum-Space Hartree-Fock: Derivation

## 1. Real-Space Starting Point

Consider the Coulomb interaction Hamiltonian projected onto a Wannier basis, written in **InterAll** form:

$$H_{\text{int}} = \frac{1}{2}\sum_{ijkl}\sum_{abcd} V^{abcd}_{ijkl}\, c^\dagger_{ia}\,c_{jb}\,c^\dagger_{kc}\,c_{ld}$$

where $i,j,k,l$ are site indices (each summed independently) and $a,b,c,d$ label all internal degrees of freedom (orbital, spin, etc.). The interaction matrix element is defined as

$$V^{abcd}_{ijkl} = \iint d\mathbf{r}_1\,d\mathbf{r}_2\;
w^*_{ia}(\mathbf{r}_1)\,w_{jb}(\mathbf{r}_1)\,
\frac{e^2}{|\mathbf{r}_1-\mathbf{r}_2|}\,
w^*_{kc}(\mathbf{r}_2)\,w_{ld}(\mathbf{r}_2)$$

where the Wannier functions satisfy $w_{i\alpha}(\mathbf{r}) = w_\alpha(\mathbf{r}-\mathbf{R}_i)$. The physical picture of the operator ordering is: $c^\dagger_{ia}c_{jb}$ contributes to the charge density at $\mathbf{r}_1$, and $c^\dagger_{kc}c_{ld}$ contributes to the charge density at $\mathbf{r}_2$.

---

## 2. Fourier Transform to $k$-Space

### 2.1 Fourier Transform Convention

$$c_{i\alpha} = \frac{1}{\sqrt{N}}\sum_{\mathbf{k}} e^{i\mathbf{k}\cdot\mathbf{R}_i}\,c_{\mathbf{k}\alpha}, \qquad
c^\dagger_{i\alpha} = \frac{1}{\sqrt{N}}\sum_{\mathbf{k}} e^{-i\mathbf{k}\cdot\mathbf{R}_i}\,c^\dagger_{\mathbf{k}\alpha}$$

### 2.2 Translational Invariance

Since $w_{i\alpha}(\mathbf{r}) = w_\alpha(\mathbf{r}-\mathbf{R}_i)$, the matrix element depends only on three relative displacements. Taking $\mathbf{R}_l$ as the reference site, define

$$\boldsymbol{\tau}_1 = \mathbf{R}_i - \mathbf{R}_l, \quad
\boldsymbol{\tau}_2 = \mathbf{R}_j - \mathbf{R}_l, \quad
\boldsymbol{\tau}_3 = \mathbf{R}_k - \mathbf{R}_l$$

Writing $\bar{V}^{abcd}(\boldsymbol{\tau}_1,\boldsymbol{\tau}_2,\boldsymbol{\tau}_3) \equiv V^{abcd}_{ijkl}$, the Hamiltonian becomes

$$H_{\text{int}} = \frac{1}{2}\sum_l\sum_{\boldsymbol{\tau}_1\boldsymbol{\tau}_2\boldsymbol{\tau}_3}\sum_{abcd}
\bar{V}^{abcd}(\boldsymbol{\tau}_1,\boldsymbol{\tau}_2,\boldsymbol{\tau}_3)\,
c^\dagger_{l+\tau_1,\,a}\,c_{l+\tau_2,\,b}\,c^\dagger_{l+\tau_3,\,c}\,c_{l,d}$$

### 2.3 Substituting the Fourier Expansion

Expanding each of the four operators, the four phase factors are

$$e^{-i\mathbf{k}_1\cdot\mathbf{R}_i}\cdot e^{+i\mathbf{k}_2\cdot\mathbf{R}_j}\cdot e^{-i\mathbf{k}_3\cdot\mathbf{R}_k}\cdot e^{+i\mathbf{k}_4\cdot\mathbf{R}_l}$$

Rewriting in relative coordinates and separating out the part depending on the reference site $\mathbf{R}_l$:

$$= e^{-i\mathbf{k}_1\cdot\boldsymbol{\tau}_1 +i\mathbf{k}_2\cdot\boldsymbol{\tau}_2 -i\mathbf{k}_3\cdot\boldsymbol{\tau}_3}
\cdot e^{\,i(-\mathbf{k}_1+\mathbf{k}_2-\mathbf{k}_3+\mathbf{k}_4)\cdot\mathbf{R}_l}$$

Summing over all $\mathbf{R}_l$ imposes **momentum conservation**:

$$\sum_l e^{\,i(-\mathbf{k}_1+\mathbf{k}_2-\mathbf{k}_3+\mathbf{k}_4)\cdot\mathbf{R}_l}
= N\,\delta_{\mathbf{k}_1+\mathbf{k}_3,\,\mathbf{k}_2+\mathbf{k}_4}$$

### 2.4 Three-Momentum Interaction Kernel

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

## 3. Hartree-Fock Decoupling

### 3.1 Density Matrix

For a ground state that preserves discrete translational symmetry, the single-particle density matrix is diagonal in momentum space:

$$\langle c^\dagger_{\mathbf{k}\alpha}\,c_{\mathbf{k}'\beta}\rangle
= \delta_{\mathbf{k},\mathbf{k}'}\,\rho^{\alpha\beta}(\mathbf{k})$$

States at different $\mathbf{k}$ points are uncorrelated. If translational symmetry is spontaneously broken (e.g., antiferromagnetic order, charge density wave), the magnetic unit cell must be adopted as the new unit cell and $\mathbf{k}$ redefined in the corresponding Brillouin zone.

### 3.2 Wick Decomposition

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

### 3.3 Momentum Constraints

Using $\langle c^\dagger_{\mathbf{k}\alpha}c_{\mathbf{k}'\beta}\rangle = \delta_{\mathbf{k},\mathbf{k}'}\rho^{\alpha\beta}(\mathbf{k})$, each contraction under the momentum conservation constraint $\mathbf{k}_1+\mathbf{k}_3 = \mathbf{k}_2+\mathbf{k}_4$ yields the following:

| Contracted pair | Channel | Constraint | Remaining bilinear |
|:------:|:--:|:--------:|:----------:|
| $(c^\dagger_{\mathbf{k}_1 a},\,c_{\mathbf{k}_2 b})$ | Hartree | $\mathbf{k}_1=\mathbf{k}_2=\mathbf{k}$, $\mathbf{k}_3=\mathbf{k}_4=\mathbf{q}$ | $c^\dagger_{\mathbf{q}c}c_{\mathbf{q}d}$ |
| $(c^\dagger_{\mathbf{k}_3 c},\,c_{\mathbf{k}_4 d})$ | Hartree | $\mathbf{k}_3=\mathbf{k}_4=\mathbf{k}$, $\mathbf{k}_1=\mathbf{k}_2=\mathbf{q}$ | $c^\dagger_{\mathbf{q}a}c_{\mathbf{q}b}$ |
| $(c^\dagger_{\mathbf{k}_1 a},\,c_{\mathbf{k}_4 d})$ | Fock | $\mathbf{k}_1=\mathbf{k}_4=\mathbf{k}$, $\mathbf{k}_2=\mathbf{k}_3=\mathbf{q}$ | $-c^\dagger_{\mathbf{q}c}c_{\mathbf{q}b}$ |
| $(c^\dagger_{\mathbf{k}_3 c},\,c_{\mathbf{k}_2 b})$ | Fock | $\mathbf{k}_2=\mathbf{k}_3=\mathbf{k}$, $\mathbf{k}_1=\mathbf{k}_4=\mathbf{q}$ | $-c^\dagger_{\mathbf{q}a}c_{\mathbf{q}d}$ |

Each contraction reduces the four-operator product to a bilinear $c^\dagger_{\mathbf{q}\alpha}c_{\mathbf{q}\beta}$ at momentum $\mathbf{q}$, reflecting the $k$-diagonal structure of the effective Hamiltonian in a translationally symmetric ground state.

---

## 4. Hartree-Fock Self-Energy

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
\Bigr]\rho^{\mu\nu}(\mathbf{k})}$$

where:
- **Hartree terms**: from same-side contractions; the three-momentum kernel is evaluated with the first two or last two momentum slots equal ($\mathbf{k},\mathbf{k}$ or $\mathbf{q},\mathbf{q}$), and summed over the density matrix $\rho^{\mu\nu}(\mathbf{k})$;
- **Fock terms**: from cross-side contractions; the kernel is evaluated with the last two or first and third momentum slots equal, carrying a minus sign.

The two Hartree terms and two Fock terms are each related by the particle-exchange symmetry of the Coulomb integral $V^{abcd}_{ijkl} = V^{cdab}_{klij}$ (and exist independently in the fully general case).

---

## 5. Effective Single-Particle Hamiltonian and Self-Consistent Iteration

The single-body hopping term (diagonal in $\mathbf{k}$ after Fourier transform)

$$T^{\alpha\beta}(\mathbf{k}) = \sum_{\mathbf{r}} T^{\alpha\beta}(\mathbf{r})\,e^{i\mathbf{k}\cdot\mathbf{r}}$$

is combined with the mean-field self-energy to give the effective Hamiltonian at each $\mathbf{k}$ point:

$$\boxed{H^{\text{eff}}(\mathbf{k}) = T(\mathbf{k}) + \Sigma(\mathbf{k})}$$

The original four-body problem depending on three independent momenta is reduced, via HF decoupling and translational symmetry, to $N_k$ independent $d\times d$ eigenvalue problems (where $d$ is the number of degrees of freedom per unit cell).

Self-consistent iteration procedure:

1. Initialize $\rho^{\alpha\beta}(\mathbf{k})$ (from the occupied states of $T(\mathbf{k})$ or randomly)
2. Compute the self-energy $\Sigma(\mathbf{k})$ and construct $H^{\text{eff}}(\mathbf{k})$
3. Diagonalize $H^{\text{eff}}(\mathbf{k})$: $H^{\text{eff}}(\mathbf{k})\,|\psi_{n\mathbf{k}}\rangle = \varepsilon_{n\mathbf{k}}\,|\psi_{n\mathbf{k}}\rangle$
4. Update the density matrix according to occupation numbers (step function at $T=0$ or Fermi-Dirac at finite temperature): $\rho^{\alpha\beta}(\mathbf{k}) = \sum_n f_{n\mathbf{k}}\,\psi^*_{n\alpha}(\mathbf{k})\,\psi_{n\beta}(\mathbf{k})$
5. Mix old and new density matrices, return to step 2, and repeat until convergence

---

## 6. Special Case: Density-Density Interactions

For the most common lattice models (Hubbard $U$, nearest-neighbor Coulomb $V$, Hund's coupling, exchange interactions, etc.), the interaction has a "density-density" structure: the Wannier function integral on each side is dominated by a single site, i.e., $i=j$ ($\mathbf{r}_1$-side density localized at site $i$) and $k=l$ ($\mathbf{r}_2$-side density localized at site $k$). In this case,

$$\bar{V}^{abcd}(\boldsymbol{\tau}_1,\boldsymbol{\tau}_2,\boldsymbol{\tau}_3)
= W^{abcd}(\boldsymbol{\tau})\,\delta_{\boldsymbol{\tau}_1,\boldsymbol{\tau}_2}\,\delta_{\boldsymbol{\tau}_3,\mathbf{0}}$$

where $\boldsymbol{\tau} = \boldsymbol{\tau}_1 = \boldsymbol{\tau}_2 = \mathbf{R}_i - \mathbf{R}_l$ is the displacement between the two density operators. The three-momentum kernel reduces to a **single-momentum** function:

$$\widetilde{V}^{abcd}(\mathbf{k}_1,\mathbf{k}_2,\mathbf{k}_3)
= \widetilde{W}^{abcd}(\mathbf{k}_2-\mathbf{k}_1)$$

The self-energy simplifies accordingly:

**Hartree** ($\mathbf{k}$-independent):

$$\Sigma_H^{\alpha\beta} = \sum_{\mu\nu}\widetilde{W}^{\mu\nu\alpha\beta}(\mathbf{0})\,\bar{\rho}^{\mu\nu}
= \sum_{\mu\nu}\widetilde{W}^{\alpha\beta\mu\nu}(\mathbf{0})\,\bar{\rho}^{\mu\nu}$$

where $\bar{\rho}^{\mu\nu} = \frac{1}{N}\sum_\mathbf{k}\rho^{\mu\nu}(\mathbf{k})$ is the real-space on-site density matrix.

**Fock** ($\mathbf{k}$-dependent, FFT-acceleratable):

$$\Sigma_F^{\alpha\beta}(\mathbf{q}) = -\frac{1}{N}\sum_{\mathbf{k}}\sum_{\mu\nu}
\frac{1}{2}\!\left[\widetilde{W}^{\mu\beta\alpha\nu}(\mathbf{q}-\mathbf{k})
+\widetilde{W}^{\alpha\nu\mu\beta}(\mathbf{q}-\mathbf{k})\right]\rho^{\mu\nu}(\mathbf{k})$$

This is a **convolution** in $\mathbf{k}$-space. By the convolution theorem, it can be transformed into a pointwise product in real space followed by an FFT, reducing the naive $O(N_k^2)$ cost to $O(N_k\log N_k)$:

$$\Sigma_F^{\alpha\beta}(\mathbf{q}) = -\mathcal{F}_{\mathbf{r}\to\mathbf{q}}\!\left[\sum_{\mu\nu}
\frac{1}{2}\!\left[W^{\mu\beta\alpha\nu}(\mathbf{r})+W^{\alpha\nu\mu\beta}(\mathbf{r})\right]
\rho^{\mu\nu}(\mathbf{r})\right]$$

where $\rho^{\mu\nu}(\mathbf{r}) = \frac{1}{N}\sum_\mathbf{k} e^{-i\mathbf{k}\cdot\mathbf{r}}\rho^{\mu\nu}(\mathbf{k})$ is the real-space density matrix.
