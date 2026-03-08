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

$$\langle c^\dagger_{\mathbf{k}a}\,c_{\mathbf{k}'b}\rangle
= \delta_{\mathbf{k},\mathbf{k}'}\,G^{ab}(\mathbf{k})$$

Its real-space counterpart is defined by the inverse Fourier transform

$$G^{ab}(\mathbf{r}) \equiv \frac{1}{N}\sum_{\mathbf{k}} G^{ab}(\mathbf{k})\,e^{-i\mathbf{k}\cdot\mathbf{r}} = \langle c^\dagger_{\mathbf{r},a}\,c_{\mathbf{0},b}\rangle$$

which uses the **negative** phase convention (creation operator carries $e^{-i\mathbf{k}\cdot\mathbf{r}}$, consistent with §2.1). This pairs with the interaction kernel convention $\widetilde{W}(\mathbf{q})=\sum_\mathbf{r}W(\mathbf{r})\,e^{+i\mathbf{q}\cdot\mathbf{r}}$ so that both $G$ and $\widetilde{W}$ are forward Fourier transforms; the FFT identities in §6 hold under this consistent choice.

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

Using $\langle c^\dagger_{\mathbf{k}a}c_{\mathbf{k}'b}\rangle = \delta_{\mathbf{k},\mathbf{k}'}G^{ab}(\mathbf{k})$, each contraction under the momentum conservation constraint $\mathbf{k}_1+\mathbf{k}_3 = \mathbf{k}_2+\mathbf{k}_4$ yields the following:

| Contracted pair | Channel | Constraint | Remaining bilinear |
|:------:|:--:|:--------:|:----------:|
| $(c^\dagger_{\mathbf{k}_1 a},\,c_{\mathbf{k}_2 b})$ | Hartree | $\mathbf{k}_1=\mathbf{k}_2=\mathbf{k}$, $\mathbf{k}_3=\mathbf{k}_4=\mathbf{q}$ | $c^\dagger_{\mathbf{q}c}c_{\mathbf{q}d}$ |
| $(c^\dagger_{\mathbf{k}_3 c},\,c_{\mathbf{k}_4 d})$ | Hartree | $\mathbf{k}_3=\mathbf{k}_4=\mathbf{k}$, $\mathbf{k}_1=\mathbf{k}_2=\mathbf{q}$ | $c^\dagger_{\mathbf{q}a}c_{\mathbf{q}b}$ |
| $(c^\dagger_{\mathbf{k}_1 a},\,c_{\mathbf{k}_4 d})$ | Fock | $\mathbf{k}_1=\mathbf{k}_4=\mathbf{k}$, $\mathbf{k}_2=\mathbf{k}_3=\mathbf{q}$ | $-c^\dagger_{\mathbf{q}c}c_{\mathbf{q}b}$ |
| $(c^\dagger_{\mathbf{k}_3 c},\,c_{\mathbf{k}_2 b})$ | Fock | $\mathbf{k}_2=\mathbf{k}_3=\mathbf{k}$, $\mathbf{k}_1=\mathbf{k}_4=\mathbf{q}$ | $-c^\dagger_{\mathbf{q}a}c_{\mathbf{q}d}$ |

Each contraction reduces the four-operator product to a bilinear $c^\dagger_{\mathbf{q}a}c_{\mathbf{q}b}$ at momentum $\mathbf{q}$, reflecting the $k$-diagonal structure of the effective Hamiltonian in a translationally symmetric ground state.

---

### 4. Hartree-Fock Self-Energy

Substituting all contractions back into $H_{\text{int}}$, the mean-field interaction takes the form

$$H_{\text{MF}} = \sum_{\mathbf{q}}\sum_{ab} \Sigma^{ab}(\mathbf{q})\,c^\dagger_{\mathbf{q}a}c_{\mathbf{q}b}$$

Collecting contributions from all four contraction channels (tracing through §3.3 with $a,b$ as the free output indices of $\Sigma^{ab}$ and $c,d$ as dummy summation indices), the **Hartree-Fock self-energy** is

$$\boxed{\Sigma^{ab}(\mathbf{q}) = \frac{1}{2N}\sum_{\mathbf{k}}\sum_{cd}
\Bigl[
\underbrace{
\widetilde{V}^{cdab}(\mathbf{k},\mathbf{k},\mathbf{q})
+\widetilde{V}^{abcd}(\mathbf{q},\mathbf{q},\mathbf{k})
}_{\text{Hartree}}
\underbrace{
-\widetilde{V}^{cbad}(\mathbf{k},\mathbf{q},\mathbf{q})
-\widetilde{V}^{adcb}(\mathbf{q},\mathbf{k},\mathbf{k})
}_{\text{Fock}}
\Bigr]G^{cd}(\mathbf{k})}$$

where:
- **Hartree terms**: from same-side contractions; the kernel is evaluated with the first two or last two momentum slots equal ($\mathbf{k},\mathbf{k}$ or $\mathbf{q},\mathbf{q}$), and contracted with $G^{cd}(\mathbf{k})$. The index order $cdab$ (H1) and $abcd$ (H2) follows directly from which pair is contracted and which remains.
- **Fock terms**: from cross-side contractions; the kernel is evaluated with mixed momentum slots, carrying a minus sign. The index order $cbad$ (F1) and $adcb$ (F2) arises because the cross contraction swaps which creation/annihilation index is "free" vs "summed".

The two Hartree terms and two Fock terms are each related by the particle-exchange symmetry of the Coulomb integral $V^{abcd}_{ijkl} = V^{cdab}_{klij}$ (and exist independently in the fully general case).

---

### 5. Effective Single-Particle Hamiltonian and Self-Consistent Iteration

The single-body hopping term (diagonal in $\mathbf{k}$ after Fourier transform)

$$T^{ab}(\mathbf{k}) = \sum_{\mathbf{r}} T^{ab}(\mathbf{r})\,e^{i\mathbf{k}\cdot\mathbf{r}}$$

is combined with the mean-field self-energy to give the effective Hamiltonian at each $\mathbf{k}$ point:

$$\boxed{H^{\text{eff}}(\mathbf{k}) = T(\mathbf{k}) + \Sigma(\mathbf{k})}$$

The original four-body problem depending on three independent momenta is reduced, via HF decoupling and translational symmetry, to $N_k$ independent $d\times d$ eigenvalue problems (where $d$ is the number of degrees of freedom per unit cell).

Self-consistent iteration procedure:

1. Initialize $G^{ab}(\mathbf{k})$ (from the occupied states of $T(\mathbf{k})$ or randomly)
2. Compute the self-energy $\Sigma^{ab}(\mathbf{k})$ and construct $H^{\text{eff}}(\mathbf{k})$
3. Diagonalize $H^{\text{eff}}(\mathbf{k})$: $H^{\text{eff}}(\mathbf{k})\,|\psi_{n\mathbf{k}}\rangle = \varepsilon_{n\mathbf{k}}\,|\psi_{n\mathbf{k}}\rangle$
4. Update the one-body Green's function according to occupation numbers (step function at $T=0$ or Fermi-Dirac at finite temperature): $G^{ab}(\mathbf{k}) = \sum_n f_{n\mathbf{k}}\,\psi^*_{na}(\mathbf{k})\,\psi_{nb}(\mathbf{k})$
5. Mix old and new $G(\mathbf{k})$, return to step 2, and repeat until convergence

---

### 6. Single-Variable Interactions and FFT Acceleration

The naive evaluation of the HF self-energy (§4) requires a double loop over $\mathbf{k}$ and $\mathbf{q}$, costing $O(N_k^2 d^4)$. A significant reduction to $O(N_k\log N_k)$ is possible whenever $\bar{V}^{abcd}(\boldsymbol{\tau}_1,\boldsymbol{\tau}_2,\boldsymbol{\tau}_3)$ depends on only a **single displacement variable**, i.e., two of the three $\boldsymbol{\tau}$'s are related by equality or vanish. There are three distinct structural cases.

> **Convolution Identities.** All three cases reduce to one of the following key identities (each verified by substituting the FT convention $\widetilde{f}(\mathbf{q})=\sum_\mathbf{r}f(\mathbf{r})\,e^{i\mathbf{q}\cdot\mathbf{r}}$ directly):
>
> $$\frac{1}{N}\sum_\mathbf{k}\widetilde{A}(\mathbf{q}-\mathbf{k})\,\widetilde{B}(\mathbf{k}) = \mathcal{F}_{\mathbf{r}\to\mathbf{q}}[A(\mathbf{r})\,B(\mathbf{r})] \qquad\text{(standard convolution)}$$
>
> $$\frac{1}{N}\sum_\mathbf{k}\widetilde{A}(\mathbf{k}-\mathbf{q})\,\widetilde{B}(\mathbf{k}) = \mathcal{F}_{\mathbf{r}\to\mathbf{q}}[A(-\mathbf{r})\,B(\mathbf{r})] \qquad\text{(cross-correlation)}$$
>
> $$\frac{1}{N}\sum_\mathbf{k}\widetilde{A}(-(\mathbf{k}+\mathbf{q}))\,\widetilde{B}(\mathbf{k}) = \mathcal{F}_{\mathbf{r}\to\mathbf{q}}[A(-\mathbf{r})\,B(-\mathbf{r})]$$
>
> The sign of $\mathbf{k}$ in the $\widetilde{A}$ argument determines which real-space function appears: $(\mathbf{q}-\mathbf{k})$ gives $A(\mathbf{r})$; $(\mathbf{k}-\mathbf{q})$ gives $A(-\mathbf{r})$; $-(\mathbf{k}+\mathbf{q})$ gives both $A(-\mathbf{r})$ and $B(-\mathbf{r})$. Mixing up these cases is the most common source of sign errors in the FFT acceleration.

#### 6.1 Case A — Density-Density ($\boldsymbol{\tau}_1=\boldsymbol{\tau}_2=\boldsymbol{\tau},\;\boldsymbol{\tau}_3=\mathbf{0}$)

This is the structure of Hubbard $U$, nearest-neighbor Coulomb $V$, and Hund's coupling: the $\mathbf{r}_1$-side pair $c^\dagger_i c_j$ has $i=j$, and the $\mathbf{r}_2$-side pair $c^\dagger_k c_l$ has $k=l$. In creation-annihilation alternating order this corresponds to operators $(c^\dagger_{\mathbf{R}_1},\, c_{\mathbf{R}_1},\, c^\dagger_{\mathbf{R}_2},\, c_{\mathbf{R}_2})$, giving

$$\bar{V}^{abcd}(\boldsymbol{\tau}_1,\boldsymbol{\tau}_2,\boldsymbol{\tau}_3)
= W^{abcd}(\boldsymbol{\tau})\,\delta_{\boldsymbol{\tau}_1,\boldsymbol{\tau}_2}\,\delta_{\boldsymbol{\tau}_3,\mathbf{0}}, \qquad
\boldsymbol{\tau} = \mathbf{R}_1 - \mathbf{R}_2$$

The three-momentum kernel collapses to a single-momentum function:

$$\widetilde{V}^{abcd}(\mathbf{k}_1,\mathbf{k}_2,\mathbf{k}_3) = \widetilde{W}^{abcd}(\mathbf{k}_2-\mathbf{k}_1)$$

Substituting into the four self-energy channels (using $a,b$ as free indices and $c,d$ as summation indices, matching §4):

| Channel | $\widetilde{V}$ at that channel | Result |
|---|---|---|
| Hartree 1: $\widetilde{V}^{cdab}(k,k,q)$ | $\widetilde{W}^{cdab}(\mathbf{0})$ | **k-independent** |
| Hartree 2: $\widetilde{V}^{abcd}(q,q,k)$ | $\widetilde{W}^{abcd}(\mathbf{0})$ | **k-independent** |
| Fock 1: $\widetilde{V}^{cbad}(k,q,q)$ | $\widetilde{W}^{cbad}(\mathbf{q}-\mathbf{k})$ | **convolution** in $\mathbf{k}$ |
| Fock 2: $\widetilde{V}^{adcb}(q,k,k)$ | $\widetilde{W}^{adcb}(\mathbf{k}-\mathbf{q})$ | **cross-correlation** in $\mathbf{k}$ |

**Hartree** ($\mathbf{q}$-independent):

$$\Sigma_H^{ab} = \frac{1}{2}\sum_{cd}\left[\widetilde{W}^{cdab}(\mathbf{0})+\widetilde{W}^{abcd}(\mathbf{0})\right]\bar{G}^{cd}$$

where $\bar{G}^{cd} = \frac{1}{N}\sum_\mathbf{k}G^{cd}(\mathbf{k}) = G^{cd}(\mathbf{r}=\mathbf{0})$ is the on-site Green's function.

**Fock** (FFT-acceleratable via convolution theorem):

$$\Sigma_F^{ab}(\mathbf{q}) = -\frac{1}{2}\mathcal{F}_{\mathbf{r}\to\mathbf{q}}\!\left[\sum_{cd}\left[W^{cbad}(\mathbf{r})+W^{adcb}(-\mathbf{r})\right]G^{cd}(\mathbf{r})\right]$$

Real-space kernel: Fock 1 contributes $W^{cbad}(\mathbf{r})\cdot G^{cd}(\mathbf{r})$ (standard convolution, argument $\mathbf{q}-\mathbf{k}$); Fock 2 contributes $W^{adcb}(-\mathbf{r})\cdot G^{cd}(\mathbf{r})$ (cross-correlation, argument $\mathbf{k}-\mathbf{q}$). They share the same FFT after summing the two $W$ kernels pointwise.

---

#### 6.2 Case B — Exchange-Type ($\boldsymbol{\tau}_1=\mathbf{0},\;\boldsymbol{\tau}_2=\boldsymbol{\tau}_3=\boldsymbol{\tau}$)

This structure arises from operators in creation-annihilation alternating order of the form $(c^\dagger_{\mathbf{R}_1},\, c_{\mathbf{R}_2},\, c^\dagger_{\mathbf{R}_2},\, c_{\mathbf{R}_1})$, i.e., $\boldsymbol{\tau}_1 = \mathbf{R}_1-\mathbf{R}_1=\mathbf{0}$ and $\boldsymbol{\tau}_2=\boldsymbol{\tau}_3=\mathbf{R}_2-\mathbf{R}_1$:

$$\bar{V}^{abcd}(\boldsymbol{\tau}_1,\boldsymbol{\tau}_2,\boldsymbol{\tau}_3)
= W^{abcd}(\boldsymbol{\tau})\,\delta_{\boldsymbol{\tau}_1,\mathbf{0}}\,\delta_{\boldsymbol{\tau}_2,\boldsymbol{\tau}_3}$$

$$\widetilde{V}^{abcd}(\mathbf{k}_1,\mathbf{k}_2,\mathbf{k}_3) = \widetilde{W}^{abcd}(\mathbf{k}_2-\mathbf{k}_3)$$

The channel structure is the complement of Case A:

| Channel | $\widetilde{V}$ at that channel | Result |
|---|---|---|
| Hartree 1: $\widetilde{V}^{cdab}(k,k,q)$ | $\widetilde{W}^{cdab}(\mathbf{k}-\mathbf{q})$ | **cross-correlation** in $\mathbf{k}$ |
| Hartree 2: $\widetilde{V}^{abcd}(q,q,k)$ | $\widetilde{W}^{abcd}(\mathbf{q}-\mathbf{k})$ | **convolution** in $\mathbf{k}$ |
| Fock 1: $\widetilde{V}^{cbad}(k,q,q)$ | $\widetilde{W}^{cbad}(\mathbf{0})$ | **k-independent** |
| Fock 2: $\widetilde{V}^{adcb}(q,k,k)$ | $\widetilde{W}^{adcb}(\mathbf{0})$ | **k-independent** |

**Hartree** (FFT-acceleratable):

$$\Sigma_H^{ab}(\mathbf{q}) = \frac{1}{2N}\sum_{\mathbf{k}}\sum_{cd}\left[\widetilde{W}^{cdab}(\mathbf{k}-\mathbf{q})+\widetilde{W}^{abcd}(\mathbf{q}-\mathbf{k})\right]G^{cd}(\mathbf{k})$$

$$= \mathcal{F}_{\mathbf{r}\to\mathbf{q}}\!\left[\frac{1}{2}\sum_{cd}\left[W^{cdab}(-\mathbf{r})+W^{abcd}(\mathbf{r})\right]G^{cd}(\mathbf{r})\right]$$

Real-space kernel: H1 gives $W^{cdab}(-\mathbf{r})\cdot G^{cd}(\mathbf{r})$ (cross-correlation, argument $\mathbf{k}-\mathbf{q}$); H2 gives $W^{abcd}(\mathbf{r})\cdot G^{cd}(\mathbf{r})$ (standard convolution, argument $\mathbf{q}-\mathbf{k}$).

**Fock** ($\mathbf{q}$-independent):

$$\Sigma_F^{ab} = -\frac{1}{2}\sum_{cd}\left[\widetilde{W}^{cbad}(\mathbf{0})+\widetilde{W}^{adcb}(\mathbf{0})\right]\bar{G}^{cd}$$

---

#### 6.3 Case C — Pair-Hopping ($\boldsymbol{\tau}_1=\boldsymbol{\tau}_3=\boldsymbol{\tau},\;\boldsymbol{\tau}_2=\mathbf{0}$)

This structure arises for pair-hopping interactions, e.g., operators in creation-annihilation alternating order of the form $(c^\dagger_{\mathbf{R}_1},\, c_{\mathbf{R}_2},\, c^\dagger_{\mathbf{R}_1},\, c_{\mathbf{R}_2})$, giving $\boldsymbol{\tau}_1=\boldsymbol{\tau}_3=\mathbf{R}_1-\mathbf{R}_2$ and $\boldsymbol{\tau}_2=\mathbf{0}$:

$$\bar{V}^{abcd}(\boldsymbol{\tau}_1,\boldsymbol{\tau}_2,\boldsymbol{\tau}_3)
= W^{abcd}(\boldsymbol{\tau})\,\delta_{\boldsymbol{\tau}_1,\boldsymbol{\tau}_3}\,\delta_{\boldsymbol{\tau}_2,\mathbf{0}}$$

$$\widetilde{V}^{abcd}(\mathbf{k}_1,\mathbf{k}_2,\mathbf{k}_3) = \widetilde{W}^{abcd}(-(\mathbf{k}_1+\mathbf{k}_3)) = \widetilde{W}^{abcd}(-(\mathbf{k}_2+\mathbf{k}_4))$$

where the second equality uses momentum conservation $\mathbf{k}_1+\mathbf{k}_3=\mathbf{k}_2+\mathbf{k}_4$. All four channels now depend on $\mathbf{k}+\mathbf{q}$:

| Channel | $\widetilde{V}$ at that channel | Result |
|---|---|---|
| Hartree 1: $\widetilde{V}^{cdab}(k,k,q)$ | $\widetilde{W}^{cdab}(-(\mathbf{k}+\mathbf{q}))$ | **cross-correlation** in $\mathbf{k}$ |
| Hartree 2: $\widetilde{V}^{abcd}(q,q,k)$ | $\widetilde{W}^{abcd}(-(\mathbf{q}+\mathbf{k}))$ | **cross-correlation** in $\mathbf{k}$ |
| Fock 1: $\widetilde{V}^{cbad}(k,q,q)$ | $\widetilde{W}^{cbad}(-(\mathbf{k}+\mathbf{q}))$ | **cross-correlation** in $\mathbf{k}$ |
| Fock 2: $\widetilde{V}^{adcb}(q,k,k)$ | $\widetilde{W}^{adcb}(-(\mathbf{q}+\mathbf{k}))$ | **cross-correlation** in $\mathbf{k}$ |

The full self-energy is:

$$\Sigma^{ab}(\mathbf{q}) = \frac{1}{2N}\sum_{\mathbf{k}}\sum_{cd}
\left[\widetilde{W}^{cdab}(-(\mathbf{k}+\mathbf{q}))+\widetilde{W}^{abcd}(-(\mathbf{k}+\mathbf{q}))
-\widetilde{W}^{cbad}(-(\mathbf{k}+\mathbf{q}))-\widetilde{W}^{adcb}(-(\mathbf{k}+\mathbf{q}))\right]G^{cd}(\mathbf{k})$$

Expanding $\widetilde{W}(-(\mathbf{k}+\mathbf{q})) = \sum_\mathbf{r} W(\mathbf{r})\,e^{-i(\mathbf{k}+\mathbf{q})\cdot\mathbf{r}}$ and using $\frac{1}{N}\sum_\mathbf{k}G(\mathbf{k})\,e^{-i\mathbf{k}\cdot\mathbf{r}} = G(\mathbf{r})$:

$$\frac{1}{N}\sum_{\mathbf{k}}\widetilde{W}(-(\mathbf{k}+\mathbf{q}))\,G(\mathbf{k})
= \sum_\mathbf{r} W(\mathbf{r})\,G(\mathbf{r})\,e^{-i\mathbf{q}\cdot\mathbf{r}}
= \mathcal{F}_{\mathbf{r}\to\mathbf{q}}\!\left[W(-\mathbf{r})\cdot G(-\mathbf{r})\right]$$

where the last step substitutes $\mathbf{r}\to-\mathbf{r}$. Therefore:

$$\Sigma^{ab}(\mathbf{q}) = \frac{1}{2}\mathcal{F}_{\mathbf{r}\to\mathbf{q}}\!\left[\sum_{cd}
\left[W^{cdab}(-\mathbf{r})+W^{abcd}(-\mathbf{r})
-W^{cbad}(-\mathbf{r})-W^{adcb}(-\mathbf{r})\right]
G^{cd}(-\mathbf{r})\right]$$

Real-space kernel: $\boldsymbol{W}(-\mathbf{r})\cdot\boldsymbol{G}(-\mathbf{r})$ (pointwise product of interaction and time-reversed Green's function, both evaluated at $-\mathbf{r}$, then FFT).

---

#### 6.4 Summary

The three FFT-acceleratable cases are distinguished by which $\boldsymbol{\tau}$ indices coincide:

| Case | $\boldsymbol{\tau}$ structure | $\widetilde{W}$ argument | Hartree | Fock | Real-space kernel |
|---|---|---|---|---|---|
| **A** density-density | $\tau_1=\tau_2=\tau,\;\tau_3=0$ | $\mathbf{k}_2-\mathbf{k}_1$ | $[\widetilde{W}^{cdab}(\mathbf{0}){+}\widetilde{W}^{abcd}(\mathbf{0})]\bar{G}^{cd}$ | $[W^{cbad}(\mathbf{r}){+}W^{adcb}(-\mathbf{r})]\cdot G^{cd}(\mathbf{r})$ via FFT | $[W(\mathbf{r}){+}W(-\mathbf{r})]\cdot G(\mathbf{r})$ |
| **B** exchange-type | $\tau_1=0,\;\tau_2=\tau_3=\tau$ | $\mathbf{k}_2-\mathbf{k}_3$ | $[W^{cdab}(-\mathbf{r}){+}W^{abcd}(\mathbf{r})]\cdot G^{cd}(\mathbf{r})$ via FFT | $[\widetilde{W}^{cbad}(\mathbf{0}){+}\widetilde{W}^{adcb}(\mathbf{0})]\bar{G}^{cd}$ | $[W(-\mathbf{r}){+}W(\mathbf{r})]\cdot G(\mathbf{r})$ |
| **C** pair-hopping | $\tau_1=\tau_3=\tau,\;\tau_2=0$ | $-(\mathbf{k}_1+\mathbf{k}_3)$ | $[W^{cdab}(-\mathbf{r}){+}W^{abcd}(-\mathbf{r})]\cdot G^{cd}(-\mathbf{r})$ via FFT | $[W^{cbad}(-\mathbf{r}){+}W^{adcb}(-\mathbf{r})]\cdot G^{cd}(-\mathbf{r})$ via FFT | $W(-\mathbf{r})\cdot G(-\mathbf{r})$ |

All three cases reduce the $O(N_k^2 d^4)$ direct summation to $O(N_k\log N_k)$. In Cases A and B the two sub-channels ($\widetilde{W}(\mathbf{q}-\mathbf{k})$ and $\widetilde{W}(\mathbf{k}-\mathbf{q})$) contribute $W(\mathbf{r})$ and $W(-\mathbf{r})$ respectively, so the combined real-space kernel is $[W(\mathbf{r})+W(-\mathbf{r})]\cdot G(\mathbf{r})$. In Case C all channels give $\widetilde{W}(-(\mathbf{k}+\mathbf{q}))$, which maps to $W(-\mathbf{r})\cdot G(-\mathbf{r})$. Any interaction that does not fall into one of these three cases requires the full three-momentum evaluation via `build_Uk` at $O(N_k^2 d^4)$ cost.

---

## Code Implementation

### Array Layout Convention

All 3-tensors storing per-$k$ matrices use **column-major (d, d, Nk)** layout so that each $k$-slice is a contiguous $d\times d$ block in memory:

| Quantity | Shape | Access |
|---|---|---|
| `G_k`, `H_k`, `evecs` | `(d, d, Nk)` | `@view arr[:,:,ki]` |
| `evals` | `(d, Nk)` | `@view evals[:,ki]` |
| `G_taus[n]` | `(d, d)` | direct |

### 1. Preprocessing: Kinetic Term

**`build_Tr(dofs, ops, irvec)`** — Parses one-body operators and builds the real-space hopping table. Returns `(mats, delta)` where `mats[n]` is a $d\times d$ matrix for displacement `delta[n]`.

**`build_Tk(T_r)`** — Returns a **closure** `T_func(k) -> Matrix{ComplexF64}` that evaluates

$$T_{ab}(\mathbf{k}) = \sum_{\mathbf{r}} T_{ab}(\mathbf{r})\,e^{i\mathbf{k}\cdot\mathbf{r}}$$

at any momentum $\mathbf{k}$ on demand. Returns `nothing` if there are no hopping terms.

### 2. Preprocessing: Interaction Term

**`build_Vr(dofs, ops, irvec)`** — Parses two-body operators. Returns `(mats, taus)` where `mats[n]` is a $d\times d\times d\times d$ array for displacement triple `taus[n] = (τ1,τ2,τ3)`.

**`_classify_Vr(V_r)`** — Partitions entries of `V_r` into four cases by $\boldsymbol{\tau}$ structure (priority A > B > C > general). Returns `(A, B, C, general)`, each `(mats, taus)`.

**`build_Wr_A(mats_a, taus_a)`** — Assembles Case A kernels. Returns `(hartree, fock)` where `hartree` is a $(d^2\!\times\!d^2)$ matrix (q-independent) and `fock` is `(mats, delta)` with each kernel a $(d^2\!\times\!d^2)$ matrix.

**`build_Wr_B(mats_b, taus_b)`** — Assembles Case B kernels. Returns `(hartree, fock)` where `hartree` is `(mats, delta)` and `fock` is a $(d^2\!\times\!d^2)$ matrix (q-independent).

**`build_Wr_C(mats_c, taus_c)`** — Assembles Case C kernels, split by physical channel. Returns `(hartree, fock)` where both are `(mats, delta)` with $(d^2\!\times\!d^2)$ kernels. The Hartree channel (terms $W^{cdab}(-r)+W^{abcd}(-r)$) is always applied; the Fock channel ($W^{cbad}(-r)+W^{adcb}(-r)$) is skipped when `include_fock=false`. Both channels contract with $G(-\boldsymbol{\tau}) = G(\boldsymbol{\tau})^\dagger$.

**`build_Vk(V_r)`** — Returns a closure `(k1,k2,k3) -> Array{ComplexF64,4}` for the general three-momentum kernel. Only used for interactions not in cases A/B/C.

**`build_Uk(V_k)`** — Returns a closure `(k,q) -> Matrix{ComplexF64}` of shape $(d^2\!\times\!d^2)$ assembling the antisymmetrized HF matrix (all four Wick channels, Theory §4). Used in the general $O(N_k^2 d^4)$ path.

### 3. k-point Generation

**`build_kpoints(unitcell_vectors, box_size)`** — Generates the uniform $\mathbf{k}$-grid from direct lattice vectors $\{a_i\}$ and supercell size $\{n_i\}$:

$$\mathbf{k} = \sum_i \frac{m_i}{n_i}\,\mathbf{b}_i, \quad m_i \in \{0,\ldots,n_i-1\}$$

where $\mathbf{b}_i$ are reciprocal lattice vectors ($a_i\cdot b_j = 2\pi\delta_{ij}$). Alternatively, any `Vector{Vector{Float64}}` of k-points (e.g. a high-symmetry path) can be passed directly to `solve_hfk`.

### 4. Self-Consistent Field Iteration

**Why direct Fourier sum instead of FFT.** For cases A/B/C, the self-energy is:

$$\Sigma^{ab}(\mathbf{q}) = \sum_{\boldsymbol{\tau}} \left[\sum_{cd} K^{ab,cd}(\boldsymbol{\tau})\,G^{cd}(\boldsymbol{\tau})\right] e^{i\mathbf{q}\cdot\boldsymbol{\tau}}$$

where $G(\boldsymbol{\tau}) = \frac{1}{N_k}\sum_{\mathbf{k}} G(\mathbf{k})\,e^{-i\mathbf{k}\cdot\boldsymbol{\tau}}$. The outer sum runs over at most $N_\tau$ non-zero interaction displacements (typically 2–20), so the total cost is $O(N_\tau N_k d^2)$. FFT would require a regular dual grid of exactly $N_k$ real-space points and cannot handle arbitrary k-point sets (e.g. high-symmetry lines). Direct sum is simpler and more flexible.

**`green_k_to_tau!(G_taus, G_k, kpoints, taus)`** — In-place direct Fourier sum: for each $\boldsymbol{\tau}$ in `taus`, accumulates $G(\boldsymbol{\tau}) = \frac{1}{N_k}\sum_k G(k)\,e^{-ik\cdot\tau}$ into the pre-allocated matrices in `G_taus`. Clears buffers before accumulating.

**`build_heff_k!(H_k, T_k_func, wr_A, wr_B, wr_C, V_k_func, G_k, kpoints, G_taus_buf, g_adj_buf, f_buf, taus_needed, tau_idx; include_fock)`** — Builds $H^\text{eff}(\mathbf{q})$ in-place for all $\mathbf{q}$ simultaneously. Pre-allocated buffers (`G_taus_buf`, `g_adj_buf`, `f_buf`) are reused every SCF iteration. The inner $\mathbf{q}$-accumulation loop is parallelized with `Threads.@threads`. Case breakdown:
- **Case A Hartree**: $\mathbf{q}$-independent; $+\tfrac{1}{2}K_H\cdot\text{vec}(\bar{G})$
- **Case A Fock**: $\mathbf{q}$-dependent; $-\tfrac{1}{2}\sum_\tau K(\tau)\cdot\text{vec}(G(\tau))\cdot e^{iq\cdot\tau}$
- **Case B Hartree**: $\mathbf{q}$-dependent; $+\tfrac{1}{2}\sum_\tau K(\tau)\cdot\text{vec}(G(\tau))\cdot e^{iq\cdot\tau}$
- **Case B Fock**: $\mathbf{q}$-independent; $-\tfrac{1}{2}K_F\cdot\text{vec}(\bar{G})$
- **Case C Hartree** (always): $+\tfrac{1}{2}\sum_\tau K_H(\tau)\cdot\text{vec}(G(\tau)^\dagger)\cdot e^{iq\cdot\tau}$
- **Case C Fock** (if `include_fock`): $-\tfrac{1}{2}\sum_\tau K_F(\tau)\cdot\text{vec}(G(\tau)^\dagger)\cdot e^{iq\cdot\tau}$
- **General**: $O(N_k^2 d^4)$ via `build_Uk`

**`diagonalize_heff_k(H_k)`** — Diagonalizes $H^\text{eff}(k)$ at each k-point with `Threads.@threads`. Returns `(evals, evecs)` of shapes `(d, Nk)` and `(d, d, Nk)`.

**`find_chemical_potential_k(evals, n_electrons, temperature)`** — Finds $\mu$ by midgap rule ($T=0$) or bisection on the global Fermi-Dirac sum ($T>0$).

**`update_green_k(evecs, evals, mu, temperature)`** — Constructs

$$G^{ab}(\mathbf{k}) = \sum_n f(\varepsilon_{n\mathbf{k}}-\mu)\,u_{a,n}^*(\mathbf{k})\,u_{b,n}(\mathbf{k})$$

In matrix form: $G(k) = \overline{V\,\mathrm{diag}(f)\,V^\dagger}$ where $V$ is the eigenvector matrix. Returns shape `(d, d, Nk)`.

**`calculate_energies_k(evals, mu, temperature, G_k, H_k, T_k_func, kpoints)`** — Returns `(band, interaction, total)` energies:
$$E_\text{band} = \frac{1}{N_k}\sum_{k,n}\varepsilon_{kn}\,f_{kn}, \qquad E_\text{int} = -\frac{1}{2N_k}\sum_k \mathrm{Re}\,\mathrm{Tr}\!\left[(H^\text{eff}(k)-T(k))\,G(k)\right]$$

### 5. Public API

**`solve_hfk(dofs, onebody, twobody, kpoints, n_electrons; kwargs...)`** — Public entry point.

**Arguments:**
- `dofs::SystemDofs`: Internal DOFs of one magnetic unit cell.
- `onebody`, `twobody`: Results of `generate_onebody`/`generate_twobody`.
- `kpoints::Vector{Vector{Float64}}`: k-points (use `build_kpoints` for uniform grids, or provide any custom set).
- `n_electrons::Int`: Total electron count across all k-points.

**Keyword arguments:** `temperature`, `max_iter`, `tol`, `diis_m`, `G_init`, `ene_cutoff`, `n_restarts`, `seed`, `include_fock`, `verbose`.

**Returns** `NamedTuple` with: `G_k` (shape `(d,d,Nk)`), `eigenvalues` (shape `(d,Nk)`), `eigenvectors` (shape `(d,d,Nk)`), `energies` (`(band, interaction, total)`), `mu`, `kpoints`, `converged`, `iterations`, `residual`, `ncond`.

Runs full SCF with DIIS mixing and multi-restart (lowest-energy converged result selected). Preprocessing (`build_Tr`, `build_Vr`, classification, kernel assembly) is done once before all restarts.
