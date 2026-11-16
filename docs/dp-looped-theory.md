# Differential Privacy Meets Looped Transformers: Clipping Turns Stability from a Maximum Condition into a Quantile Condition

Working notes, v0.4.1 (16 Nov 2026). Canonical copy: `docs/dp-looped-theory.md` in github.com/StephCpa/DP-looped-Transformer; paths below are repository paths.

**Scope of the claim.** These are interim results of mechanism research. They do not support "DP generally harms looped transformers". The claim is two statements:

1. The privacy accounting of DP-SGD does not change with loop depth (Prop 1).
2. The loop-specific risk comes from per-sample gradient clipping: it changes what keeps a loop stable from the *maximum* over training samples to a *quantile* of them (Theorem 2). This is shown for linear in-context-gradient-descent loops, by fixed-point analysis and, at small scale, in trained models.

Everything else here is supporting analysis, a toy observation, or an open question, and is tagged as such in the table below.

**Trained-model pilot: complete, 120 of 120 runs (`pilot/PREREG_PILOT.md`).** Three seeds per cell, so every entry is weak evidence.

| Preregistered test | Result |
|---|---|
| E1: clipping erodes stability in a trained linear loop, $T=16$ | **Passed.** Unstable fraction 0.03–0.09% → 4.6–5.0%; +4.74 pp [+4.13, +5.36]. Holds under a finite-horizon definition too (+5.46 pp) |
| E1, clause "larger at $T=16$ than at $T=4$" | **Failed**; mis-specified against §15 |
| E3: token normalization reduces the erosion | **Passed.** 4.74 → 0.75 pp |
| E2: no clipping penalty in the softmax block, $T=4$ | **Passed.** Ratio 1.03 [0.92, 1.15] |
| E2 at $T=16$ | **Failed the predicted band, in the favourable direction** (ratio 0.51: clipping helped), and **not interpretable**: the arms selected different learning rates |
| E4: extrapolation drift under noise | **Failed; kill rule triggered.** $D_{2T}/D_T=0.67$ [0.26, 1.70]. A1 now has toy support only |

An audit prompted by the review found no silently zeroed gradients in any audited or new run (Amendment 1 in the preregistration).

**Decisions of 16 Nov.** E4 is formally closed (D7). E2 at $T=16$ is read only as "clipping permits a larger learning rate". The next experiment is Stage A alone, preregistered in `stageA/PREREG_STAGE_A.md`: a screening grid (A0) and six confirmatory contrasts on ten new seeds (A1), with the T-step gain as the primary metric. Stage B (noise) is postponed; Stage C (softmax blocks) will be preregistered separately after A1.

v0.4 changes, in response to the review of 16 Nov (`docs/review-2026-11-16-response.md`): title and claim narrowed; statement-status table added; Prop 7 restated as proven only for the isotropic linear loop; withdrawn H2 and P3 moved to Appendix A; §17 added to separate the softmax toy from the pilot's softmax block; paths changed to repository paths.

## Statement status

Tags: **proven** (complete argument in this document), **proof sketch** (argument outlined, steps missing), **conjecture**, **numerical observation** (seen in a toy or the pilot, no proof).

| Statement | Status | Scope or gap |
|---|---|---|
| Prop 1 (privacy independent of the loop) | proven | standard DP-SGD argument |
| Lemma 1, Lemma 1′, Prop 2 | proven | under A1–A2 |
| Lemma 2 (gradient norm) | proven | |
| Lemma 3 (smoothness) | proof sketch | constants not checked line by line; consistent with the numerics |
| Theorem 1 (utility bound) | proven, conditional on Lemma 3 | upper bound only; no lower bound |
| Prop 3 (coherence factor) | proven | first order in the perturbation |
| Prop 4 (tied vs untied) | proven | compares worst-case bounds, not achieved performance |
| H1 (clipping as normalization) | conjecture | heuristic progress ratio |
| Prop 5 (equipartition) | proven | local quadratic model only |
| Prop 6 (flatness = forward robustness) | proven | squared loss at interpolation |
| C1 (noise prefers contractive loops) | conjecture | two-parameter toy support |
| Prop 7 (hidden-state noise) | proven for the isotropic linear loop; proof sketch otherwise | see §10 for the four missing steps |
| Prop 8 (multistability obstruction) | proven | |
| Prop 9 (gradient profile) | proven | scalar-prompt model |
| Theorem 2 (maximum to quantile) | proven for (i), (ii) | scalar prompt, scalar parameter; (iii) is a remark; the matrix case is numerical only |
| Prop 10 (token-normalized softmax loop) | proven | the specific convex-combination block of §16 |
| Prop 11 (rate vs fixed-point parameters) | proof sketch | asymptotic; assumes convergence to a hyperbolic fixed point |
| Extrapolation drift (formerly A1, now D7) | closed | two-parameter toy only; pilot E4 failed, kill rule fired; not a finding |
| Stability erosion in trained loops (A2) | numerical observation | pilot E1, three seeds |
| Softmax-loop robustness to clipping (A6) | numerical observation | toy of §16 only; the pilot's softmax block is a different model (§17) |

## 0. Verdict

**Dead**

- **D1. "DP noise is continuously amplified along the loop."** DP-SGD puts no noise in the loop. If you do inject hidden-state noise into a contractive loop, it is exactly output perturbation in the linear case (Prop 7).
- **D2. "DP-SGD proofs cannot be reused; contractivity is required."** The privacy proof transfers verbatim (Prop 1). The utility proof transfers with constants polynomial in $S_T=\sum_{k<T}\rho^k$ (Theorem 1). Even at $\rho=1$ the cost is $T^{5/2}$, not exponential; only $\rho>1$ is exponential.
- **D3. "In-horizon fixed-point drift grows with loop instability."** At stationarity, the excess loss at the trained loop count is $\approx \eta s^2/4$ per constrained direction, independent of $\rho$ and $T$ (Prop 5; toy check within 2%).
- **D4. "Weight tying is a DP liability because the same noise adds up coherently."** Coherence multiplies signal and noise by the same factor (Prop 3). At the bound level tied is never worse than untied (Prop 4).
- **D5 (mine, v0.1). "Clipping down-weights the slowly contracting samples" (H2) as a general claim.** It depends on what the parameter does (Prop 11). Where the parameter sets only the *rate* of the loop (step size in the linear loop, §15), slow prompts have bounded gradients and the clipped prompts are the expansive ones. Where the parameter *moves the fixed point* (attention temperature in the softmax loop, §16), gradients do track $1/(1-\rho_i)$, but the resulting clipping cost is at most 8% in the toy.
- **D6 (mine, v0.2). "A convex combination cannot expand."** A softmax update bounds the state, not the Jacobian: its norm reaches 6.7 in the toy (§16).
- **D7. Extrapolation drift as a finding (closed 16 Nov by decision).** The pilot's test of it, E4, failed and its kill rule fired. It is not to be treated as a research finding. What remains is a two-parameter toy observation (§9), kept as a record: DP noise diffuses freely along directions the training loss does not pin at $T_{\text{train}}$. Those directions control behaviour at $T_{\text{test}}\neq T_{\text{train}}$. Toy: prediction variance at $2T$ is 430× larger for a marginal solution than a contractive one and grows linearly in training steps (§9). This is the precise form of "fixed-point drift".

**Alive**

- **A2. Clipping erodes the stability margin (Theorem 2, §15).** Stability of a loop is enforced by the exponentially large gradients of the few samples on which it is expansive. Clipping caps exactly those, so the restoring force becomes a count instead of a magnitude. In the in-context regression loop the fraction of unstable training prompts at $T=16$ rises from 0.03–0.08% to 2.1–2.4% (four draws), and the test-loss ratio (clipped fixed point over unclipped) is about 1.03 at $T=1$, 2.2–3.0 at $T=4$ and $5\times10^5$–$6\times10^7$ at $T=16$ (five draws). The effect survives Frobenius-norm clipping of a full matrix preconditioner, weakened. This is the strongest loop-specific DP effect found so far, and it is the closest thing to the original "DP worsens stability" intuition, through clipping, not noise.
- **A6. What makes a loop robust to clipping (§15–16; toy-level, see §17 for what the pilot does and does not test).** Three separate ingredients, in increasing cost:
  - *Token normalization* makes the stability region depend on the parameters only, not on the data.
  - *A bounded state* (convex-combination update) bounds the loss, so a shifted fixed point has bounded cost. In the normalized softmax loop the clipped/unclipped loss ratio is 0.99–1.01 at the median threshold for $T$ up to 64 (five draws), against $10^5$–$10^7$ for the raw linear loop at $T=16$.
  - *A-priori contraction* ($\beta R^2<1$) gives the gradient bound clipless DP needs, but costs 3.1–3.3× in loss in the toy, because the trained temperature sits in the multistable regime.
- **A7. Rate parameters are the fragile ones (Prop 11).** A parameter that only sets how fast the loop converges produces exponentially heterogeneous gradients; one that moves the fixed point produces polynomially heterogeneous ones. Non-private training is also fragile in the first case: at $T=64$ the unclipped raw loop is $1.6\times10^6$ worse than the population optimum (§15).
- **A3. Curvature cost.** Smoothness scales as $S_T^3$, forcing small steps; the utility bound scales as $S_T^{5/2}$ (§5).
- **A4. Memory.** Integrator-style memory has a horizon $\propto 1/\text{noise}$; attractor-style memory has a noise threshold and no horizon (§8).
- **A5. Clipless DP** needs contraction enforced by parametrization, not by a regularizer (§2).

**Corrections to my 11 Nov reply**

1. I called tied-vs-untied "an empirical question" because coherent noise and dimension "pull in opposite directions". Props 3–4 settle the bound-level question in favour of tying; what remains empirical is the coherence ratio in H1.
2. I said DP "scales the existing instability by $1/(1-\rho)$". In-horizon that is wrong at stationarity (Prop 5); the factor enters through curvature and clipping instead.
3. I presented privacy amplification by iteration as a potential "positive result". The privacy cost does stop growing in $T$ for the linear loop (Prop 7; only sketched for nonlinear loops), but it buys nothing over output perturbation in the linear case, and it excludes multistable memory (Prop 8).
4. I listed "go clipless" without the a-priori-enforcement caveat (A5).

**Corrections to v0.1 (11 Nov)**

5. H2 and its "concrete instance" in §7 predicted that DP clipping down-weights ill-conditioned prompts, with gradients scaling as $S_T(\rho_i)$. Wrong: the per-sample gradient is residual × sensitivity, and the residual decays as $\rho_i^T$, so the Lemma 2 bound is not attained on the contracting side. Prediction P3 was built on H2 and is withdrawn (§12).
6. §14 called O1 "unoccupied". The single-step ($T=1$), noise-side version is published (Bonnerjee et al.); only the looped, clipping-side version is open as far as I found.

**Corrections to v0.2 (15 Nov)**

7. I wrote that softmax attention is safe because "a convex combination cannot expand". Wrong: it bounds the state, not the sensitivity (D6). With unnormalized tokens the Jacobian norm is 10.7 and the max/median gradient ratio grows from 47 to 3,700 between $T=4$ and $T=64$.
8. I declared H2 dead outright. Its premise (gradient magnitude tracks $1/(1-\rho_i)$) holds in the softmax loop, rank correlation 0.85 for $T\ge16$. What fails is the generality and, there, the size of the consequence (D5).
9. I quoted $R_C=4.6\times10^5$ at $T=16$ as if it were a stable number. It is dominated by a handful of test prompts: five draws give $4.6\times10^5$, $5.7\times10^5$, $8.7\times10^5$, $6.8\times10^6$ and $6.3\times10^7$. The unstable fraction is the stable statistic (2.1–2.4%).

**Disclosure (v0.3).** The first version of the softmax toy had no input injection. At $T=64$ both the clipped and the unclipped run stalled at near-uniform attention, and I briefly read the clipped stall as a clipping effect ($R_C\approx4$). It was the loop forgetting its query (Prop 2: two different queries give final states equal to $3\times10^{-14}$). That version is discarded; all §16 numbers use input injection.

## 1. Setup

Block $F:\mathcal Z\times\mathcal X\times\Theta\to\mathcal Z$, loop $z_{t+1}=F(z_t,x;\theta)$, $t=0,\dots,T-1$, loss $\ell(\theta)=g(z_T)$ with $g(z)=\ell_0(h(z),y)$. Tied parameters $\theta\in\mathbb R^d$; the untied comparator has $(\theta_0,\dots,\theta_{T-1})\in\mathbb R^{Td}$.

Write $J_t=\partial_z F(z_t)$, $B_t=\partial_\theta F(z_t)$, $\Phi_{T,s}=J_{T-1}\cdots J_s$, $A_t=\Phi_{T,t+1}B_t$, $M_T=\sum_{t<T}A_t$, $v=\nabla g(z_T)$.

Assumptions, on a forward-invariant set $K\subseteq\mathcal Z$ and parameter set $\Theta$:

- **A1** $\|J_z\|\le\rho$.
- **A2** $\|J_\theta\|\le L_\theta$.
- **A3** $J_z, J_\theta$ are Lipschitz: constants $\beta_{zz},\beta_{z\theta}$ for $J_z$ in $(z,\theta)$ and $\beta_{\theta z},\beta_{\theta\theta}$ for $J_\theta$.
- **A4** $g$ is $G$-Lipschitz and $H$-smooth.

$\rho$ is any positive number unless stated; $S_T=\sum_{k<T}\rho^k$, $S_T^{(2)}=\sum_{k<T}\rho^{2k}$. For $\rho<1$, $S_T\le 1/(1-\rho)=:T_{\text{eff}}$.

Two threat models, kept separate throughout:

- **Setting A**: training-data privacy via DP-SGD on $\theta$. Noise lives in parameter space.
- **Setting B**: privacy of the input or context $x$ at inference via noise on hidden states, $z_{t+1}=F(z_t,x)+\xi_t$, releasing only $z_T$.

## 2. Privacy does not depend on the loop

**Prop 1.** DP-SGD with per-sample clipping at $C$ and noise $\sigma_{\text{DP}}C$ gives the same $(\varepsilon,\delta)$ for a looped model as for any other, for every $T$ and $\rho$. Evaluating at any $T_{\text{test}}$ is post-processing.

*Proof.* The guarantee uses only $\|\text{clip}(g_i)\|\le C$. ∎

**Clipless variant.** By Lemma 2, if A1, A2, A4 hold for *every* $\theta$ the optimizer can reach and *every* possible record, then $\|g_i\|\le GL_\theta S_T$ and clipping at that value is a no-op. A Jacobian penalty or a measured $\hat\rho$ does not qualify: the bound must hold by construction (spectral parametrization, bounded state), otherwise the sensitivity claim is false and so is the DP claim.

## 3. Forward sensitivity

**Lemma 1.** For $\theta,\theta'=\theta+\delta$ in $\Theta$: $\|z_T(\theta')-z_T(\theta)\|\le L_\theta S_T\|\delta\|$.

*Proof.* $e_{t+1}=[F(z'_t;\theta')-F(z_t;\theta')]+[F(z_t;\theta')-F(z_t;\theta)]$, so $\|e_{t+1}\|\le\rho\|e_t\|+L_\theta\|\delta\|$, $e_0=0$. ∎

Tight for the scalar linear loop $z_{t+1}=az_t+bx$ perturbed in $b$.

**Lemma 1′ (trajectory-wise local contraction).** Fix $\theta$. Suppose that for all $t$ and all $\|\theta'-\theta\|\le r_\theta$, $F(\cdot,x;\theta')$ is $\rho$-Lipschitz ($\rho<1$) on the ball $B(z_t(\theta),r)$ and $L_\theta$-Lipschitz in $\theta$ there. If $\|\delta\|\le\min\{r_\theta,\ r(1-\rho)/L_\theta\}$ then $\|e_t\|\le L_\theta\|\delta\|/(1-\rho)\le r$ for all $t$, uniformly in $T$.

*Proof.* Induction: if $\|e_t\|\le r$ both points lie in the ball, so the Lemma 1 recursion applies and keeps $\|e_{t+1}\|\le L_\theta\|\delta\|/(1-\rho)\le r$. ∎

Lemma 1′ is the usable form. It needs contraction only transversally to the nominal trajectory, so it covers loops that move between several attractors (finite-state computation). Above the threshold $\|\delta\|>r(1-\rho)/L_\theta$ it says nothing: the perturbed trajectory can leave the tube and land in another basin.

**Prop 2 (effective depth).** Under global A1 with $\rho<1$, $\|z_T-z^\ast\|\le\rho^T\|z_0-z^\ast\|$. Iterations beyond $T_\epsilon\approx T_{\text{eff}}\log(D/\epsilon)$ change the output by less than $\epsilon$. A globally contractive loop is a depth-$T_{\text{eff}}$ network with $d$ parameters (cf. Miller & Hardt on stable recurrent models). Contraction does not remove the cost of depth; it caps the depth you pay for.

## 4. Gradient structure

Adjoint: $\lambda_T=v$, $\lambda_t=J_t^\top\lambda_{t+1}$, and $\nabla_\theta\ell=\sum_{t<T}B_t^\top\lambda_{t+1}=M_T^\top v$.

**Lemma 2.** $\|\nabla_\theta\ell\|\le GL_\theta S_T$.

*Proof.* $\|\lambda_{t+1}\|\le G\rho^{T-t-1}$; sum. ∎

**Lemma 3.** $\ell$ is $\beta_\ell$-smooth in $\theta$ with
$$\beta_\ell\le HL_\theta^2S_T^2+G\big[\beta_{\theta\theta}S_T+(\beta_{\theta z}+\beta_{z\theta})L_\theta S_T^2+\beta_{zz}L_\theta^2S_T^3\big].$$

*Proof sketch.* With $k=T-t-1$ and $c=\beta_{zz}L_\theta S_T+\beta_{z\theta}$:
$\|\Delta\lambda_T\|\le HL_\theta S_T\|\delta\|$ and $\|\Delta\lambda_t\|\le\rho\|\Delta\lambda_{t+1}\|+c\,G\rho^{T-t-1}\|\delta\|$, giving $\|\Delta\lambda_{t+1}\|\le\rho^kHL_\theta S_T\|\delta\|+k\,cG\rho^{k-1}\|\delta\|$. Then $\|\Delta\nabla\ell\|\le\sum_t\big[L_\theta\|\Delta\lambda_{t+1}\|+(\beta_{\theta z}L_\theta S_T+\beta_{\theta\theta})G\rho^k\|\delta\|\big]$, using $\sum_k\rho^k=S_T$ and $\sum_kk\rho^{k-1}\le S_T^2$. ∎

The $S_T^3$ leading term matches the $\kappa^3$ smoothness of the hyperobjective in bilevel optimization (Ghadimi & Wang), with $T_{\text{eff}}$ in the role of the condition number.

## 5. Utility

**Theorem 1 (clipless DP-GD).** Let A1–A4 hold for all $\theta\in\mathbb R^d$ and all records. Run full-batch GD for $K$ steps with step $1/\beta_\ell$ and Gaussian noise of std $s=2L_\ell\sqrt K/(n\mu)$ per coordinate on the mean gradient, $L_\ell=GL_\theta S_T$. The output is $\mu$-GDP (replace-one adjacency) and
$$\frac1K\sum_{k<K}\mathbb E\|\nabla\hat L(\theta_k)\|^2\le\frac{2\beta_\ell\Delta_0}{K}+\frac{4dKL_\ell^2}{n^2\mu^2},\qquad\text{optimized over }K:\ \le\frac{4\sqrt2\,L_\ell\sqrt{\beta_\ell\Delta_0d}}{n\mu}.$$

*Proof.* Sensitivity of the mean gradient is $2L_\ell/n$; $K$-fold Gaussian composition is $\mu$-GDP with $\mu=2L_\ell\sqrt K/(ns)$. Descent lemma with $\eta=1/\beta_\ell$: $\mathbb E\hat L(\theta_{k+1})\le\hat L(\theta_k)-\frac1{2\beta_\ell}\|\nabla\hat L\|^2+\frac{ds^2}{2\beta_\ell}$. Telescope and optimize $K$. ∎

**Reading.** $L_\ell\propto S_T$ and $\beta_\ell\propto S_T^3$, so the bound is $\propto S_T^{5/2}\sqrt d/(n\mu)$ with $K^\ast\propto S_T^{1/2}$.

| regime | $S_T$ | bound scales as |
|---|---|---|
| contractive, $\rho<1$ | $\le T_{\text{eff}}$ | $T_{\text{eff}}^{5/2}$, uniform in $T$ |
| marginal, $\rho=1$ | $T$ | $T^{5/2}$ |
| expansive, $\rho>1$ | $\approx\rho^T/(\rho-1)$ | $\rho^{5T/2}$ |

Caveats: this is an upper bound (no matching lower bound, open problem O2); gradient-norm stationarity is not scale-free across parametrizations; $d$ is the block size, not $Td$.

## 6. Tied versus untied

**Prop 3 (coherence factor).** To first order in the perturbation:

- (a) Tied, $\delta\sim N(0,s^2I_d)$: $\mathbb E\|\Delta z_T\|^2=s^2\|\sum_tA_t\|_F^2$. Untied, independent $\delta_t$: $s^2\sum_t\|A_t\|_F^2$. The ratio $\kappa_T\in[0,T]$.
- (b) Tied gradient $\sum_tA_t^\top v$; untied gradient has blocks $A_t^\top v$. Squared-norm ratio $\kappa_T(v)\in[0,T]$.
- (c) $\nabla^2_{\text{tied}}=P^\top\nabla^2_{\text{untied}}P$ with $P=\mathbf 1_T\otimes I_d$, $\|P\|^2=T$.

So the mechanism that makes tied noise add coherently makes the tied gradient add coherently by the same matrix $M_T$.

**Prop 4.** Worst-case (sensitivity × √dimension), the quantity that enters Theorem 1: tied $GL_\theta S_T\sqrt d$, untied $GL_\theta\sqrt{S_T^{(2)}Td}$. Ratio $S_T/\sqrt{TS_T^{(2)}}\le1$ by Cauchy–Schwarz, with equality only at $\rho=1$, and $\to\sqrt{(1+\rho)/((1-\rho)T)}$ as $T\to\infty$.

| $\rho$ | $T=10$ | $T=100$ | $T=1000$ |
|---|---|---|---|
| 0.5 | 0.55 | 0.17 | 0.05 |
| 0.9 | 0.96 | 0.44 | 0.14 |
| 0.99 | 1.00 | 0.96 | 0.45 |

**H1 (clipping as normalization; heuristic).** DP noise std is $\sigma_{\text{DP}}C/B$ and $C$ tracks the per-sample gradient scale, so a coherent gain in gradient norm is cancelled by the threshold. In the noise-dominated regime the per-step progress is $\|\bar g\|^4/(2\beta ds^2)$, and
$$\frac{\text{progress}_{\text{tied}}}{\text{progress}_{\text{untied}}}=T\cdot\frac{\kappa'^2}{\kappa\,\kappa_H},$$
with $\kappa$ the per-sample gradient coherence, $\kappa'$ the mean-gradient coherence, $\kappa_H$ the curvature coherence. If the three agree, tying wins by $T$. Tying loses only when $\kappa'^2<\kappa\kappa_H/T$: per-sample gradients coherent across loop steps while the population gradient is not. This is the same normalization mechanism as in the DP-GRPO clipping question.

Numerically, random tanh loops have $\kappa_T\in[0.76,1.37]$, i.e. incoherent. Trained loops that implement an iterative algorithm should be much more coherent; this is the first thing to measure.

Empirical caution: for *embedding* tying under DP-SGD the two papers I found disagree (Ding et al. report sharing is essential; El Mais et al. report untied is better on GPT-2). Neither is depth tying.

## 7. Clipping bias: the general bound

Per-sample gradient $g_i=M_T(x_i)^\top v_i$ with $\|g_i\|\le\|v_i\|L_\theta S_T(\rho_i)$, where $\rho_i$ is the contraction rate along sample $i$'s trajectory. Clipping weights $w_i=\min(1,C/\|g_i\|)$; bias $\|\bar g_{\text{clip}}-\bar g\|\le\frac1n\sum_i(\|g_i\|-C)_+$.

Known results to lean on: no bias when the gradient distribution is symmetric (Chen, Wu, Hong); otherwise clipped SGD reaches only a neighbourhood governed by the gradient spread (Koloskova et al.). Here the spread is bounded by $G^2L_\theta^2\,\mathbb E[S_T(\rho_i)^2]$, so the controlling statistic is the second moment of $S_T(\rho_i)$ across samples.

Which samples exceed the threshold depends on what the parameter does (Prop 11, §16). For a rate parameter it is the samples on which the loop expands (Theorem 2, §15). The original hypothesis H2 was wrong about this; it is in Appendix A.

## 8. Memory

- **Integrator.** $z_{t+1}=az_t$ holds a value only at $a=1$. With $a=1+\epsilon$ the relative error after $T$ steps is $\approx T|\epsilon|$. Parameter noise of scale $s_\theta$ gives a usable horizon $T_{\max}\approx\tau/s_\theta$ at tolerance $\tau$.
- **Attractor.** $z_{t+1}=\tanh(gz_t)$, $g>1$, has stable fixed points $\pm z^\ast$ with local slope $\rho_{\text{loc}}=g(1-z^{\ast2})<1$. Perturbing $g$ shifts them by $z^\ast(1-z^{\ast2})\delta/(1-\rho_{\text{loc}})$ with no accumulation. The bit survives any $T$ while $g+\delta>1$.

Lemma 1′ generalizes the second case. Under DP a loop that must carry state should store it in locally contracting attractors with margin $r\gg s_\theta L_\theta/(1-\rho_{\text{loc}})$; degradation is then a threshold in noise, not a slope in $T$. The cost: sharp attractors mean large gain away from them, hence larger gradients for samples near basin boundaries, which clipping then caps (§15).

## 9. Where the DP noise goes

Here $s=\sigma_{\text{DP}}C/B$ is the per-step noise std on the mean gradient.

**Prop 5 (equipartition, local quadratic model).** Near a minimum with Hessian eigenpairs $(h_j,u_j)$, noisy GD has stationary variance $\eta s^2/(h_j(2-\eta h_j))$ along $u_j$ for $h_j>0$, and excess loss $\sum_{h_j>0}\eta s^2/(2(2-\eta h_j))\approx(\eta s^2/4)\,\text{rank}(H)$. Along flat directions the variance is $K\eta^2s^2$, unbounded in $K$.

**Prop 6 (flatness is forward robustness).** For squared loss on $z_T$ at interpolation, $\nabla^2\hat L=\frac1n\sum_iM_T(x_i)^\top M_T(x_i)$, so $\text{tr}\,\nabla^2\hat L$ equals the first-order output noise power per unit isotropic parameter noise.

**Consequence.** Output noise power at loop count $T'$ is $\sum_j\text{Var}_j\|M_{T'}u_j\|^2$.

- At $T'=T$ on the training distribution, curvature and sensitivity cancel: $\approx(\eta s^2/2)\,\text{rank}(H)$, independent of $\rho$ and $T$. The loop does not amplify stationary DP noise in-horizon.
- At $T'\neq T$, a direction flat at $T$ contributes $K\eta^2s^2\|M_{T'}u_{\text{flat}}\|^2$, which no curvature restrains.

**Toy check.** $z_{t+1}=az_t+bx$, loss $\frac12(bS_T(a)-c)^2$, $T=10$. The zero-loss valley $bS_T(a)=c$ contains contractive and marginal solutions; the prediction at $T'$ is $cS_{T'}(a)/S_T(a)$.

| start | in-horizon excess loss (pred.) | variance of prediction at $2T$, $K=500\to4000$ |
|---|---|---|
| $a_0=0.3$ | $3.2\times10^{-6}$ ($3.16\times10^{-6}$) | flat at $6.3\times10^{-6}$ |
| $a_0=0.9$ | $3.65\times10^{-6}$ ($3.66\times10^{-6}$) | $3.6\times10^{-4}\to2.7\times10^{-3}$, linear in $K$ |

**C1 (conjecture: noise prefers contractive loops).** Isotropic noise drifts along the valley toward lower transverse curvature, which by Prop 6 and Lemma 2 means smaller $\|M_T\|_F$, i.e. more contraction. In the toy the mean of $a$ moves by $-0.0109\pm0.0006$ (from 0.9) and $-0.0033\pm0.0005$ (from 0.3) over 16k steps. The sign is right, but the drift is about 7× below the diffusion (sd 0.077), so it will not rescue extrapolation at realistic privacy budgets.

## 10. Hidden-state noise (Setting B)

**Prop 7 (privacy cost of hidden-state noise in a contractive loop).** *Status: proven for the isotropic linear loop; a candidate upper bound, with a proof sketch only, for general linear and nonlinear loops.*

Setting, stated precisely:

- State $z_t\in\mathbb R^m$, $z_0$ fixed and data-independent. $K\subseteq\mathbb R^m$ closed and convex, $\Pi_K$ the Euclidean projection.
- Update with input injection and noise placed after the block: $z_{t+1}=F(\Pi_K(z_t),x)+\xi_t$, $\xi_t\sim N(0,\sigma^2I_m)$ independent. The private input $x$ enters every step.
- $F(\cdot,x)$ is $\rho$-Lipschitz on $K$ with $0\le\rho\le1$, for every $x$. Then $\psi_x=F(\Pi_K(\cdot),x)$ is $\rho$-Lipschitz on all of $\mathbb R^m$, which is what the noisy iterates need.
- Adjacent $x,x'$: $\sup_{z\in K}\|F(z,x)-F(z,x')\|\le s_x$.
- Only $z_T$ is released.

Claim: the release is $(\alpha,\varepsilon_\alpha)$-RDP with
$$\varepsilon_\alpha\le\frac{\alpha s_x^2}{2\sigma^2}\,\Gamma_T,\qquad\Gamma_T=\frac{S_T^2}{S_T^{(2)}}=\frac{(1+\rho)(1-\rho^T)}{(1-\rho)(1+\rho^T)}\le\min\Big\{T,\frac{1+\rho}{1-\rho}\Big\}\quad(\Gamma_T=T\text{ at }\rho=1).$$

**(a) Proven: isotropic linear loop.** $F(z,x)=\rho z+b(x)$, $K=\mathbb R^m$. Then $z_T$ is Gaussian with mean shift $S_T\,(b(x)-b(x'))$ between adjacent inputs and covariance $\sigma^2S_T^{(2)}I$, so its Rényi divergence is exactly $\alpha\|b(x)-b(x')\|^2S_T^2/(2\sigma^2S_T^{(2)})$. The bound holds with equality at $s_x=\|b(x)-b(x')\|$.

**(b) Proof sketch only: general case.** Shifted-Rényi argument of Feldman et al. with a contraction-reduction step: track a shift $z_t=\rho z_{t-1}+s_x-a_t$, pay $\frac{\alpha}{2\sigma^2}\sum_ta_t^2$. Because $x$ is injected at every step, the shift is fed at every step. Minimizing $\sum_ta_t^2$ subject to $z_T=0$ gives $a_t\propto\rho^{T-t}$ and the value above; feasibility $z_t\ge0$ reduces to $\rho^{T-t}\le1$ (checked numerically for the schedule, `checks/check_theory.py`).

What a complete proof still has to supply:

1. The contraction-reduction lemma for vector-valued states and a general nonlinear $\rho$-Lipschitz $\psi$, in the form $D_\alpha^{(\rho z+s)}(\psi(X)\,\|\,\psi'(X'))\le D_\alpha^{(z)}(X\,\|\,X')$ when $\sup\|\psi-\psi'\|\le s$. I have not written this out; I am relying on the published non-expansive version and its strongly-convex extension.
2. The shift-reduction lemma for Gaussian noise in $\mathbb R^m$ with the stated noise placement.
3. That the control sequence $a_t$ is admissible (non-negative shifts, terminal shift zero) for every $T$ and $\rho\in[0,1]$; only the closed-form schedule has been checked, numerically.
4. That projecting before the block, as defined above, is the right construction, and what changes if the noise is added before the projection.

Until then, read "the privacy cost stops growing with loop depth" as established for the isotropic linear loop and conjectured elsewhere. Naive composition gives $\Gamma_T=T$; at $\rho=0.9$, $\Gamma_{100}=19.0$.

**No free lunch.** In the linear case $z_T$ is the noiseless output plus $N(0,\sigma^2S_T^{(2)})$, which is output perturbation at sensitivity $s_xS_T$: identical privacy, identical noise. In-loop noise can only help through nonlinearity.

**Prop 8 (multistability obstruction).** If $F(\cdot,x)$ has two distinct fixed points its Lipschitz constant is $\ge1$. So any loop whose answer depends on the initial state or on which attractor it enters is outside Prop 7.

Together: T-uniform hidden-state privacy, denoising attractors, and state-dependent memory cannot all hold under current tools. Privacy for locally contractive noisy loops is open (O4); the non-convex convergent bounds of Asoodeh & Diaz are the nearest tool.

## 11. Does a transformer block satisfy the assumptions?

- **Vanilla looped block** (Yang et al.: $Z_{t+1}=\text{TF}_\theta(Z_t+P)$ with an internal residual stream): Jacobian $I+\partial(\text{attn}+\text{mlp})$, so $\rho\approx1$ or above by construction. A1 fails globally; this is the $S_T=T$ row of the table.
- **Getting $\rho<1$**: damped update $Z_{t+1}=(1-\gamma)Z_t+\gamma G_\theta(Z_t,P)$ gives $\rho=1-\gamma(1-\rho_G)$ and needs $\text{Lip}(G)=\rho_G<1$.
- **$\text{Lip}(G)$**: dot-product attention is not globally Lipschitz (Kim et al.). On bounded tokens it is, with a constant growing in the radius, the key–query norm and, in the worst case, sequence length (Castin et al., from memory). A norm-clipped or normalized state supplies the bounded set $K$.
- **$L_\theta$, $\beta$'s**: finite and polynomial in the radius and weight norms on $K$. No obstruction, large constants.

Realistic target: global A1 for a real block needs architectural changes that probably cost expressivity. Use Lemma 1′ with measured local $\hat\rho$ for utility explanations, and keep clipping for privacy.

## 12. Predictions and kill criteria

Ratios throughout; seed is the unit.

| # | prediction | kill criterion |
|---|---|---|
| P1 | DP-induced loss increase at $2T_{\text{train}}$ divided by that at $T_{\text{train}}$ is $\gg1$ and grows with training steps for fixed-$T$ training; $\approx1$ for random-$T$ training | ratio within seed CI of 1 for fixed-$T$ models with $\hat\rho\ge0.9$ |
| P2 | noise-only arm (no clipping), matched $\eta s^2$: in-horizon excess loss ratio across $\hat\rho$ bins $\approx1$ | excess loss scales like $S_T^2$ |
| P3′ | clip-only arm, trained to convergence: the fraction of training samples with $\hat\rho_i>1$ is at least 3× the unclipped fraction, and the test-loss ratio clipped/unclipped grows with $T$ | ratio $\le1.5$ at $T\ge16$ |
| P6 | the P3′ ratio stays $\approx1$ in $T$ for blocks with normalized updates (softmax attention, normalized state) and grows for linear-attention or unnormalized residual loops. *Toy status: holds (§16).* | no difference between the two block types |
| P7 | in a trained looped transformer, per-sample gradient norm tracks $1/(1-\hat\rho_i)$ for attention-temperature-like parameters and not for step-size-like ones (Prop 11) | rank correlation $<0.3$ for the first group, or equal in both |
| P4 | tied/untied progress ratio tracks $T\kappa'^2/(\kappa\kappa_H)$ | tied worse at matched $\varepsilon$ where the ratio predicts $>1$ |
| P5 | under post-hoc weight noise, accuracy vs $T$ is linear for integrator-like solutions and flat-then-cliff for attractor-like | no difference in shape |

Null control for P1: inject noise of equal norm (i) in the Hessian null space at $T_{\text{train}}$ and (ii) in random directions. A1 predicts (i) produces the extrapolation loss and (ii) does not; if they match, A1 is dead.

## 13. Open problems, in order of attack

- **Pilot v1 is done (summary at the top). Next: Stage A, preregistered in `stageA/PREREG_STAGE_A.md`.** The original plan, for the record: The toy-level theory now makes four checkable predictions (P1, P3′, P6, P7). A small looped transformer on in-context regression, with a linear-attention arm and a softmax arm, clip-only and clip-plus-noise, $T\in\{4,16\}$, would test all four. Preregister before running; none of this has touched a trained model.
- **O1.** In-context linear regression with a looped block. **Done at the stationary-point level (§15), including the matrix preconditioner at $\Gamma=\gamma I$ and the softmax loop (§16).** Remaining: (a) the matrix case away from $\Gamma\propto I$ (anisotropic prompts); (b) a finite-sample statement for the clipped fixed point in terms of the law of $\lambda_{\max}$; (c) a softmax loop that does regression, not retrieval.
- **O2.** Lower bound: is $S_T^{5/2}$ necessary?
- **O3.** Theorem 1 under Lemma 1′ with clipping retained (high-probability stay in the tube).
- **O4.** Privacy accounting for locally contractive, multistable noisy loops.
- **O5.** Prove or refute C1 beyond the toy.

## 14. Prior art status

**Gate, 14–15 Nov: passed, with one adjacent paper that narrows O1.** Twelve web queries in total (six on 11 Nov, six on 14 Nov: DP + deep equilibrium/implicit models; DP + in-context linear regression; DP + looped transformer; DP + cross-layer sharing/ALBERT/Universal Transformer; clipping bias + stability edge; DP + TRM/HRM). No paper found on DP training of looped, depth-tied, equilibrium or recursive models.

| Adjacent work | What it covers | What it leaves |
|---|---|---|
| Bonnerjee, Wei, Asch, Nandy, Ghosal, arXiv 2504.16000 | DP pretraining of a single linear attention head for in-context linear regression; excess-risk bound; early stopping. Sensitivity is bounded by clipping responses and projecting features and parameters, not by per-sample gradient clipping | No loop, no clipping-bias or fixed-point analysis; looped models named only as future work |
| Hussain & Fang, arXiv 2512.19019 | Edge-of-stability behaviour of optimizers under DP (sharpness vs learning rate) | Optimizer stability, not forward-loop stability; the loop edge in §15 is a different object |
| RD-ViT, arXiv 2605.03999 (describing the Recurrent-Depth Transformer) | An injection path whose transition matrix has eigenvalues in (0,1) by parametrization | No privacy; relevant as an existing design that satisfies A5 on that path |
| Recurrent-depth work in 2026 (arXiv 2608.15062, 2608.26973, TRM 2510.04871) | Architectures and small-data results | No privacy |

Limits: web search only, not an arXiv listing sweep; snippets and one fetched paper were read, not full texts of the others. Before a submission, sweep cs.LG/cs.CR 2025–26 for "recurrent depth", "weight-tied", "equilibrium" with "private".

## 15. O1 result: looped in-context gradient descent

**Model (scalar prompts).** Prompt $i$ has in-context second moment $\lambda_i>0$ and scale $c_i$. The trainable parameter is one shared step size $\gamma$. Loop: $w_{t+1}=w_t-\gamma\lambda_i(w_t-w_i^\ast)$, $w_0=0$. With $u_i=\gamma\lambda_i$: residual $r_i=c_i(1-u_i)^T$, loss $\ell_i=\frac12r_i^2$, $\rho_i=|1-u_i|$. Prompt $i$ is unstable iff $u_i>2$.

$$g_i=\partial_\gamma\ell_i=-\frac{Tc_i^2}{\gamma}\,u_i(1-u_i)^{2T-1}.$$

**Prop 9 (gradient profile).**

- $u_i<1$ (contracting; pulls $\gamma$ up): $|g_i|\le c_i^2/(4\gamma)$ for every $T$. The supremum of $Tu(1-u)^{2T-1}$ is $\frac14$ at $T=1$ and falls to $1/(2e)$.
- $1\le u_i\le2$: $0\le g_i\le(2Tc_i^2/\gamma)\rho_i^{2T-1}$, negligible except within $O(\log T/T)$ of $u=2$.
- $u_i>2$ (expansive; pulls $\gamma$ down): $g_i\ge(2Tc_i^2/\gamma)\rho_i^{2T-1}$.

So gradient heterogeneity between an expansive and a contracting prompt is at least $8T\rho_i^{2T-1}$: linear in $\rho_i$ at $T=1$, exponential in the loop count.

**Theorem 2 (clipping turns a magnitude constraint into a count constraint).**

- (i) *Unclipped.* At any stationary point, every unstable prompt satisfies $\rho_i^{2T-1}\le nc_{\max}^2/(8Tc_i^2)$, so $\max_i\rho_i\to1$ as $T$ grows.
- (ii) *Clipped at $C$.* Stationarity reads $C\,N_C+Q_C=P_C$, where $N_C=\#\{u_i\ge1,\ g_i\ge C\}$, $Q_C=\sum_{u_i\ge1,\,g_i<C}g_i$, $P_C=\sum_{u_i<1}\min(|g_i|,C)$. So $N_C=(P_C-Q_C)/C$ prompts sit at or beyond the stability edge, and nothing bounds how far beyond: $\gamma_C$ is fixed by an order statistic, $\gamma_C\lambda_{(N_C)}\approx2$.
- (iii) *Cost of switching clipping off.* It needs $C\ge\max_ig_i$, and by (i) a single prompt's gradient can equal the whole pull-up $\sum_{u_j<1}|g_j|$, i.e. $n$ times the mean. The noise std on the mean gradient, $\sigma_{\text{DP}}C/n$, then does not shrink with $n$.

*Proof.* (i) The pull-up $\sum_{u_j<1}|g_j|\le nc_{\max}^2/(4\gamma)$ must balance the pull-down, which is at least $g_i\ge(2Tc_i^2/\gamma)\rho_i^{2T-1}$. (ii) Split the clipped sum by the sign of $g$ and by whether $g_i\ge C$. (iii) Immediate from the balance in (i). ∎

**Reading.** Unclipped training places $\gamma$ by the *maximum* of $\lambda$: $\gamma^\ast\lambda_{\max}\le2+O(\log n/T)$. Clipped training places it by a *quantile*: $\gamma_C\approx2/\lambda_{(N_C)}$, leaving the top $N_C$ prompts with $\rho_i=2\lambda_i/\lambda_{(N_C)}-1>1$ and loss $\propto\rho_i^{2T}$. Stability of a loop is a max-type property of the data; DP delivers quantile-type properties. That is a statement about sensitivity, so no clipping variant fixes it.

**Numerics** (`check_o1.py`): $p=5$, $N=10$ in-context examples, Gaussian prompts, one shared $\gamma$, 4,000 training prompts. $C$ is in units of the median per-prompt gradient at initialization.

Stationary points on the training set:

| $T$ | unclipped $\gamma\lambda_{\max}$ | unstable, unclipped | unstable, $C=$ median | test $L/L_0$, unclipped | test $L/L_0$, $C=$ median |
|---|---|---|---|---|---|
| 1 | 2.80 | 4.8% | 12.7% | 0.38 | 0.39 |
| 4 | 2.29 | 0.70% | 7.5% | 0.13 | 0.23 |
| 16 | 2.04 | 0.08% | 2.3% | 0.029 | $6.4\times10^2$ |
| 64 | 2.009 | 0.03% | 0.70% | $2.4\times10^7$ | $5.9\times10^{19}$ |

($L_0$ is the untrained loss, $\gamma=0$. Unstable = fraction of training prompts with $\gamma\lambda_{\max,i}>2$.)

Loss ratios on a fresh draw with 40,000 test prompts. $R_{\text{erm}}$ = unclipped ERM over the population-optimal $\gamma$; $R_C$ = clipped fixed point over unclipped ERM:

| block | $T$ | $R_{\text{erm}}$ | $R_C$, $C=10\times$ median | $R_C$, $C=$ median | $R_C$, $C=0.1\times$ median |
|---|---|---|---|---|---|
| raw | 1 | 1.00 | 1.003 | 1.03 | 1.04 |
| raw | 4 | 1.01 | 1.12 | 2.5 | 7.7 |
| raw | 16 | 1.78 | $3.2\times10^2$ | $4.6\times10^5$ | $3.7\times10^8$ |
| raw | 64 | $1.6\times10^6$ | $4.3\times10^{10}$ | $1.2\times10^{20}$ | $6.8\times10^{28}$ |
| normalized | 1–64 | 0.99–1.00 | 1.000 | 1.000 | 1.000 |

Normalized block: $\hat\Sigma_i$ replaced by $\hat\Sigma_i/\text{tr}\,\hat\Sigma_i$ and $\gamma\in(0,2)$, so $u_{ij}<\gamma<2$ for every prompt by construction. Its test $L/L_0$ is 0.45, 0.17, 0.030, 0.0016 at $T=1,4,16,64$, against 0.38, 0.14, 0.054 for unclipped raw at $T=1,4,16$ and $3.4\times10^3$ at $T=64$.

Dynamics ($T=16$, $C=$ median, $\mu=1$ GDP, 5 seeds): clipped noisy GD reaches the clipped fixed point ($\gamma=0.579$; root 0.578; unclipped 0.454) within 300 steps and stays there, sd of $\gamma\le0.007$.

**What the numerics do and do not show.**

- The Theorem 2 direction holds: clipping raises $\gamma$, raises the unstable fraction, and the loss ratio grows explosively with $T$.
- The effect is from clipping. Noise is negligible here: one parameter, sd of $\gamma\le0.007$.
- The unclipped baseline is itself fragile at large $T$. ERM puts $\gamma$ at the *sample* stability edge and test prompts beyond the sample maximum explode: $R_{\text{erm}}=1.6\times10^6$ at $T=64$. The raw loop is ill-posed at long horizons with or without DP.
- The normalized block's zero bias is partly trivial: all gradients share a sign, the optimum sits on the boundary $\gamma\to2$, and clipping cannot move a boundary optimum. It demonstrates the mechanism, not a general guarantee.
- These are fixed points. The trajectory from a small initial $\gamma$ passes through the unclipped optimum before step 100, so an early-stopped run can land near it. A stopping rule that sees the data costs privacy; Bonnerjee et al. analyse early stopping for the single-step case.
- One scalar parameter, linear loop, Gaussian prompts with an unbounded $\lambda$ tail. A matrix preconditioner or a real attention block may behave differently.

**Matrix preconditioner (v0.3).** Replace $\gamma$ by a full $p\times p$ matrix $\Gamma$ and clip the Frobenius norm of the matrix gradient. At $\Gamma=\gamma I$ the stationarity condition in the trace direction is the same count balance with different weights. Three draws, threshold at the median norm:

| $T$ | unstable, unclipped | unstable, scalar clip | unstable, matrix clip | $R_C$ scalar | $R_C$ matrix |
|---|---|---|---|---|---|
| 1 | 3.9–5.4% | 11–12% | 10–11% | 1.03 | 1.02–1.03 |
| 4 | 0.2–0.8% | 6.5–7.6% | 3.5–4.5% | 2.2–3.0 | 1.2–1.7 |
| 16 | 0.03–0.05% | 2.1–2.4% | 0.73–0.77% | $5.7\times10^5$–$6.8\times10^6$ | $1.0\times10^4$–$1.4\times10^5$ |

The mechanism persists and is weaker under matrix clipping. Only the trace direction at $\Gamma\propto I$ is checked; transverse stability and anisotropic prompts are open.

**Design implication (A6).** Remove the data dependence of the stability region instead of asking DP training to learn it. Candidates: per-prompt or token normalization (the toy), a bounded state, a transition parametrized to be contractive. §16 separates what each of these buys; v0.2's claim here that softmax attention "cannot expand" was wrong (correction 7).

**Relation to the label-blind clipping note** (a separate study, not in this repository)**.** Same family: clipping moves the fixed point of the objective. The difference is that there a reweighting that ignores the label restores the fixed point, because the restoring force is carried by sign balance. Here it is carried by magnitude, so no reweighting restores it.

## 16. Softmax-attention loop

**Model.** Tokens $x_1,\dots,x_M$ (keys = values), query $q$, input injection: $z_0=0$,
$$z_{t+1}=(1-\alpha)z_t+\alpha\sum_jp_j(z_t)\,x_j,\qquad p(z)=\text{softmax}_j\big(\beta\,x_j^\top(q+z)\big).$$
One trainable inverse temperature $\beta$. Task in the toy: retrieve the stored token nearest a noisy query.

**Prop 10 (token-normalized softmax loop).** If $\|x_j\|\le R$ and $\|q\|\le R$, then for every prompt, every $\beta\ge0$ and every $T$:

- (a) $\|z_t\|\le R$.
- (b) $\|J_z\|\le(1-\alpha)+\alpha\beta R^2$.
- (c) $\|\partial_\beta F\|\le2\alpha R^3$.
- (d) If $\beta R^2<1$, then $\rho=1-\alpha(1-\beta R^2)<1$ and $|\partial_\beta\ell|\le4\alpha R^4S_T(\rho)\le4R^4/(1-\beta R^2)$.

*Proof.* (a) $z_{t+1}$ is a convex combination of $z_t$ and the tokens. (b) $J_z=(1-\alpha)I+\alpha\beta\,\text{Cov}_p(x)$ and $\text{Cov}_p(x)\preceq R^2I$. (c) $\partial_\beta F=\alpha\,\text{Cov}_p(x)(q+z)$. (d) Lemma 2 with $G\le2R$. With separate keys and values the bound in (b) is $(1-\alpha)+\alpha\beta R_kR_v$ by Cauchy–Schwarz on the cross-covariance. ∎

Every bound is in terms of the parameters and the normalization radius, not the data. With $\beta$ constrained by parametrization to $\beta R^2\le1-\epsilon$, (d) is the a-priori sensitivity that clipless DP needs (A5). Token normalization does the same for linear attention: $\lambda_{\max}(\hat\Sigma_i)\le\text{tr}\,\hat\Sigma_i\le R^2$, so $\gamma<2/R^2$ is stable for every prompt, which is the "normalized block" of §15.

**Prop 11 (rate parameters versus fixed-point parameters).** Let the loop converge to a fixed point $z^\ast(\theta,x_i)$ at local rate $\rho_i<1$. Then $\nabla_\theta\ell_i\to v_i^\top(I-J_i)^{-1}B_i$ as $T\to\infty$, with $J_i,B_i$ evaluated at $z^\ast$.

- (a) If $B_i=0$ at the fixed point (the parameter changes how fast the loop converges, not where), the gradient of a converged sample vanishes. Large gradients come only from samples that have not converged, and on the expansive side they are exponential in $T$ (Prop 9).
- (b) Otherwise $\|\nabla_\theta\ell_i\|\le GL_\theta/(1-\rho_i)$, and the heterogeneity across samples is that of $1/(1-\rho_i)$: polynomial, and saturating in $T$.

*Proof.* Differentiate $z^\ast=F(z^\ast;\theta)$: $\partial_\theta z^\ast=(I-J)^{-1}B$. In the linear loop of §15, $\partial_\gamma F=-\lambda_i(w-w_i^\ast)=0$ at $w=w_i^\ast$, which is case (a). ∎

This is what was wrong with H2: it is case (b) reasoning applied to a case (a) loop.

**Numerics** (`check_softmax.py`; $M=8$ tokens, $p=8$, $\alpha=0.5$, unit-norm tokens unless stated).

- *Prop 10.* Observed over bound is at most 0.89 for the Jacobian and 0.074 for the gradient, for $\beta\in\{0.3,0.6,0.9,0.99\}$ and $T$ up to 256. Inside the contraction regime the max/median gradient ratio is at most 3.8 and flat in $T$.
- *Price of contraction.* Test $L/L_0$ is 0.68, 0.63, 0.63 at $\beta=0.99$ against 0.22, 0.20, 0.20 at $\beta=3.3$ ($T=4,16,64$): 3.1–3.3× worse. The trained $\beta$ is 2.9–3.4 for $T\ge16$, in the multistable regime.
- *Beyond contraction* ($T=64$). The state stays bounded (max loss 1.1–1.6, cap 2). The Jacobian norm reaches 6.7 at $\beta=20$. The max/median gradient ratio is 21 at $\beta=2$ and $2.1\times10^3$ at $\beta=5$.
- *Prop 11 (b) diagnostic*, $\beta=3$: rank correlation between $|g_i|$ and $1/(1-\rho_i)$ is 0.46, 0.85, 0.85, 0.85 at $T=4,16,64,256$. Max/median $|g|$ is 9.5, 100, 473, 545, i.e. saturating. The top 1% of prompts carry 5%, 14%, 17%, 16% of $\sum|g|$.
- *Unnormalized tokens*, $\beta=5$: Jacobian norm 10.7; max/median $|g|$ 47, 664, 3,714 at $T=4,16,64$.
- *Forgetting without injection*, $\beta=0.5$: two different queries on the same tokens end $1.6\times10^{-1}$, $2.4\times10^{-4}$, $3.3\times10^{-14}$ apart at $T=4,16,64$; with injection, 0.19–0.21 at every $T$.

Clipping fixed points, reached from soft attention. $C$ in units of the median gradient at $\beta=1$; $R_C$ as in §15:

| $T$ | $\beta$ unclipped | max/median $\lvert g\rvert$ | $R_C$, $C=10\times$ median | $R_C$, $C=$ median | $R_C$, $C=0.1\times$ median |
|---|---|---|---|---|---|
| 1 | 31.6 | 526 | 1.000 | 1.000 | 1.000 |
| 4 | 5.56 | 145 | 1.000 | 1.001 | 1.037 |
| 16 | 3.29 | 69 | 1.000 | 1.002 | 1.079 |
| 64 | 3.43 | 554 | 0.996 | 0.997 | 1.068 |

Replication at $C=$ median over five independent draws: $R_C$ median 1.000, range 0.999–1.001 at $T=16$; median 1.004, range 0.990–1.009 at $T=64$.

**Reading.**

- Gradient heterogeneity is present in the softmax loop (max/median in the hundreds) and it is the $1/(1-\rho_i)$ kind. Clipping nevertheless costs at most 8% in loss, and under 1% at the median threshold, flat in $T$.
- Against the raw linear loop of §15 at the same thresholds: $10^5$–$10^7$ at $T=16$. P6 holds at toy level.
- Two things differ between the loops and the toys do not separate them: the state is bounded, and the parameter moves the fixed point instead of the rate. Prop 11 says the second controls how heavy the gradient tail is; boundedness controls how much a displaced fixed point can cost.

**Caveats.** One scalar parameter; keys equal values; a retrieval task, not regression; $\rho_i$ is the Jacobian norm at the final state, an upper proxy for the local rate; no DP noise in these runs; $T\le64$ for the clipping table.

## 17. The softmax toy and the pilot's softmax block are different models

The review of 16 Nov pointed out that §16 and pilot prediction E2 were being read as one result. They are not.

| | Softmax toy (§16, `checks/check_softmax.py`) | Pilot block `soft_norm` (`pilot/pilot.py`) |
|---|---|---|
| Trainable parameters | one inverse temperature | about 8.8k: embedding, Q, K, V, O, MLP, LayerNorm gains, head |
| Keys and values | identical (the tokens) | separate learned projections |
| Tokens | unit norm | raw Gaussian; LayerNorm before attention and before the MLP |
| Update | convex combination of state and tokens | residual stream: identity path plus attention plus MLP |
| State | bounded by construction | unbounded; grows with the loop count through input injection |
| Task | retrieval | in-context linear regression |
| What is proven | Prop 10 bounds | nothing |

**What E2 can show.** That clipping has a small effect on this particular small pre-LayerNorm residual block, at the loop counts tested. It does not test Prop 10, and it does not validate "bounded softmax loops are robust" from §16.

**Shared between the two.** Attention weights are normalized, so the attention output is a convex combination of value vectors; the parameter that sharpens attention moves the loop's fixed point instead of only its rate (Prop 11 b).

**Different.** The toy's robustness argument rests on a bounded state and a single fixed-point parameter. The pilot block has neither a bounded state nor a contraction bound: its Jacobian has an identity path, so the jvp-based expansion rate is at least 1 for every prompt. Whatever makes it insensitive to clipping in E2 is not established; LayerNorm before the read-out, which bounds the prediction, is the obvious candidate and is untested.

**Consequence for the claims.** A6 is a statement about the toy. A trained-model test of it needs a pilot arm that implements the toy's block (convex-combination update, normalized tokens), which the current pilot does not have.

## Numerical checks

`checks/check_softmax.py` (numpy, about 10 minutes; each function reseeds and can be run alone): Props 10–11, the forgetting check, the softmax clipping table and replication of §16, and the matrix-preconditioner table of §15.

`checks/check_theory.py` (numpy, about a minute).

- Lemmas 1–3 on a tanh loop, $\rho\in\{0.5,0.9,0.99\}$, $T\in\{5,\dots,400\}$, 200 pairs each: observed/bound $\le0.20$, $0.17$, $0.01$. The bounds hold and are loose for random loops.
- Prop 3: Monte Carlo over predicted within 2.4% for tied and untied; gradient identity $M_T^\top v$ to $10^{-16}$.
- Props 4 and 7: tables above; the optimal shift schedule is feasible in every case.
- §9 toy: as tabulated.

`checks/check_o1.py` (numpy, about 2.5 minutes).

- Prop 9: supremum of the contracting-side profile is 0.250, 0.211, 0.196, 0.187, 0.185, 0.184 at $T=1,2,4,16,64,256$; at $u=2.2$ the expansive-side value is 2.6, 7.6, 32, $1.0\times10^4$, $1.6\times10^{12}$.
- Theorem 2 (i): unclipped $\max_i\rho_i$ is 1.29, 1.04, 1.009 at $T=4,16,64$.
- Theorem 2 (ii) and the ratios: §15 tables.

## References

IDs marked ✓ were seen in search results on 11–15 Nov; the rest are from memory and need verifying before citation.

- Bonnerjee, Wei, Asch, Nandy, Ghosal. How Private is Your Attention? Bridging Privacy with In-Context Learning. arXiv 2504.16000 ✓.
- Hussain, Fang. Optimizer Dynamics at the Edge of Stability with Differential Privacy. arXiv 2512.19019 ✓.
- RD-ViT: Recurrent-Depth Vision Transformer for Semantic Segmentation with Reduced Data Dependence. arXiv 2605.03999 ✓.
- Jolicoeur-Martineau. Less is More: Recursive Reasoning with Tiny Networks. arXiv 2510.04871 ✓.

- Yang, Lee, Nowak, Papailiopoulos. Looped Transformers are Better at Learning Learning Algorithms. arXiv 2311.12424.
- Fan, Du, Ramchandran, Lee. Looped Transformers for Length Generalization. arXiv 2409.15647.
- Gatmiry, Saunshi, Reddi, Jegelka, Kumar. Can Looped Transformers Learn to Implement Multi-step Gradient Descent for In-context Learning? arXiv 2410.08292.
- Miller, Hardt. Stable Recurrent Models. arXiv 1805.10369.
- Bai, Koltun, Kolter. Stabilizing Equilibrium Models by Jacobian Regularization. arXiv 2106.14342.
- Kim, Papamakarios, Mnih. The Lipschitz Constant of Self-Attention. arXiv 2006.04710.
- Castin, Ablin, Peyré. How Smooth Is Attention? arXiv 2312.14820.
- Ghadimi, Wang. Approximation Methods for Bilevel Programming. arXiv 1802.02246.
- Dong, Roth, Su. Gaussian Differential Privacy. arXiv 1905.02383.
- Feldman, Mironov, Talwar, Thakurta. Privacy Amplification by Iteration. arXiv 1808.06651.
- Altschuler, Talwar. Privacy of Noisy SGD: More Iterations without More Privacy Loss. arXiv 2205.13710 ✓.
- Asoodeh, Diaz. Privacy Loss of Noisy SGD Might Converge Even for Non-Convex Losses. arXiv 2305.09903 ✓.
- Chen, Wu, Hong. Understanding Gradient Clipping in Private SGD: A Geometric Perspective. arXiv 2006.15429 ✓.
- Koloskova, Hendrikx, Stich. Revisiting Gradient Clipping: Stochastic Bias and Tight Convergence Guarantees. arXiv 2305.01588.
- Béthune et al. DP-SGD Without Clipping: The Lipschitz Neural Network Way. arXiv 2305.16202 ✓.
- Ding et al. Delving into Differentially Private Transformer. arXiv 2405.18194 ✓.
- El Mais, Chehab, Issa, Tajeddine. Is Weight Tying Still Beneficial for Decoder-Only LLMs in Private Settings Under DP-SGD? (Sept 2026; ID not retrieved).

## Appendix A. Withdrawn material

Kept so the record is complete. Nothing in this appendix is part of the current theory.

**H2 (v0.1, withdrawn in v0.2, scope clarified in v0.3).**

In looped models $w_i$ is anti-correlated with the loop depth sample $i$ needs. Slowly contracting samples have gradients larger by up to $S_T(\rho_i)/S_T(\rho_{\text{med}})\le T(1-\rho_{\text{med}})$, so the effective objective $\sum_iw_i\ell_i$ underweights them, increasingly with $T$ until $T\gtrsim1/(1-\max_i\rho_i)$.

*Worked instance given with H2.* A looped linear-attention block doing in-context gradient descent on least squares has $w_{t+1}=(I-\eta\hat\Sigma_x)w_t+\eta X^\top y/n$, so $\rho_i=1-\eta\lambda_{\min}(\hat\Sigma_{x_i})$. Heterogeneity of $\rho_i$ is heterogeneity of in-context condition numbers, and H2 predicts DP clipping down-weights ill-conditioned prompts.

Why it was wrong: this is fixed-point-parameter reasoning (Prop 11 b) applied to a loop whose parameter only sets the rate (Prop 11 a). §15 has the corrected statement.

**P3 (v0.1, withdrawn in v0.2).** "Clip weight $w_i$ anti-correlated with $S_T(\hat\rho_i)$; the clip-only arm hurts ill-conditioned prompts more, increasingly with $T$." Built on H2. Replaced by P3′.

**Pilot E1, ordering clause (preregistered 15 Nov, failed).** "The unstable-fraction difference at $T=16$ exceeds the one at $T=4$." It failed (+4.74 pp against +5.30 pp), and it was inconsistent with this document's own §15 table, where the unstable fraction falls with $T$ while the loss consequence grows. The failure stays on record in `pilot/PREREG_PILOT.md`; the clause will not be carried into the next preregistration as a confirmatory test.
