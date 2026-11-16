# Stage A preregistration: is clipping-induced stability erosion robust to threshold, optimizer and loop count?

Written 16 Nov 2026, before any Stage A training run on a selection, screening or confirmation seed.
Code: `stageA/stageA.py`, SHA-256 `54d4ff81ad0d4a9c2ca67133a73f19ae96b2548861a0edd2f9e9d1c4ca521f23` (JAX, CPU, float64).

## Scope
Stage A only. It asks one question: is pilot result E1 (clipping raises the unstable fraction of a trained linear loop)
an accident of median clipping + Adam + one loop count? Stage B (noise) is postponed until Stage A is in. Stage C
(softmax blocks) gets its own preregistration after the A1 results. The three pilot seeds stay a pilot and are not pooled
with anything here.

Two tiers:

| Tier | Design | Seeds | Role |
|---|---|---|---|
| A0 screening | full grid: 3 variants × 2 optimizers × 5 loop counts × (6 thresholds + unclipped) = 210 cells | 3 (seeds 11–13) | Descriptive map of how the effect depends on C, T, optimizer and normalization. A mechanism sweep, not 210 confirmatory results; no tests |
| A1 confirmation | 17 cells carrying 6 primary contrasts | 10 new (seeds 101–110) | Formal tests, Holm-corrected |

The A1 contrasts are fixed here, before A0 is run, and will not be changed in the light of A0.

## Model and variants
Weight-tied linear loop doing in-context gradient descent on 10 context pairs in R^5 with a trainable 5×5 preconditioner A
(the sparse parametrization of the pilot). In w-space the error obeys e_{t+1} = M_i e_t, M_i = I − A·Sigma_i.
All variants start cold at A ≈ 0.05·I, so every arm has to train.

- `raw`: raw Gaussian tokens, A free.
- `tok`: token-normalized prompts (every token scaled to ||x_j|| = sqrt(5)), A free.
- `con`: the stability constraint, defined explicitly. Token-normalized prompts, and A = c·G/lambda_max(G) with
  G = L·Lᵀ + 1e-6·I symmetric positive definite and c = ((2 − 0.1)/5)·sigmoid(a). Then ||A||₂ ≤ 0.38, and because
  lambda_max(Sigma_i) ≤ tr(Sigma_i) = 5 for every token-normalized prompt, every eigenvalue of A·Sigma_i lies in (0, 1.9].
  So the spectral radius of M_i is below 1 for every admissible prompt and every parameter value. This is a certificate
  under a bounded data distribution, obtained by constraining the token norm first and the spectrum second.
  - It is an asymptotic certificate. M_i is non-normal, so its T-step gain may still exceed 1.
  - Raw Gaussian prompts have no finite bound on Sigma_i, so no such certificate exists for `raw`; none is claimed.

## Arms and training
- Unclipped reference, and per-prompt clipping at C = c × (median per-prompt gradient norm at initialization, on 1,024
  calibration prompts), c ∈ {0.1, 0.25, 0.5, 1, 2, 10}. No noise anywhere in Stage A.
- Optimizers: Adam (0.9, 0.999) and plain SGD, both on the Poisson-sampled mean gradient (q = 1/128), 2,000 steps.
- Loop counts T ∈ {1, 4, 8, 16, 32}. 16,384 training prompts; 16,384 fresh test prompts per seed.
- Within a seed every arm shares data, initialization and the batch sequence.

**Non-finite gradients.** Nothing is zeroed. If any sampled prompt has a non-finite gradient norm, or the parameters become
non-finite, the run stops and is recorded as invalid with the step and the affected fraction. Invalid runs are never
replaced by a number.

## Learning rate: two estimands, kept apart
- **Primary (effect of clipping under identical training dynamics).** For each (variant, optimizer, T) the learning rate is
  chosen on the unclipped arm only, on selection seed 10, by validation median loss at T (grids: Adam 3e-4 … 1e-1, six
  values; SGD 1e-3 … 3, eight values). Every arm of that setting uses it.
- **Secondary (does clipping change trainability).** Each clipped A1 cell also tunes its own learning rate the same way.
  Reported with the same statistics, labelled as the trainability estimand, never mixed with the primary one, no tests
  in the confirmatory family. Pilot E2 at T=16 was an instance of this second estimand.

## Outcomes
- **Primary: `f_gain`**, the fraction of the 16,384 test prompts whose T-step gain ||M_i^T||₂ exceeds 1.
  **Erosion** of a clipped cell = f_gain(clipped) − f_gain(unclipped), paired by seed.
- Secondary: `f_asym` (spectral radius of M_i above 1), `f_err10` (error at 2T more than 10× the error at T), `fcat`
  (error above 10·L0), median, q90, q99 and mean loss, 99th-percentile gain.
- Recorded with every run: code hash, library versions, configuration, clipped fraction over the last 100 steps,
  relative change of A over the last 200 steps, distance from initialization, final A.

## Validity rules
- V1. A seed-pair with an invalid run in either arm is dropped from that contrast and counted.
- V2. A contrast with more than 2 of its 10 pairs dropped is **not interpretable**.
- V3. A confirmatory cell whose unclipped arm has median L/L0 above 0.3 at T on more than 2 seeds is a failed baseline,
  and its contrasts are not interpretable.

## A1: primary contrasts (the confirmatory family)
Six one-sided tests on per-seed paired effects. Test: exact sign-flip permutation test on the mean (1,024 sign patterns);
Holm correction across the six at family level 0.05. Reported for each: mean, 95% t-interval, every per-seed effect,
raw and adjusted p. All use the shared-learning-rate estimand and `f_gain`.

| | Contrast | Cells | Hypothesis |
|---|---|---|---|
| H1 | Replication | raw, Adam, T=16, c=1 | erosion > 0 |
| H2 | Dose-response | raw, Adam, T=16, c from 0.1 to 10 | erosion falls as c rises: per-seed slope of erosion on log10(c) is negative |
| H3 | Optimizer | raw, SGD, T=16, c=1 | erosion > 0 without Adam |
| H4 | Token normalization | raw vs tok, Adam, T=16, c=1 | erosion(raw) − erosion(tok) > 0 |
| H5 | Stability constraint | raw vs con, Adam, T=16, c=1 | erosion(raw) − erosion(con) > 0 |
| H6 | Loop count | raw, Adam, c=1, T=4 and T=32 | erosion > 0 at both (p = the larger of the two) |

Decision rules:
- A contrast **passes** if its Holm-adjusted p is below 0.05. H1 additionally needs a mean of at least +0.3 pp to count
  as a replication; below that it is "detectable but negligible".
- H1 fails → the maximum-to-quantile claim loses its trained-model support and rests on fixed-point analysis only.
- H2 fails → the effect is not controlled by the threshold the way Theorem 2 says; the mechanism reading is weakened
  even if H1 passes.
- H3 fails → the claim is stated for Adam only, and the accident hypothesis is partly right.
- H4 or H5 fails → that ingredient of "make stability data-independent" is unsupported.
- H6 fails → the effect is tied to T=16.
- No ordering of erosion across T is predicted (the pilot's ordering clause failed and is not carried forward).

What I expect, and how sure I am:
- H1, H2, H4: I expect these to pass. The fixed-point toy gives a monotone dose-response, and the pilot gave +5.5 pp on
  this metric. The cold start and the shared learning rate are new, so this is a conceptual replication, not an exact one.
- H5: I expect it to pass on the asymptotic metric by construction (f_asym is 0 in `con`). On the primary metric the
  constraint permits transient gain, so the size is uncertain.
- H3: I do not know. Under SGD the shared learning rate is set by the unclipped arm, which is only valid at small
  learning rates (see stage 0), so the clipped arm may simply not travel far enough in 2,000 steps.
- H6: I expect T=32 to pass and am less sure about T=4.

## A0: screening (descriptive only)
For each variant × optimizer: the table of mean erosion over c × T with per-seed spread, the unclipped f_gain and median
loss, and the invalid count. Known limits fixed in advance:
- At T=1, f_gain is the one-step spectral-norm criterion, which the pilot showed does not separate arms. T=1 is screening only.
- Under SGD with a shared learning rate, a smaller c shrinks the update in proportion, so the SGD dose-response is
  confounded with step size. Only c=1 is confirmatory for SGD.
- Three seeds: no tests, no claims of interaction.

## What was looked at before freezing (stage 0)
- G1 gradient against finite difference in float64: agree to 10 digits.
- G2 constraint certificate: 200 random parameter draws × 2,000 token-normalized prompts: ||A||₂ never above 0.38 and the
  spectral radius of M_i never reaches 1 (it approaches 1 only as A → 0).
- G3 non-finite handling: an unclipped SGD run with an absurd learning rate stops at step 2 and is marked invalid.
- G4 the sign-flip test and Holm correction reproduce known values.
- G5 clipped code path: three 20-step runs on seed 99; only validity was inspected.
- G6 unclipped arm only, seed 99 (not an analysis seed), 1,000 steps, two rounds:
  1. Warm start at 0.3·I (the pilot's). For raw + Adam at T=16 and 32 the best learning rate was the smallest in the
     grid: the warm start was already near the best median loss, so the selection rule would have chosen a rate that
     barely trains any arm. Unclipped raw SGD was invalid at every rate tried at T=16. → Changed to a cold start
     (0.05·I) and wider grids.
  2. Cold start. Adam selections are interior. Unclipped raw SGD is valid only up to 0.01 at T=16 and 32. For `con`
     the best rate was the top of the grid. → Grids extended once more (Adam 0.1, SGD 3) without a further look.
- Steps raised from 1,000 to 2,000 after seeing A still moving about 10% over the last 200 steps. One timing run at
  2,000 steps (unclipped, raw, Adam, seed 99): still 11% at T=16 and 4% at T=32.
- No clipped run longer than 20 steps has been executed, and no run of any kind on seeds 10–13 or 101–110.

## Limits stated in advance
- Runs are compared at a matched budget, not at fixed points: unclipped Adam does not settle within 2,000 steps.
- If the unclipped SGD rate selected on seed 10 is invalid on more than two A1 seeds, H3 is not interpretable (V2).
- Cold start, float64 and the shared learning rate all differ from the pilot.
- One task, one model size, synthetic data, no noise. A pass supports the mechanism for linear in-context loops at this
  scale, nothing broader.

## Size and order of execution
About 1,190 runs at roughly 7 seconds each, a little over 2 hours on 2 CPU cores:
selection 210 → A0 630 → A1 170 → secondary-estimand selection 68 and evaluation 110.
`python3 stageA.py run --tier select`, then `a0`, `a1`, `a1tune_select`, `a1tune`; `python3 stageA.py analyze`.

## Amendments
Any change after this file is committed is logged here with its date; results obtained under the earlier design are kept.
