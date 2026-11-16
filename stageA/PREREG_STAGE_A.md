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

(No amendments. Everything above this line is the frozen text, SHA-256
`e44fb77b5f3f01d9a15656f8d32e8e4613313e988cf36e7f45f4e4991980b3ca`.)

## Outcome (16 Nov 2026): all 1,188 runs
Results: `stageA/results/stageA_results.jsonl`, SHA-256 `118d55f7a008c812260949ad8a8eb523fcbfa8f56354844a91a53b41dd758438`;
full analysis output: `stageA/results/analysis.txt`. All rows carry the frozen code hash.
Run notes: the last tier (secondary estimand) was interrupted by an environment shutdown at 57 of 110 runs and resumed;
the file has 1,188 rows and 1,188 distinct keys. The confirmation tier was started before any screening result was looked at.

### Confirmatory family (primary estimand, primary metric, 10 seeds, Holm at 0.05)

| | Contrast | Mean [95% interval] | Holm p | Result |
|---|---|---|---|---|
| H1 | erosion > 0 (raw, Adam, T=16, c=1) | −17.72 pp [−38.54, +3.10] | 1.00 | **Failed** |
| H2 | erosion falls as c rises | +1.18 pp per decade [+1.00, +1.37] | 0.006 | **Passed** (10/10 seeds) |
| H3 | erosion > 0 under SGD | −0.00 pp [−0.01, +0.00], n = 9 | 1.00 | **Failed** |
| H4 | erosion(raw) − erosion(tok) > 0 | −17.73 pp [−38.55, +3.08] | 1.00 | **Failed** |
| H5 | erosion(raw) − erosion(con) > 0 | −17.72 pp [−38.54, +3.10] | 1.00 | **Failed** |
| H6 | erosion > 0 at T=4 and T=32 | T=4: −36.29 pp [−67.57, −5.01]; T=32: −11.40 pp [−27.25, +4.45] | 1.00 | **Failed** |

Per-seed H1 effects (pp): −92.36, −36.12, −4.35, −26.87, +0.54, +0.85, −3.61, +0.49, −12.84, −2.91.

By the decision rules fixed above: **H1 failed, so the "clipping erodes stability relative to unclipped training" claim
has lost its trained-model support.** The accident hypothesis is largely right: pilot E1 does not survive a change of
initialization and learning-rate rule. H2 passed, so the threshold dose-response stands.

### Not interpretable
None by rules V1–V3 (one dropped pair, in H3; one seed of ten with a poor unclipped baseline at T=16 and at T=4).

### Why H1 failed (descriptive; read off the preregistered runs)
- **The unclipped arm is not a stable reference.** Under Adam at the selected rate its primary-metric fraction ranges
  from 0.07% to 93.26% across the ten seeds at T=16 (1.04% to 99.60% at T=4, 0.04% to 69.68% at T=32). The clipped arm at
  c=1 ranges from 0.65% to 1.01%.
- So "erosion" measured how erratic unclipped Adam is. At every threshold tested, clipped training was *more* stable
  than unclipped training on the primary metric (−15.7 pp at c=0.1 to −18.1 pp at c=10).
- Among clipped arms the dose-response is clean: 2.87%, 1.95%, 1.30%, 0.86%, 0.67%, 0.48% unstable at
  c = 0.1, 0.25, 0.5, 1, 2, 10. Lowest against highest threshold, paired: +2.39 pp [+2.04, +2.73], 10/10 seeds.
- The learning rate was selected on one seed, and the unclipped arm's outcome varies a lot between seeds: at T=4 the
  rule chose 0.1 by a margin of 0.0011 in validation loss, and that rate is poor on most confirmation seeds.
  Selecting on a single seed was a weakness of this design.
- H3: under the shared rate (0.003) both SGD arms sit at about 0.02% and only 0.2% of gradients are clipped, so the
  contrast had almost nothing to act on. Every one of the 42 invalid runs in the whole study is an unclipped SGD run;
  no clipped run was ever invalid.
- H4, H5: token normalization: erosion +0.02 pp [−0.04, +0.07]. Constraint: exactly 0.00% on the primary metric in both
  arms on every seed, so the transient-gain concern about the certificate did not materialize. Both variants are stable;
  the contrasts failed because raw erosion was negative, not because the variants misbehaved.
- Secondary metric (asymptotic), same H1 cells: clipped − unclipped per seed is −64.68, +0.59, −1.65, +0.43, +0.70,
  +0.62, +0.58, +0.54, +0.55, +0.45: median +0.55 pp, positive in 8 of 10, mean dominated by one blown-up unclipped run.

### Secondary estimand (each clipped arm tunes its own rate; trainability, no confirmatory weight)
Same picture for H1, H2, H4, H5, H6. One difference: under SGD the clipped arm selects 0.1 against the unclipped arm's
0.003, reaches a 9× lower median loss (0.00025 against 0.0023) and has 0.63–0.99% unstable prompts against about 0.02%:
+0.80 pp [+0.72, +0.89], 9/9 seeds. Clipping lets SGD train to where it otherwise cannot go, and that solution is better
for the bulk and worse for the tail.

### A0 screening (descriptive, 3 seeds)
- raw + Adam: clipped arms more unstable than unclipped at T = 1, 4, 8 with a threshold dose-response (T=4: +11.9 pp at
  c=0.1 falling to about +3 pp at c ≥ 1), and less unstable at T = 16, 32, where the unclipped arm itself is unstable
  (23.5% and 3.2%). The sign follows whether the unclipped arm happened to be stable.
- A0 and A1 disagree at raw + Adam, T=4, c=1 (+3.58 pp on 3 seeds, −36.29 pp on 10): three seeds cannot characterize an
  erratic reference.
- raw + SGD: within ±1 pp at T=4 with no consistent sign, about 0 at T=8 and 16; no valid unclipped run at T=32. tok: small positive erosion with a dose-response (T=4: +1.9 pp at c=0.1 to +0.06 at
  c=10), under 0.3 pp at T ≥ 16. con: 0.00 everywhere.

### Exploratory, post hoc, not preregistered
Run after the analysis above was seen; `stageA/results/exploratory_unclipped_lr.jsonl`. Question: at the pilot's rate
(0.003), is the unclipped Adam reference stable, and does erosion reappear? Ten unclipped runs (raw, Adam, T=16) on the
confirmation seeds, paired with the clipped c=1 arm that had tuned to the same rate.
- The unclipped arm is stable at this rate (0.07–1.98% on the primary metric).
- Primary metric: clipped − unclipped = +0.18 pp [−0.29, +0.65], positive in 7/10. No clear effect.
- Asymptotic metric: +0.51 pp [+0.41, +0.61], positive in 10/10.
- So where the unclipped reference is stable, an effect in the pilot's direction exists on the pilot's metric at about a
  tenth of the pilot's size (0.5 pp against 4.7 pp), and is not established on the primary metric. This is a lead for a
  future preregistration, not a result.
