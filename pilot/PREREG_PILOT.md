# DP × looped transformer pilot: preregistration (15 Nov 2026, written before any clip or dp training run)

First test on trained models of the predictions in `claude/dp-looped-theory.md` v0.3. Code: `claude/dp-looped/pilot.py`,
SHA-256 `6240354aeb844e0f3c78fa0ecb93377de704cb6bcf1138d09f78b2c02a11be09` (JAX, CPU; about 2 hours on 2 cores).

## What was looked at before freezing (stage 0)
- Accountant: reproduces the commonly quoted eps of about 3.0 for q = 256/60000, sigma = 1.1, 60 epochs (got 3.009).
- Per-prompt gradient against a finite difference: 0.03598 vs 0.03599.
- Non-private arm only, seed 100 (not an analysis seed), 300 steps, lr 3e-3. Test median L/L0:
  lin_raw 0.027 (T=4), 0.002 (T=16); lin_norm 0.012, 0.000; soft_norm 0.062, 0.065. All baselines learn.
- Mechanics check of the clip and dp code paths: 20, 20 and 5 steps on seed 100; only finiteness of the outputs was checked.
- Two changes made at stage 0, before any clip or dp run:
  1. The linear arm first used dense P, Q matrices. At T=16 it gave non-finite per-prompt gradients at initialization and
     the non-private run diverged. Replaced by the sparse parametrization below.
  2. First sizing (q = 1/64, 1500 steps, 5 evaluation seeds) cut for compute to q = 1/128, 1000 steps, 3 seeds.
- No clip or dp run of more than 20 steps has been executed at any seed, and none at an analysis seed.

## Design
- **Task.** In-context linear regression: 10 context pairs (x_j, y_j = w·x_j) and a query, x in R^5 Gaussian, w ~ N(0, I/5).
  One training record = one prompt (the privacy unit). 16,384 training prompts; 16,384 fresh test prompts per seed.
- **Architectures** (weight-tied, T loop iterations):
  - `lin_raw`: looped linear transformer in the sparse form of Ahn et al. / Gatmiry et al. One trainable 5×5 matrix A;
    the loop is in-context gradient descent with preconditioner A; prompt i is unstable iff the spectral radius of
    I − A·Sigma_i exceeds 1. This is the matrix model of §15.
  - `lin_norm`: the same block on token-normalized prompts (every token scaled so that ||x_j|| = sqrt(5)).
  - `soft_norm`: softmax attention + MLP, pre-LayerNorm, input injection (Yang et al. style), width 32, about 8.8k parameters.
- **Arms.** `np`: no clipping, no noise. `clip`: per-prompt clipping at C, no noise. `dp`: clipping at C plus Gaussian
  noise, multiplier 1.0. C = median per-prompt gradient norm at initialization on 1,024 fresh calibration prompts, fixed.
- **Training.** Poisson sampling, q = 1/128, in every arm; Adam on the sampled mean gradient; 1000 steps.
  dp arm: eps = 2.11 at delta = 1e-5 (RDP accountant for the Poisson-subsampled Gaussian).
- **Loop counts.** T in {4, 16}. Plus two cells with random loop count (uniform in 8..16 per step): soft_norm, T=16, clip and dp.
- **Learning rate.** Grid 1e-3, 3e-3, 1e-2. Chosen per cell on seed 0 by validation median loss at T (4,096 fresh prompts,
  synthetic, treated as public). Evaluated on seeds 1–3. A diverged run counts as infinite validation loss.
- **Size.** 20 cells × (3 selection + 3 evaluation) = 120 runs.
- **Pairing.** Within a seed, all arms share training data, initialization and the Poisson sampling sequence.

## Outcomes
- `med`: median per-prompt squared error / L0, with L0 = mean y² (the trivial predictor). At T and at 2T.
- `fcat`: fraction of test prompts with squared error > 10·L0. A diverged run scores 1.
- `f_unstable_exact` (linear arms): fraction of the 16,384 test prompts with spectral radius of I − A·Sigma_i above 1.
- Mean loss is reported but not used for any decision: it is dominated by a handful of prompts (theory doc, correction 9).
- All effects are paired over seeds 1–3: mean and 95% t-interval (t = 4.303, df 2); ratios on the log scale.

## Validity gate
V1. A cell enters the confirmatory analysis only if its np arm reaches med ≤ 0.3 at T on all three evaluation seeds and
at most one np run diverges. Cells that fail are reported as failed baselines.

## Predictions
**E1 (P3′, stability erosion; lin_raw).** At T=16, f_unstable_exact(clip) − f_unstable_exact(np) ≥ +0.3 percentage points,
interval excluding 0, and the T=16 point estimate exceeds the T=4 one.
- Kill: point estimate < +0.1 pp, or interval includes 0. Then Theorem 2's mechanism does not show in a trained
  matrix-preconditioner loop at this scale, and A2 is downgraded to "fixed-point analysis of a toy".
- Toy fixed-point values were 0.7–2.3%. Here training runs 1000 Adam steps from a small step size and may stop short of
  the fixed point (§15 caveat). I do not know whether this holds.

**E2 (P6; soft_norm).** At T=4 and T=16: median-loss ratio clip/np within [0.8, 1.25], and |fcat(clip) − fcat(np)| ≤ 0.1 pp at T.
- Kill: at T=16 the ratio exceeds 1.25 with interval excluding 1, or the fcat difference exceeds +0.1 pp with interval
  excluding 0. Then P6 fails for a real softmax block.
- Caveat fixed in advance: if the clipped fraction at the end of training is below 1% in soft_norm, E2 holding is weak
  evidence (nothing was clipped) and is reported as such.

**E3 (token normalization; lin_norm vs lin_raw).** At T=16 the E1 difference is smaller in lin_norm than in lin_raw:
[lin_raw erosion] − [lin_norm erosion] > 0, interval excluding 0.
- Kill: interval includes 0 or the sign is reversed. Then token normalization without a parameter constraint does not
  reduce erosion, and that ingredient of A6 is unsupported.

**E4 (P1, extrapolation drift; soft_norm, T=16).** Let D = med(dp)/med(clip). Fixed-T training: D_2T/D_T > 1.5, interval
on the log scale excluding 0. Random-T training: D_2T/D_T smaller than the fixed-T value.
- Kill: fixed-T interval includes 1. Then A1 has toy support only.
- This is the prediction I trust least: it rests on the two-parameter toy of §9, and a model trained at fixed T may
  extrapolate badly in every arm, which would swamp the noise effect. I do not know whether this holds.

## Not confirmatory
- P7 (gradient norm vs local contraction rate by parameter type). No clean operationalization here: in the linear loop
  the gradient profile also rises with the contraction rate up to rho ≈ 1 − 1/(2T), so a rank correlation does not
  separate the two cases. Reported descriptively: Spearman(gradient norm, rho_i), q99/median gradient norm.
- E4 for the linear arms; dp vs np; the jvp-based unstable fraction; the clipped fraction at the end of training.
- The Hessian-null-space control for P1 is deferred.

## What this pilot cannot show
Tiny models on a synthetic task; one noise level (eps = 2.11); three seeds; C fixed at the initialization median; Adam
only. A pass supports the mechanisms at toy-transformer scale, not for looped language models.

## Amendments
Any change after this file is written to the project is logged here with its date; results obtained under the earlier
design are kept.

(The text above this line is unchanged from the frozen version, SHA-256
`48d1761a86a8ed0c72bdedaa96d557583da7ac2934e82c652f3a67de4437da50`. File paths in it are the original project paths;
in the repository the files are `pilot/pilot.py`, `pilot/PREREG_PILOT.md` and `docs/dp-looped-theory.md`.)

**Amendment 1 (16 Nov 2026), logging only, made after 91 runs and before the last 29.**
- Reason: the review of 16 Nov pointed out that `pilot.py` zeroes non-finite per-prompt gradients in the clip and dp arms
  without reporting it. A gradient whose norm overflows is dropped the same way. Either would silently remove the most
  expansive prompts from training.
- Change: `pilot/pilot_v1_1.py` (SHA-256 `aae5b5ae0a6bc5bdfee47e40cf507b0525acec05f0368f9398ba28bbf8344a2b`) counts
  per-prompt gradients with a non-finite norm at every step (count, first step, largest per-step fraction), records the
  code hash, library versions, epsilon and configuration with every result, saves A for the linear arms, and adds
  finite-horizon stability measures. The training arithmetic is unchanged.
- Check: `pilot/audit_replay.py` re-ran all 45 completed evaluation-seed runs (36 linear, 9 softmax T=4) under v1.1.
  Every preregistered outcome is bit-identical to the frozen results (maximum relative difference 0), and the number of
  non-finite gradients is 0 in every run. So the zeroing never fired and the partial outcome below stands as reported.
- Not audited: the 46 selection-seed runs (seed 0), which only chose learning rates.
- The last 29 runs (soft_norm, T=16) were run under v1.1. Non-finite gradients: 0 in all 29.
- Not changed: the stability outcome. `f_unstable_exact` is the asymptotic criterion (spectral radius of I − A·Sigma_i
  above 1). The finite-horizon measures are reported as descriptive additions, not as replacements.

## Outcome, partial (15 Nov 2026): 91 of 120 runs
Done: every linear cell and the soft_norm T=4 cells. Pending: the soft_norm T=16 cells (29 runs, about 80 minutes on
this machine), which carry E2 at T=16 and all of E4. No run diverged. V1 passed for every completed cell.
Results file SHA-256 `a5ab959dca0e284b3161de1aeedb0601ac8976942853dfd4b6a89746a5557339`.

- **E1: main clause held, ordering clause failed, kill rule not triggered.**
  - lin_raw, T=16: unstable fraction 0.03–0.09% (np) vs 4.55–4.97% (clip). Difference +4.74 pp [+4.13, +5.36].
    Both arms selected the same learning rate (3e-3).
  - The clause "T=16 exceeds T=4" failed: T=4 gives +5.30 pp [+3.82, +6.78]. That clause was mis-specified: the theory
    doc's own §15 table has the unstable fraction falling with T while the loss consequence grows. It is the
    consequence that grows here too: fcat at T is 0.47% (T=4) vs 2.17% (T=16) under clipping; fcat at 2T, clip − np,
    is +1.49 pp vs +3.18 pp; mean loss is 0.63 vs 2.4×10⁹ (L0 units).
  - Typical prompts do *better* under clipping (median L/L0 at T=16: 0.00012 vs 0.0025). Clipped training takes a
    larger preconditioner, which helps the bulk and destabilizes the tail: the max-versus-quantile reading of Theorem 2.
  - At T=4 the two arms selected different learning rates (1e-3 vs 1e-2), so the T=4 difference is confounded with lr.
- **E3: held.** [lin_raw erosion] − [lin_norm erosion] at T=16 = +4.00 pp [+3.35, +4.65]. Token normalization cuts
  erosion from 4.74 to 0.75 pp. It does not remove it: lin_norm, T=16 is +0.75 pp [+0.46, +1.04].
- **E2 at T=4: held.** Median-loss ratio clip/np 1.029 [0.923, 1.146]; fcat difference +0.01 pp [−0.02, +0.03].
  The test is not vacuous: 19–32% of per-prompt gradients exceed C at the end of clipped training. T=16 pending.
- **E4: pending** (confirmatory cell is soft_norm, T=16). Descriptive so far: at T=4 noise costs soft_norm 2.84×
  [2.12, 3.80] in median loss at eps = 2.11, with D_2T/D_T = 0.67 [0.37, 1.21], i.e. no extrapolation drift at T=4.
- **Descriptive.**
  - Linear arms, dp vs clip: median-loss ratio 0.99–1.04 and the same unstable fractions (4.52–4.73% at T=16).
    The damage is from clipping; noise adds nothing in a 25-parameter model.
  - q99/median per-prompt gradient norm at the end of training: lin_raw T=16 70 (np) vs 1.8×10⁷ (clip); soft_norm T=4 7–9.
  - The jvp-based unstable fraction is 100% for soft_norm in every arm: the pre-LN block has an identity residual
    path, so this measure is uninformative there (as §11 of the theory doc anticipated).
  - Spearman(gradient norm, rho_i): 0.06–0.29 in the linear arms, about 0 in soft_norm.

## Outcome, final (16 Nov 2026): 120 of 120 runs
Results: `pilot/results/results_full_120.jsonl`, SHA-256 `5ff8194453bc04c49abd4115fd38bf34d75119919951c6c605b847134b6e764d`
(its first 91 lines are the partial file). No run diverged. V1 passed for every cell. Three evaluation seeds per cell;
intervals are 95% t-intervals with 2 degrees of freedom, which is weak evidence for every entry below.

**Passed**
- E1, main clause (lin_raw, T=16): +4.74 pp [+4.13, +5.36]; per seed +4.46, +4.84, +4.93. Same learning rate in both arms.
- E3 (token normalization): +4.00 pp [+3.35, +4.65].
- E2 at T=4 (soft_norm): median-loss ratio clip/np 1.029 [0.923, 1.146]; per seed 0.988, 1.078, 1.022.
  fcat difference +0.01 pp [−0.02, +0.03].

**Failed**
- E1, ordering clause: T=4 gives +5.30 pp, above T=16. Mis-specified (see the partial outcome). Not to be carried forward
  as a confirmatory test.
- E2 at T=16, band condition: the median-loss ratio clip/np is 0.513 [0.406, 0.649]; per seed 0.468, 0.566, 0.510.
  It is outside the predicted band [0.8, 1.25] because the clipped arm is *better*. The kill rule (harm from clipping)
  is not triggered; fcat is 0.00% in both arms.
- E4 (extrapolation drift, soft_norm, T=16): fixed-T D_2T/D_T = 0.669 [0.263, 1.701]; per seed 0.438, 0.899, 0.759.
  The interval includes 1 and the point estimate is on the wrong side. **Kill rule triggered: A1 has toy support only.**
  Random-T: 0.912 [0.749, 1.111], which is not smaller than the fixed-T value, so that clause fails too.

**Not interpretable**
- E2 at T=16 as a test of any mechanism. The arms selected different learning rates (clip 1e-2, np 3e-3), and np
  collapses at 1e-2 (validation median loss 0.355 against 0.018 at 3e-3). The clipped arm's advantage is at least
  partly "clipping permits a larger step". The clipped fraction at the end of training is 0.6–3.1%, straddling the 1%
  caveat fixed in advance. And this block is not the bounded softmax loop of the theory (theory document, §17).
- E4's D compares dp at lr 3e-3 with clip at lr 1e-2, so D_T = 10.2 mixes noise with learning rate. E4 gives no
  support for drift either way one reads it.
- Any comparison of the linear arms across T=4 and T=16: at T=4 np and clip selected different learning rates.

**Descriptive**
- Finite-horizon stability, linear arms (computed under v1.1 on the replayed runs; % of 16,384 test prompts, mean of 3 seeds):

  | cell | asymptotic (spectral radius > 1) | T-step gain > 1 | error grows ≥10× from T to 2T | one-step spectral norm > 1 |
  |---|---|---|---|---|
  | lin_raw T=16 np | 0.05 | 0.13 | 0.98 | 14.7 |
  | lin_raw T=16 clip | 4.80 | 5.59 | 3.69 | 96.2 |
  | lin_norm T=16 np | 0.02 | 0.06 | 0.12 | 88.3 |
  | lin_norm T=16 clip | 0.77 | 0.91 | 0.51 | 97.0 |

  E1 restated with the T-step gain: +5.46 pp [+4.99, +5.93]; with ≥10× error growth: +2.70 pp [+2.42, +2.99].
  The one-step spectral norm exceeds 1 for most prompts in every arm (the matrices are non-normal), so it does not
  separate the arms. 99th-percentile T-step gain at T=16: 0.88 (np) against 96 (clip) in lin_raw.
- Noise in the softmax block at a matched learning rate (dp and np both selected 3e-3): median-loss ratio dp/np is 2.93
  at T=4 and 5.21 at T=16, while the absolute excess is the same, 0.081 and 0.080 in L0 units. Post hoc; consistent
  with Prop 5 of the theory document (in-horizon excess loss independent of loop count), not a test of it.
- soft_norm, T=16, error at 2T: median L/L0 is 0.023 (np), 0.021 (clip), 0.132 (dp) against 0.019, 0.010, 0.099 at T.
  The fixed-T model does not blow up at 2T in any arm.
- Linear arms, dp against clip: unchanged from the partial outcome (noise adds nothing to the erosion).
