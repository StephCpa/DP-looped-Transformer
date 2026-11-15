# Response to the review of 16 Nov 2026

The review's main judgement is accepted: this is interim mechanism research, and the supportable core is (1) DP-SGD's
privacy accounting does not depend on loop depth, and (2) per-sample clipping moves the stability condition of a loop from
a maximum over samples to a quantile. The theory document's title and opening now say exactly that.

## Point by point

| # | Review point | Status | What was done |
|---|---|---|---|
| 1 | Finish the pilot; four-way summary; drop the failed E1 ordering clause | **Done** | 120/120 runs. `pilot/PREREG_PILOT.md` has the final outcome split into passed / failed / not interpretable / descriptive. The ordering clause stays on record as failed and is listed in the theory document's Appendix A as not to be carried forward |
| 2 | Non-finite gradients silently zeroed | **Audited; logging fixed; numerically stable computation deferred to v2** | `pilot/pilot_v1_1.py` counts them (count, first step, largest per-step fraction). `pilot/audit_replay.py` re-ran the 45 completed evaluation runs: outcomes bit-identical, 0 non-finite gradients. The 29 new runs: 0. The zeroing never fired, so no reported number changes |
| 3 | Stability definition | **Done for the linear arms** | Docstring corrected to the spectral-radius statement. Added: one-step spectral norm, T-step gain of the non-normal iteration matrix, fraction of prompts whose error grows from T to 2T. E1 holds under the T-step gain (+5.46 pp [+4.99, +5.93]) |
| 4 | Prop 7 is not a theorem | **Done** | Restated: proven for the isotropic linear loop by direct Gaussian computation; proof sketch and candidate bound otherwise. Setting made explicit (range of rho, projection, input injection, noise placement) and the four missing proof steps listed |
| 5 | Pilot softmax block ≠ softmax toy | **Done (documentation); the missing arm is deferred to v2** | New §17 in the theory document separates the two models, what E2 can show, and what is shared. A6 is now marked toy-level |
| 6 | Weak statistics | **Accepted; deferred to v2** | Every interval is flagged as df = 2 in the outcome. Per-seed paired effects are now reported for the confirmatory tests. The learning-rate confounds are listed under "not interpretable" |
| 7 | Sweeps over threshold and privacy strength | **Proposed below; not run** | Needs a decision on scope first |
| 8 | Reproducibility | **Mostly done** | `requirements.txt` pins JAX and NumPy and records the Python version; every v1.1 result row carries code hash, versions, epsilon and configuration; check scripts now assert their bounds and stop on failure; `.DS_Store` ignored; paths in the theory document, README and scripts are repository paths |
| – | Editorial cleanup | **Done** | H2 and P3 moved to Appendix A; the duplicate P3 row removed; a statement-status table tags every result as proven, proof sketch, conjecture or numerical observation |
| – | Title | **Adopted** | "Differential Privacy Meets Looped Transformers: Clipping Turns Stability from a Maximum Condition into a Quantile Condition" |

## Where I differ from the review, or could not comply

- **Direction of the non-finite-gradient bias (point 2).** The review says zeroing "may artificially suppress
  instability". In this model it would do the opposite: the prompts whose gradients overflow are the expansive ones, and
  their clipped gradient is the force that pulls the preconditioner back. Dropping them removes that force and would
  inflate the erosion measured in E1. Either direction would have invalidated the result; the audit shows it did not
  happen.
- **Frozen paths (point 8).** The frozen part of `PREREG_PILOT.md` still contains the original project paths. Editing
  it would break the recorded hash, so the mapping is given in the Amendments section instead. `pilot/pilot.py` is
  likewise left byte-identical; the corrected docstring is in `pilot_v1_1.py`.
- **Unaudited runs.** The 46 selection-seed runs were not replayed. They only chose learning rates.
- **Assertions.** The statistical toy runs in the check scripts (noisy gradient descent, clipping fixed points across
  draws) print their numbers and are not asserted; only the bounds and identities are.

## What the completed pilot changes

- E4 failed and its kill rule fired: extrapolation drift (A1), the precise form of the original "fixed-point drift"
  hypothesis, has toy support only. It should not appear in a paper as a finding.
- E2 at T=16 landed outside its band because clipping *helped* the softmax block (ratio 0.51). The arms used different
  learning rates and the unclipped arm collapses at the larger one, so this is "clipping permits a larger step", not
  evidence about loop stability.
- One post hoc observation worth a preregistered test: at a matched learning rate the absolute excess loss from DP
  noise in the softmax block is the same at T=4 and T=16 (0.081 and 0.080 in L0 units), as Prop 5 predicts.

## Proposed v2 mechanism grid (points 6 and 7): needs a decision before preregistration

The review's table is a 6 × 4 × 5 × 2 × 3 full factorial, 720 cells before seeds. Proposed staging instead:

| Stage | Question | Cells | Runs at 10 seeds | Cost on 2 CPU cores |
|---|---|---|---|---|
| A | Is erosion specific to median clipping with Adam? Linear arms, clip only | C in {0.1, 0.25, 0.5, 1, 2, 10} × T in {1, 4, 8, 16, 32} × {raw, token norm, step-size constraint} × {SGD, Adam}, plus unclipped references | 2,100 | about 5 hours |
| B | Does noise change it? Linear arms | sigma in {0.5, 1, 2, 4} × C in {0.5, 1, 2} × T in {4, 16} × {raw, token norm}, Adam | 480 | about 1 hour |
| C | Softmax: toy-faithful convex-combination block against the pre-LN block | C in {0.1, 1, 10} × T in {4, 16} × 2 blocks, clip only | 120 plus references | about 6 hours |

Fixed across stages:

- One learning rate per (optimizer, T, normalization), chosen on the unclipped arm and shared by every clipped and
  noised arm, so that no comparison is confounded by learning rate.
- float64 for the linear arms; a run with any non-finite per-prompt gradient is classed as not interpretable.
- Outcomes: unstable fraction (asymptotic and T-step gain), median, mean, q90 and q99 loss, error growth from T to 2T.
- Analysis: 10 seeds; cluster bootstrap over seeds for fractions; log scale for loss ratios; Holm correction across the
  confirmatory family; per-seed paired effects reported.
- Confirmatory predictions limited to the maximum-to-quantile claim: erosion increases as C decreases, is present under
  SGD as well as Adam, and is removed by a step-size constraint. Everything else descriptive.

Decision needed: whether to run all three stages or only A, and whether 10 seeds is the right budget.
