# Differential Privacy Meets Looped Transformers

**Clipping turns stability from a maximum condition into a quantile condition.**

Interim results of mechanism research on combining differential privacy with looped (weight-tied, iterated)
transformers. This repository does not support the claim that DP generally harms looped transformers. It supports two
narrower statements:

1. The privacy accounting of DP-SGD does not change with loop depth.
2. Under per-sample clipping, the stability of a loop is a quantile condition governed by the threshold: the lower the
   threshold, the larger the unstable fraction. Proven for the fixed points of a linear in-context-gradient-descent
   loop and confirmed in trained models on ten seeds (Stage A, H2).

Not supported: that clipped training is less stable than unclipped training. The pilot suggested it (E1); the
preregistered replication failed (Stage A, H1), because unclipped training is itself erratic.

## Status (16 Nov 2026)

- Theory: `docs/dp-looped-theory.md`, v0.4. Every statement is tagged proven, proof sketch, conjecture or numerical
  observation in the table at the top of that document.
- Pilot: complete, 120 of 120 runs, preregistered. Three seeds per cell, so all of it is weak evidence.

| Preregistered test | Result |
|---|---|
| E1: clipping erodes stability in a trained linear loop (T=16) | Passed: unstable fraction 0.03–0.09% → 4.6–5.0% |
| E1, clause "larger at T=16 than at T=4" | Failed (mis-specified) |
| E3: token normalization reduces the erosion | Passed: 4.74 → 0.75 percentage points |
| E2: no clipping penalty in a softmax block, T=4 | Passed |
| E2 at T=16 | Failed its band in the favourable direction; not interpretable (learning-rate confound) |
| E4: extrapolation drift under DP noise | Failed; kill rule triggered |

- Review of 16 Nov and the response to it: `docs/review-2026-11-16-response.md`.
- Extrapolation drift (pilot E4) is closed: it is not a finding of this study.
- Stage A (`stageA/PREREG_STAGE_A.md`): complete, 1,188 runs. Of six confirmatory contrasts on ten new seeds, one
  passed (H2, the threshold dose-response) and five failed, including the replication of E1 (H1: −17.7 pp [−38.5, +3.1]).
  The unclipped reference ranged from 0.07% to 93% unstable across seeds; clipped arms were tight (0.65–1.01% at C=1).

## Contents

| Path | What it is |
|---|---|
| `docs/dp-looped-theory.md` | Theory, predictions with kill criteria, prior-art gate, corrections log, withdrawn material (Appendix A) |
| `docs/review-2026-11-16-response.md` | Point-by-point response to the review; proposed v2 grid |
| `checks/check_theory.py` | Lemmas 1–3, Props 3, 4, 7 and the two-parameter toy (about 1 minute) |
| `checks/check_o1.py` | Looped in-context gradient descent under clipping: Prop 9, Theorem 2 (about 2.5 minutes) |
| `checks/check_softmax.py` | Softmax toy loop: Props 10–11, clipping test, replication; matrix preconditioner (about 10 minutes) |
| `checks/outputs/` | Saved output of the three check scripts |
| `stageA/PREREG_STAGE_A.md` | Stage A preregistration (frozen text, hash recorded) and outcome |
| `stageA/results/` | All 1,188 runs, the analysis output, run logs, and one labelled exploratory file |
| `stageA/stageA.py` | Stage A code (float64; non-finite gradients stop and invalidate a run), SHA-256 `54d4ff81…` |
| `pilot/PREREG_PILOT.md` | Preregistration (frozen text, hash recorded), Amendment 1, partial and final outcomes |
| `pilot/pilot.py` | Frozen pilot code, SHA-256 `6240354a…`; used for the first 91 runs |
| `pilot/pilot_v1_1.py` | Amendment 1: adds logging of non-finite gradients, provenance fields and finite-horizon stability measures; training arithmetic unchanged; used for the last 29 runs |
| `pilot/audit_replay.py` | Replays completed runs under v1.1 and compares them with the frozen results |
| `pilot/results/` | `results_full_120.jsonl` (all runs), `results_partial_91of120.jsonl` (its first 91 lines, as hashed in the preregistration), `audit_replay_v1_1.jsonl`, run logs |

`pilot/pilot.py` and the frozen part of `pilot/PREREG_PILOT.md` are kept byte-identical to what was hashed, so they still
mention the original project paths (`claude/dp-looped/...`). Everything else uses the paths above.

## Reproducing

Environment used for every number here: Python 3.13.18, NumPy 2.5.3, JAX 0.12.0.1 (CPU, float32 for the pilot,
float64 for the spectral diagnostics), Linux x86_64.

```
pip install -r requirements.txt

python3 checks/check_theory.py      # asserts its bounds; exits non-zero on failure
python3 checks/check_o1.py
python3 checks/check_softmax.py

cd pilot
python3 pilot_v1_1.py stage0                  # mechanics and baseline gates
python3 pilot_v1_1.py grid --budget 400       # resumable; rerun until no jobs are pending
python3 pilot_v1_1.py summarize               # preregistered analysis
python3 audit_replay.py results/results_partial_91of120.jsonl replay.jsonl lin   # audit of the frozen runs
```

`grid` and `summarize` read and write `pilot/results.jsonl`; copy `results/results_full_120.jsonl` there to rerun the
analysis without retraining.

## Limits

- Toy loops and tiny models on a synthetic task; three seeds; one clipping threshold; one noise level.
- The pilot's softmax block is not the bounded softmax loop analysed in the theory (theory document, §17).
- The hidden-state-noise result (Prop 7) is proven only for the isotropic linear loop.
- Nothing has been run on a looped language model, and nothing has been checked in a proof assistant.
