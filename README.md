# DP × Looped Transformers

Is the Looped Transformer suitable for combination with differential privacy? Theory, numerical checks and a
preregistered pilot on small trained models.

Status (15 Nov 2026): theory at v0.3; pilot preregistered and 91 of 120 runs done. The softmax T=16 cells, which carry
the extrapolation-drift test (E4) and the second half of E2, are still pending.

## Contents

| Path | What it is |
|---|---|
| `docs/dp-looped-theory.md` | The theoretical analysis (v0.3): what is dead, what is alive, lemmas, propositions, theorems, predictions with kill criteria, prior-art gate |
| `checks/check_theory.py` | Numerical checks for Lemmas 1–3, Props 3, 4, 7 and the §9 toy (numpy, about 1 minute) |
| `checks/check_o1.py` | Looped in-context gradient descent under clipping: Prop 9, Theorem 2 (numpy, about 2.5 minutes) |
| `checks/check_softmax.py` | Softmax-attention loop: Props 10–11, clipping test, replication; matrix preconditioner (numpy, about 10 minutes) |
| `checks/outputs/` | Saved output of the three check scripts |
| `pilot/PREREG_PILOT.md` | Preregistration of the trained-model pilot, with the partial outcome appended |
| `pilot/pilot.py` | Pilot code (JAX, CPU). SHA-256 `6240354aeb844e0f3c78fa0ecb93377de704cb6bcf1138d09f78b2c02a11be09`, as recorded in the preregistration |
| `pilot/results/` | Raw results for the 91 completed runs (one JSON line per run) and the two run logs |

The documents refer to files by their original paths (`claude/dp-looped-theory.md`, `claude/dp-looped/...`); the table
above gives where each one lives here.

## Main findings so far

- DP-SGD puts no noise inside the loop, and its privacy proof does not depend on the loop. The loop affects utility only.
- The loop-specific effect is from clipping, not noise: stability of a loop is enforced by the very large gradients of
  the few samples on which it expands, and per-sample clipping caps exactly those (Theorem 2).
- On a trained looped linear transformer at T=16, clipping raises the fraction of unstable test prompts from
  0.03–0.09% to 4.6–5.0% (pilot E1, main clause). Token normalization cuts that erosion from 4.7 to 0.75 percentage
  points (E3). A softmax block shows no clipping penalty at T=4 (E2).
- Corrections and retractions are logged in the theory document's verdict section and in the preregistration.

## Reproducing

```
python3 checks/check_theory.py
python3 checks/check_o1.py
python3 checks/check_softmax.py

pip install jax
cd pilot
python3 pilot.py stage0                 # mechanics and baseline gates
python3 pilot.py grid --budget 400      # resumable; rerun until no jobs are pending
python3 pilot.py summarize              # preregistered analysis
```

To continue the pilot from the saved runs, copy `pilot/results/results_partial_91of120.jsonl` to `pilot/results.jsonl`
before running `grid`.

## Limits

Toy loops and tiny models on a synthetic task; three seeds; one noise level. Nothing here has been run on a looped
language model, and nothing has been checked in a proof assistant.
