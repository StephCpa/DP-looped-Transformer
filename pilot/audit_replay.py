"""Replay completed pilot runs with pilot_v1_1.py and compare with the frozen results.

Purpose (Amendment 1): (a) show that v1.1 reproduces the frozen pilot.py outputs, (b) report how many per-prompt
gradients were non-finite (and therefore silently zeroed) in the runs behind the partial outcome.
Usage: python3 audit_replay.py OLD_RESULTS.jsonl OUT.jsonl ARCH_PREFIX [BUDGET_SECONDS]
"""
import json, math, sys, time
import pilot_v1_1 as p

old_path, out_path, prefix = sys.argv[1:4]
budget = float(sys.argv[4]) if len(sys.argv) > 4 else 1e9
KEYS = ("med_T", "med_2T", "mean_T", "mean_2T", "fcat_T", "fcat_2T", "f_unstable", "f_unstable_exact", "het_q99", "val_med", "C")
done = set()
try:
    for l in open(out_path):
        r = json.loads(l); done.add((r["arch"], r["T"], r["arm"], r["randT"], r["seed"], r["lr"]))
except FileNotFoundError:
    pass
t0 = time.time()
for l in open(old_path):
    o = json.loads(l)
    k = (o["arch"], o["T"], o["arm"], o["randT"], o["seed"], o["lr"])
    if not o["arch"].startswith(prefix) or o["seed"] not in p.CFG["eval_seeds"] or k in done:
        continue
    if time.time() - t0 > budget:
        print("budget reached"); break
    r = p.run(o["arch"], o["T"], o["arm"], o["seed"], o["lr"], randT=o["randT"])
    worst = 0.0
    for key in KEYS:
        if key in o:
            a, b = o[key], r[key]
            d = 0.0 if a == b else (abs(a - b) / max(abs(a), abs(b), 1e-300) if math.isfinite(a) and math.isfinite(b) else float("inf"))
            worst = max(worst, d)
    r["replay_max_rel_diff"] = worst
    open(out_path, "a").write(json.dumps(r) + "\n")
    print(f"{o['arch']:>9} T={o['T']:2d} {o['arm']:>4} seed {o['seed']}: max rel diff vs frozen {worst:.1e}; "
          f"non-finite gradients {r['nonfinite_grad_count']:.0f} (first step {r['nonfinite_first_step']}, max step fraction {r['nonfinite_max_step_frac']:.3f})", flush=True)
