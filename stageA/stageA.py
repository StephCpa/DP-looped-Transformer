"""Stage A mechanism grid for stageA/PREREG_STAGE_A.md (JAX, CPU, float64).

Question: is the stability erosion of pilot E1 an accident of median clipping + Adam + one loop count?

Model: weight-tied linear loop doing in-context gradient descent with a trainable 5x5 preconditioner A.
  y_i <- y_i - (1/N) sum_{j in context} (x_i^T A x_j) y_j, T iterations; prediction = -y_T[query].
  In w-space the error obeys e_{t+1} = M_i e_t with M_i = I - A Sigma_i, Sigma_i = X_i^T X_i / N.
Variants:
  raw  raw Gaussian tokens, A free.
  tok  token-normalized prompts (every token scaled so ||x_j|| = sqrt(p)), A free.
  con  token-normalized prompts and a certified constraint: A = c * G / lambda_max(G), G = L L^T + 1e-6 I symmetric
       positive definite, c = ((2 - eps)/p) * sigmoid(a), eps = 0.1. Then ||A||_2 <= (2 - eps)/p, and since
       lambda_max(Sigma_i) <= tr(Sigma_i) = p for every token-normalized prompt, every eigenvalue of A Sigma_i lies in
       (0, 2 - eps]: the spectral radius of M_i is below 1 for every admissible prompt and every parameter value.
       (The certificate is asymptotic. M_i is non-normal, so its T-step gain can still exceed 1.)
All variants start cold, at a small isotropic step (0.05), so every arm has to train.
Arms: unclipped (C = None) and per-prompt clipping at C = c_mult x (median per-prompt gradient norm at initialization).
No noise in Stage A. Poisson sampling, q = 1/128, identical batches across arms within a seed.
Non-finite per-prompt gradients are never zeroed: the run stops, is marked invalid, and the step and fraction are recorded.

Usage:
  python3 stageA.py stage0 [--tier baselines]       (gates G1-G5; with --tier baselines also the unclipped-baseline table G6)
  python3 stageA.py run --tier select|a0|a1|a1tune_select|a1tune [--budget SECONDS]     (resumable)
  python3 stageA.py analyze
"""
import argparse, hashlib, itertools, json, math, os, platform, sys, time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax

P_DIM, N_CTX = 5, 10
CFG = dict(n_train=16384, n_test=16384, n_val=4096, n_cal=1024, q=1 / 128, bmax=192, steps=2000, eps_con=0.1,
           init_step=0.05, lrs=dict(adam=(3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1), sgd=(1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0)),
           sel_seed=10, a0_seeds=(11, 12, 13), a1_seeds=tuple(range(101, 111)))
VARIANTS, OPTS, TS, CS = ("raw", "tok", "con"), ("adam", "sgd"), (1, 4, 8, 16, 32), (0.1, 0.25, 0.5, 1.0, 2.0, 10.0)
# A1 cells: (variant, optimizer, T, c_mult); c_mult None is the unclipped reference
A1_CELLS = ([("raw", "adam", 16, c) for c in (None,) + CS] + [("raw", "sgd", 16, None), ("raw", "sgd", 16, 1.0),
            ("tok", "adam", 16, None), ("tok", "adam", 16, 1.0), ("con", "adam", 16, None), ("con", "adam", 16, 1.0),
            ("raw", "adam", 4, None), ("raw", "adam", 4, 1.0), ("raw", "adam", 32, None), ("raw", "adam", 32, 1.0)])
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "stageA_results.jsonl")
CODE_SHA = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
VERSIONS = dict(python=sys.version.split()[0], jax=jax.__version__, numpy=np.__version__, platform=platform.platform(), precision="float64")


# ----------------------------------------------------------------------------- data and model
_data_cache = {}
def make_data(seed, n, normalize):
    key = (seed, n, normalize)
    if key not in _data_cache:
        k = jax.random.split(jax.random.PRNGKey(seed), 2)
        X = jax.random.normal(k[0], (n, N_CTX + 1, P_DIM), dtype=jnp.float64)
        w = jax.random.normal(k[1], (n, P_DIM), dtype=jnp.float64) / jnp.sqrt(P_DIM)
        y = jnp.einsum("nkp,np->nk", X, w)
        if normalize:
            s = jnp.sqrt(P_DIM) / jnp.linalg.norm(X, axis=-1)
            X, y = X * s[..., None], y * s
        tok = jnp.concatenate([X, y[..., None]], -1).at[:, -1, -1].set(0.0)
        if len(_data_cache) > 12:
            _data_cache.clear()
        _data_cache[key] = (tok, y[:, -1])
    return _data_cache[key]


def init_params(variant, seed):
    z = jax.random.normal(jax.random.PRNGKey(10_000 + seed), (P_DIM, P_DIM), dtype=jnp.float64)
    # cold start: a small isotropic step (0.05), far below the trained values, so that every arm has to train
    s0 = CFG["init_step"]
    if variant == "con":
        cmax = (2 - CFG["eps_con"]) / P_DIM
        return dict(L=jnp.eye(P_DIM) + 0.005 * z, a=jnp.asarray(math.log((s0 / cmax) / (1 - s0 / cmax))))
    return dict(A=s0 * jnp.eye(P_DIM) + 0.005 * z)


def A_of(variant, params):
    if variant != "con":
        return params["A"]
    G = params["L"] @ params["L"].T + 1e-6 * jnp.eye(P_DIM)
    return ((2 - CFG["eps_con"]) / P_DIM) * jax.nn.sigmoid(params["a"]) * G / jnp.linalg.eigvalsh(G)[-1]


def predict(A, tok, T):
    X = tok[:, :P_DIM]
    K = (X @ A @ X[:-1].T) / N_CTX
    y, _ = lax.scan(lambda y_, _: (y_ - K @ y_[:-1], None), tok[:, P_DIM], None, length=T)
    return -y[-1]


def tree_norms(g):
    return jnp.sqrt(sum(jnp.sum(jnp.reshape(l, (l.shape[0], -1)) ** 2, axis=1) for l in jax.tree_util.tree_leaves(g)))


_step_cache = {}
def get_fns(variant, T, opt, clipped):
    key = ("con" if variant == "con" else "free", T, opt, clipped)
    if key in _step_cache:
        return _step_cache[key]
    loss_one = lambda params, tok, y: (predict(A_of(variant, params), tok, T) - y) ** 2
    psg = jax.jit(jax.vmap(jax.grad(loss_one), in_axes=(None, 0, 0)))

    @jax.jit
    def step(params, m, v, i, tok, y, wts, C, lr, qn):
        g = jax.vmap(jax.grad(loss_one), in_axes=(None, 0, 0))(params, tok, y)
        nrm = tree_norms(g)
        nbad = jnp.sum(wts * (~jnp.isfinite(nrm)))
        nclip = jnp.sum(wts * (nrm > C))
        fac = wts * jnp.minimum(1.0, C / (nrm + 1e-300)) if clipped else wts
        gbar = jax.tree_util.tree_map(lambda l: jnp.tensordot(fac, l, axes=1) / qn, g)
        if opt == "adam":
            m = jax.tree_util.tree_map(lambda a, b: 0.9 * a + 0.1 * b, m, gbar)
            v = jax.tree_util.tree_map(lambda a, b: 0.999 * a + 0.001 * b * b, v, gbar)
            upd = jax.tree_util.tree_map(lambda a, b: (a / (1 - 0.9 ** i)) / (jnp.sqrt(b / (1 - 0.999 ** i)) + 1e-8), m, v)
        else:
            upd = gbar
        params = jax.tree_util.tree_map(lambda p_, u: p_ - lr * u, params, upd)
        return params, m, v, nbad, nclip

    pred = jax.jit(jax.vmap(lambda A, t_: predict(A, t_, T), in_axes=(None, 0)))
    pred2 = jax.jit(jax.vmap(lambda A, t_: predict(A, t_, 2 * T), in_axes=(None, 0)))
    _step_cache[key] = (step, psg, pred, pred2)
    return _step_cache[key]


# ----------------------------------------------------------------------------- metrics
def metrics(A, tok, y, T, pred, pred2):
    out = {}
    A = np.array(A, dtype=np.float64)
    L0 = float(jnp.mean(y ** 2))
    with np.errstate(all="ignore"):
        eT = np.array((pred(jnp.asarray(A), tok) - y) ** 2); e2 = np.array((pred2(jnp.asarray(A), tok) - y) ** 2)
        eT = np.where(np.isfinite(eT), eT, np.inf); e2 = np.where(np.isfinite(e2), e2, np.inf)
        out.update(med_T=float(np.median(eT) / L0), q90_T=float(np.quantile(eT, 0.9) / L0), q99_T=float(np.quantile(eT, 0.99) / L0),
                   mean_T=float(np.mean(eT) / L0), med_2T=float(np.median(e2) / L0), fcat_T=float(np.mean(eT > 10 * L0)),
                   fcat_2T=float(np.mean(e2 > 10 * L0)), f_err10=float(np.mean(e2 > 10 * eT)))
        X = np.array(tok[:, :-1, :P_DIM]); S = np.einsum("nki,nkj->nij", X, X) / N_CTX
        M = np.eye(P_DIM)[None] - A[None] @ S
        MT = M.copy()
        for _ in range(T - 1):
            MT = MT @ M
        gain = np.linalg.norm(np.nan_to_num(MT, nan=1e300, posinf=1e300, neginf=-1e300), 2, axis=(1, 2))
        rad = np.max(np.abs(np.linalg.eigvals(M)), axis=1)
        out.update(f_gain=float(np.mean(gain > 1.0)),                      # PRIMARY: T-step gain of M_i above 1
                   f_asym=float(np.mean(rad > 1.0)),                       # secondary: spectral radius above 1
                   f_spec1=float(np.mean(np.linalg.norm(M, 2, axis=(1, 2)) > 1.0)),
                   gain_q50=float(np.quantile(gain, 0.5)), gain_q90=float(np.quantile(gain, 0.9)),
                   gain_q99=float(np.quantile(gain, 0.99)), gain_max=float(np.max(gain)),
                   A_specnorm=float(np.linalg.norm(A, 2)), A=A.round(8).tolist())
    return out


# ----------------------------------------------------------------------------- one run
def run(variant, opt, T, c_mult, seed, lr, steps=None, cfg=CFG):
    steps = steps or cfg["steps"]
    norm = variant != "raw"
    tok, y = make_data(1000 + seed, cfg["n_train"], norm)
    tok_te, y_te = make_data(2000 + seed, cfg["n_test"], norm)
    tok_va, y_va = make_data(3000 + seed, cfg["n_val"], norm)
    tok_cal, y_cal = make_data(4000 + seed, cfg["n_cal"], norm)
    clipped = c_mult is not None
    step, psg, pred, pred2 = get_fns(variant, T, opt, clipped)
    params = init_params(variant, seed)
    n0 = np.array(tree_norms(psg(params, tok_cal, y_cal)))
    res = dict(variant=variant, opt=opt, T=T, c_mult=c_mult, seed=seed, lr=lr, steps=steps, valid=True, invalid_reason=None,
               nonfinite_first_step=None, nonfinite_step_frac=None, code_sha256=CODE_SHA, versions=VERSIONS,
               cfg={k_: cfg[k_] for k_ in ("n_train", "n_test", "q", "bmax", "eps_con")})
    if not np.all(np.isfinite(n0)):
        res.update(valid=False, invalid_reason="non-finite per-prompt gradient at initialization", nonfinite_first_step=0,
                   nonfinite_step_frac=float(np.mean(~np.isfinite(n0))))
        return res
    C_init_med = float(np.median(n0)); C = C_init_med * c_mult if clipped else float("inf")
    qn = cfg["q"] * cfg["n_train"]
    m = jax.tree_util.tree_map(jnp.zeros_like, params); v = jax.tree_util.tree_map(jnp.zeros_like, params)
    rng = np.random.default_rng(seed); t0 = time.time(); clipfrac = []; A_800 = None
    for i in range(1, steps + 1):
        idx = np.flatnonzero(rng.random(cfg["n_train"]) < cfg["q"])[: cfg["bmax"]]
        w = np.zeros(cfg["bmax"]); w[: len(idx)] = 1.0
        pad = np.zeros(cfg["bmax"], np.int32); pad[: len(idx)] = idx
        new, m, v, nbad, nclip = step(params, m, v, float(i), tok[pad], y[pad], jnp.asarray(w), C, lr, qn)
        nbad = float(nbad)
        finite = all(bool(jnp.all(jnp.isfinite(l))) for l in jax.tree_util.tree_leaves(new))
        if nbad > 0 or not finite:
            # never zero, never continue: the run is invalid from this step on
            res.update(valid=False, nonfinite_first_step=i, nonfinite_step_frac=nbad / max(1, len(idx)),
                       invalid_reason="non-finite per-prompt gradient" if nbad > 0 else "non-finite parameters after update")
            break
        params = new
        if i > steps - 100:
            clipfrac.append(float(nclip) / max(1, len(idx)))
        if i == steps - 200:
            A_800 = np.array(A_of(variant, params))
    res["seconds"] = round(time.time() - t0, 1)
    res["C_init_median"] = C_init_med
    if res["valid"]:
        A = A_of(variant, params)
        res.update(metrics(A, tok_te, y_te, T, pred, pred2))
        res["val_med"] = float(jnp.median((pred(A, tok_va) - y_va) ** 2) / jnp.mean(y_va ** 2))
        if not math.isfinite(res["val_med"]) or not math.isfinite(res["med_T"]):
            res.update(valid=False, invalid_reason="non-finite evaluation loss")
        An = np.array(A); A0 = np.array(A_of(variant, init_params(variant, seed)))
        res.update(clip_frac_last100=float(np.mean(clipfrac)) if clipfrac else None,
                   rel_change_last200=float(np.linalg.norm(An - A_800) / np.linalg.norm(An)) if A_800 is not None else None,
                   dist_from_init=float(np.linalg.norm(An - A0)))
    return res


# ----------------------------------------------------------------------------- job lists
def load():
    rows = []
    if os.path.exists(RESULTS):
        rows = [json.loads(l) for l in open(RESULTS)]
    return rows


def key_of(r):
    return (r["tier"], r["variant"], r["opt"], r["T"], r["c_mult"], r["seed"], r["lr"])


def select_lr(rows, tier, variant, opt, T, c_mult):
    """learning rate with the lowest validation median loss on the selection seed; None until the grid is complete"""
    c = [r for r in rows if r["tier"] == tier and (r["variant"], r["opt"], r["T"], r["c_mult"]) == (variant, opt, T, c_mult)]
    if len(c) < len(CFG["lrs"][opt]):
        return None
    ok = [r for r in c if r["valid"]]
    return min(ok, key=lambda r: r["val_med"])["lr"] if ok else float("nan")


def jobs_for(tier, rows):
    S = CFG["sel_seed"]
    if tier == "select":           # shared learning rate: chosen on the UNCLIPPED arm only
        return [("select", v, o, T, None, S, lr) for v in VARIANTS for o in OPTS for T in TS for lr in CFG["lrs"][o]]
    if tier == "a0":
        out = []
        for v in VARIANTS:
            for o in OPTS:
                for T in TS:
                    lr = select_lr(rows, "select", v, o, T, None)
                    if lr is not None and not math.isnan(lr):
                        out += [("a0", v, o, T, c, s, lr) for c in (None,) + CS for s in CFG["a0_seeds"]]
        return out
    if tier == "a1":
        out = []
        for (v, o, T, c) in A1_CELLS:
            lr = select_lr(rows, "select", v, o, T, None)
            if lr is not None and not math.isnan(lr):
                out += [("a1", v, o, T, c, s, lr) for s in CFG["a1_seeds"]]
        return out
    if tier == "a1tune_select":    # secondary estimand: each clipped arm tunes its own learning rate
        return [("a1tune_select", v, o, T, c, S, lr) for (v, o, T, c) in A1_CELLS if c is not None for lr in CFG["lrs"][o]]
    if tier == "a1tune":
        out = []
        for (v, o, T, c) in A1_CELLS:
            if c is not None:
                lr = select_lr(rows, "a1tune_select", v, o, T, c)
                if lr is not None and not math.isnan(lr):
                    out += [("a1tune", v, o, T, c, s, lr) for s in CFG["a1_seeds"]]
        return out
    raise ValueError(tier)


def run_tier(tier, budget):
    rows = load(); done = {key_of(r) for r in rows}
    jobs = [j for j in jobs_for(tier, rows) if j not in done]
    jobs.sort(key=lambda j: (j[3], j[1], j[2]))        # group by T so compiled steps are reused
    print(f"tier {tier}: {len(jobs)} pending", flush=True)
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    t0 = time.time()
    for n, j in enumerate(jobs):
        if time.time() - t0 > budget:
            print(f"budget reached after {n} runs; rerun to continue"); return
        r = run(j[1], j[2], j[3], j[4], j[5], j[6]); r["tier"] = tier
        with open(RESULTS, "a") as fh:
            fh.write(json.dumps(r) + "\n")
        if n % 20 == 0:
            print(f"  {n}/{len(jobs)} {j[1]} {j[2]} T={j[3]} c={j[4]} seed={j[5]} lr={j[6]} valid={r['valid']} ({time.time() - t0:.0f}s)", flush=True)
    print("tier complete" if jobs_for(tier, load()) and all(j in {key_of(r) for r in load()} for j in jobs_for(tier, load())) else "pass finished")


# ----------------------------------------------------------------------------- statistics
def signflip_p(d):
    """exact one-sided sign-flip permutation p-value for H1: mean(d) > 0"""
    d = np.asarray(d, float); n = len(d); obs = d.mean()
    cnt = sum(1 for s in itertools.product((1.0, -1.0), repeat=n) if np.dot(s, d) / n >= obs - 1e-15)
    return cnt / 2 ** n


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run_max = 0.0
    for rank, k in enumerate(order):
        run_max = max(run_max, min(1.0, (m - rank) * ps[k])); adj[k] = run_max
    return adj


def tci(d):
    d = np.asarray(d, float); n = len(d)
    tc = {3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}.get(n, float("nan"))
    h = tc * d.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan")
    return d.mean(), d.mean() - h, d.mean() + h


def analyze():
    rows = load()
    def cell(tier, v, o, T, c):
        return {r["seed"]: r for r in rows if r["tier"] == tier and (r["variant"], r["opt"], r["T"], r["c_mult"]) == (v, o, T, c)}
    def erosion(tier, v, o, T, c, metric="f_gain", ref_tier=None):
        cl, ref = cell(tier, v, o, T, c), cell(ref_tier or tier, v, o, T, None)
        seeds = sorted(s for s in set(cl) & set(ref) if cl[s]["valid"] and ref[s]["valid"])
        return seeds, np.array([cl[s][metric] - ref[s][metric] for s in seeds]), len(set(cl) | set(ref)) - len(seeds)
    pp = lambda c: f"{100 * c[0]:+.2f} [{100 * c[1]:+.2f}, {100 * c[2]:+.2f}] pp"

    print("code:", sorted({r["code_sha256"][:16] for r in rows}), " rows:", len(rows), " invalid:", sum(not r["valid"] for r in rows))
    print("\n== learning rates (unclipped arm, selection seed) and baseline validity ==")
    for v in VARIANTS:
        for o in OPTS:
            print(f"  {v} {o}: " + ", ".join(f"T={T}: lr {select_lr(rows, 'select', v, o, T, None)}" for T in TS))

    print("\n== A0 screening (descriptive; mean erosion in pp over seeds, primary metric f_gain; '!' marks invalid runs) ==")
    for v in VARIANTS:
        for o in OPTS:
            print(f"  {v} / {o}   columns: C = " + ", ".join(str(c) for c in CS) + "   | unclipped f_gain % | unclipped median L/L0")
            for T in TS:
                cells = [erosion("a0", v, o, T, c) for c in CS]
                ref = cell("a0", v, o, T, None)
                txt = "  ".join((f"{100 * e.mean():+6.2f}" if len(e) else "   n/a") + ("!" if bad else " ") for (_, e, bad) in cells)
                rv = [r for r in ref.values() if r["valid"]]
                print(f"    T={T:2d}: {txt} | {100 * np.mean([r['f_gain'] for r in rv]) if rv else float('nan'):6.2f} | "
                      f"{np.mean([r['med_T'] for r in rv]) if rv else float('nan'):.4f}")

    for tier, label in (("a1", "PRIMARY estimand: shared learning rate (chosen on the unclipped arm)"),
                        ("a1tune", "SECONDARY estimand: each clipped arm tunes its own learning rate")):
        ref_tier = "a1"
        if not any(r["tier"] == tier for r in rows):
            continue
        print(f"\n== A1, {label} ==")
        E = lambda v, o, T, c, metric="f_gain": erosion(tier, v, o, T, c, metric, ref_tier)
        tests = {}
        s1, e1, b1 = E("raw", "adam", 16, 1.0); tests["H1 replication: E(raw, Adam, T=16, C=1) > 0"] = (e1, b1)
        per_c = [E("raw", "adam", 16, c) for c in CS]
        common = sorted(set.intersection(*[set(s) for (s, _, _) in per_c])) if all(len(s) for (s, _, _) in per_c) else []
        slopes = []
        for sd in common:
            ys = [e[list(s).index(sd)] for (s, e, _) in per_c]
            slopes.append(-np.polyfit(np.log10(CS), ys, 1)[0])          # sign flipped: positive means erosion falls as C rises
        tests["H2 dose-response: erosion falls as C rises (minus slope per decade > 0)"] = (np.array(slopes), sum(b for (_, _, b) in per_c))
        s3, e3, b3 = E("raw", "sgd", 16, 1.0); tests["H3 optimizer: E(raw, SGD, T=16, C=1) > 0"] = (e3, b3)
        for name, other in (("H4 token normalization: E(raw) - E(tok) > 0", "tok"), ("H5 constraint: E(raw) - E(con) > 0", "con")):
            so, eo, bo = E(other, "adam", 16, 1.0); cm = sorted(set(s1) & set(so))
            tests[name] = (np.array([e1[list(s1).index(k)] - eo[list(so).index(k)] for k in cm]), b1 + bo)
        s4, e4, b4 = E("raw", "adam", 4, 1.0); s32, e32, b32 = E("raw", "adam", 32, 1.0)
        ps = []; names = list(tests)
        for nm in names:
            d, bad = tests[nm]; ps.append(signflip_p(d) if len(d) >= 5 else float("nan"))
        p4, p32 = (signflip_p(e4) if len(e4) >= 5 else float("nan")), (signflip_p(e32) if len(e32) >= 5 else float("nan"))
        names.append("H6 loop count: E(raw, Adam, C=1) > 0 at T=4 and at T=32 (p = larger of the two)"); ps.append(max(p4, p32))
        adj = holm(np.array(ps)) if not any(math.isnan(p) for p in ps) else [float("nan")] * len(ps)
        for nm, p, a in zip(names, ps, adj):
            if nm.startswith("H6"):
                print(f"  {nm}\n      T=4: {pp(tci(e4))}, p={p4:.4f}; T=32: {pp(tci(e32))}, p={p32:.4f}; Holm-adjusted p = {a:.4f}; invalid pairs {b4 + b32}")
                print(f"      per seed T=4: {np.round(100 * e4, 2).tolist()}\n      per seed T=32: {np.round(100 * e32, 2).tolist()}")
            else:
                d, bad = tests[nm]
                unit = pp(tci(d)) if len(d) > 1 else "n/a"
                print(f"  {nm}\n      {unit}; sign-flip p = {p:.4f}; Holm-adjusted p = {a:.4f}; n = {len(d)}; invalid pairs {bad}")
                print(f"      per seed: {np.round(100 * d, 2).tolist()}")
        print("  descriptive, same cells: erosion by C (raw, Adam, T=16), all metrics in pp")
        for c in CS:
            print(f"    C={c:5}: f_gain {pp(tci(E('raw', 'adam', 16, c)[1]))} | f_asym {pp(tci(E('raw', 'adam', 16, c, 'f_asym')[1]))} | "
                  f"f_err10 {pp(tci(E('raw', 'adam', 16, c, 'f_err10')[1]))} | fcat_T {pp(tci(E('raw', 'adam', 16, c, 'fcat_T')[1]))}")
        print("  descriptive: per cell (means over valid seeds) lr, median / q90 / q99 / mean L/L0, gain q99, clipped fraction, relative change in the last 200 steps, invalid")
        for (v, o, T, c) in A1_CELLS:
            cc = cell(tier if c is not None else ref_tier, v, o, T, c)
            if cc:
                ok = [r for r in cc.values() if r["valid"]]
                f = lambda k_: np.mean([r[k_] for r in ok if r.get(k_) is not None]) if ok else float("nan")
                print(f"    {v:>3} {o:>4} T={T:2d} C={str(c):>5}: lr {list(cc.values())[0]['lr']:g}, {f('med_T'):.4g} / {f('q90_T'):.4g} / {f('q99_T'):.4g} / {f('mean_T'):.4g}, "
                      f"gain q99 {f('gain_q99'):.3g}, clipfrac {f('clip_frac_last100'):.3f}, rel change {f('rel_change_last200'):.3f}, invalid {len(cc) - len(ok)}/{len(cc)}")


# ----------------------------------------------------------------------------- stage 0
def stage0(baselines=False):
    print("G1 gradient check (float64):")
    tok, y = make_data(98, 4, False); params = init_params("raw", 98)
    _, psg, _, _ = get_fns("raw", 4, "adam", True)
    g = psg(params, tok, y)
    f = lambda A: float((predict(A, tok[0], 4) - y[0]) ** 2)
    E = np.zeros((5, 5)); E[1, 2] = 1e-6
    fd = (f(params["A"] + E) - f(params["A"] - E)) / 2e-6
    print(f"   autodiff {float(g['A'][0, 1, 2]):.10f} vs finite difference {fd:.10f}")
    assert abs(float(g["A"][0, 1, 2]) - fd) < 1e-7 * max(1, abs(fd))

    print("G2 constraint certificate: random parameters, token-normalized prompts")
    tokn, _ = make_data(97, 2000, True); X = np.array(tokn[:, :-1, :P_DIM]); S = np.einsum("nki,nkj->nij", X, X) / N_CTX
    rng = np.random.default_rng(0); worst_rad = 0.0; worst_norm = 0.0
    for k in range(200):
        prm = dict(L=jnp.asarray(rng.normal(size=(5, 5)) * 10 ** rng.uniform(-2, 2)), a=jnp.asarray(rng.uniform(-20, 20)))
        A = np.array(A_of("con", prm)); worst_norm = max(worst_norm, np.linalg.norm(A, 2))
        worst_rad = max(worst_rad, np.max(np.abs(np.linalg.eigvals(np.eye(5)[None] - A[None] @ S))))
    print(f"   max ||A||_2 = {worst_norm:.6f} (cap {(2 - CFG['eps_con']) / P_DIM:.2f}); max spectral radius of M_i over 200 x 2000 = {worst_rad:.6f}")
    assert worst_norm <= (2 - CFG["eps_con"]) / P_DIM + 1e-9 and worst_rad < 1.0
    assert np.max(np.trace(S, axis1=1, axis2=2)) <= P_DIM + 1e-9

    print("G3 non-finite handling: an unclipped SGD run with an absurd learning rate must stop and be marked invalid")
    r = run("raw", "sgd", 16, None, 99, 1e150, steps=50)
    print(f"   valid={r['valid']}, reason={r['invalid_reason']}, first step={r['nonfinite_first_step']}, fraction={r['nonfinite_step_frac']}")
    assert r["valid"] is False and r["nonfinite_first_step"] is not None

    print("G4 statistics: sign-flip and Holm on known inputs")
    assert abs(signflip_p(np.ones(10)) - 1 / 1024) < 1e-12 and abs(signflip_p(np.array([1.0, -1.0] * 5)) - 1.0) < 0.5
    assert np.allclose(holm(np.array([0.01, 0.04, 0.03])), [0.03, 0.06, 0.06])
    print("   ok")

    print("G5 clipped-path smoke (seed 99, 20 steps; only validity is inspected)")
    for v, o in (("raw", "adam"), ("tok", "sgd"), ("con", "adam")):
        r = run(v, o, 16, 0.5, 99, CFG["lrs"][o][1], steps=20)
        print(f"   {v} {o}: valid={r['valid']}"); assert r["valid"]

    if not baselines:
        return
    print("G6 baseline viability and learning-rate grids: UNCLIPPED arm only, seed 99 (not an analysis seed)")
    print("   validation median L/L0 by learning rate ('inv' = invalid run; move = ||A - A_init||_F; late change = relative change of A over the last 200 steps)")
    for v in VARIANTS:
        for o in OPTS:
            for T in (4, 16, 32):
                t0 = time.time(); out = []
                for lr in CFG["lrs"][o]:
                    r = run(v, o, T, None, 99, lr)
                    out.append(f"{lr:g}: " + (f"{r['val_med']:.4f} (move {r['dist_from_init']:.2f}, late change {r['rel_change_last200']:.3f})" if r["valid"] else "inv"))
                print(f"   {v} {o:>4} T={T:2d}: " + ", ".join(out) + f"   ({(time.time() - t0) / len(out):.1f}s per run)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--tier"); ap.add_argument("--budget", type=float, default=1e9)
    a = ap.parse_args()
    if a.cmd == "stage0": stage0(baselines=a.tier == "baselines")
    elif a.cmd == "run": run_tier(a.tier, a.budget)
    elif a.cmd == "analyze": analyze()
