"""Looped-transformer DP pilot, v1.1 (Amendment 1 to pilot/PREREG_PILOT.md, 16 Nov 2026). JAX, CPU.

v1.1 differs from the frozen pilot.py (SHA-256 6240354a...) only in logging and descriptive outputs:
  - counts per-prompt gradients whose norm is non-finite (these are zeroed in the clip/dp arms and were not reported),
    with the first step at which one occurs and the largest per-step fraction;
  - records code hash, library versions, epsilon and configuration with every result; saves A for the linear arms;
  - adds finite-horizon stability measures (see diagnostics) and the fraction of prompts whose error grows from T to 2T.
The training arithmetic is unchanged; audit_replay.py checks that replayed runs reproduce the frozen results.

Task: in-context linear regression. A prompt is N context pairs (x_j, y_j = w.x_j) plus a query x_q; target w.x_q.
The privacy unit is the prompt.

Architectures (all weight-tied loops, T iterations):
  lin_raw   looped linear transformer on raw tokens, sparse parametrization (Ahn et al. / Gatmiry et al.):
            the x-slots are fixed and the y-slots follow  y_i <- y_i - (1/N) sum_{j in context} (x_i^T A x_j) y_j,
            with one trainable p x p matrix A; prediction = -y_T[query]. This is in-context gradient descent with
            preconditioner A. The w-space error obeys e_{t+1} = (I - A Sigma_i) e_t, so prompt i is asymptotically
            stable iff the spectral radius of I - A Sigma_i is below 1, i.e. every eigenvalue lam of A Sigma_i
            (possibly complex: A is not symmetric) satisfies |1 - lam| < 1.
            (Stage 0: the dense P, Q form gave non-finite per-prompt gradients at initialization at T=16.)
  lin_norm  the same block on token-normalized prompts (each token scaled so that ||x_j|| = sqrt(p)).
  soft_norm pre-LayerNorm softmax-attention block with input injection (Yang et al. style, small):
            U = Z + P; H = U + Attn(LN(U)); Z <- H + MLP(LN(H)); prediction = head(LN(Z_T[query])). Raw tokens.

Arms: np (no clipping, no noise), clip (per-prompt clipping at C, no noise), dp (clipping + Gaussian noise).
Sampling: Poisson with rate q for every arm; the accountant is RDP for the Poisson-subsampled Gaussian mechanism.

Usage:
  python3 pilot.py stage0                       # mechanics and baseline-viability gates (no analysis seeds)
  python3 pilot.py run --arch lin_raw --T 16 --arm clip --seed 1 --lr 3e-3 [--randT] [--out results.jsonl]
  python3 pilot.py grid [--budget SECONDS]       # run the preregistered grid; resumable (rerun until no jobs are pending;
                                                 # evaluation seeds are scheduled once a cell's lr selection is complete)
  python3 pilot.py summarize [--out results.jsonl]
"""
import argparse, json, math, os, sys, time
import numpy as np
import jax, jax.numpy as jnp
from jax import lax

P_DIM, N_CTX, D_MODEL, D_MLP = 5, 10, 32, 64
CFG = dict(n_train=16384, n_test=16384, n_val=4096, n_diag=1024, q=1 / 128, bmax=192, steps=1000,
           sigma_dp=1.0, delta=1e-5, lrs=(1e-3, 3e-3, 1e-2), sel_seed=0, eval_seeds=(1, 2, 3))
ARCHS, TS, ARMS = ("lin_raw", "lin_norm", "soft_norm"), (4, 16), ("np", "clip", "dp")
import hashlib, platform
CODE_SHA = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
VERSIONS = dict(python=sys.version.split()[0], jax=jax.__version__, numpy=np.__version__, platform=platform.platform(),
                precision="float32 (jax default); float64 for the spectral diagnostics")


# ----------------------------------------------------------------------------- data
def make_data(seed, n, normalize):
    k = jax.random.split(jax.random.PRNGKey(seed), 2)
    X = jax.random.normal(k[0], (n, N_CTX + 1, P_DIM))
    w = jax.random.normal(k[1], (n, P_DIM)) / jnp.sqrt(P_DIM)
    y = jnp.einsum("nkp,np->nk", X, w)
    if normalize:
        s = jnp.sqrt(P_DIM) / jnp.linalg.norm(X, axis=-1)
        X, y = X * s[..., None], y * s
    tok = jnp.concatenate([X, y[..., None]], -1)
    tok = tok.at[:, -1, -1].set(0.0)
    return tok, y[:, -1]


# ----------------------------------------------------------------------------- models
def init_params(arch, seed):
    k = jax.random.split(jax.random.PRNGKey(10_000 + seed), 8)
    D = P_DIM + 1
    if arch.startswith("lin"):
        # start at a small isotropic step (0.3) plus a random perturbation
        return dict(A=0.3 * jnp.eye(P_DIM) + 0.05 * jax.random.normal(k[0], (P_DIM, P_DIM)))
    d, m = D_MODEL, D_MLP
    g = lambda kk, a, b: jax.random.normal(kk, (a, b)) / jnp.sqrt(a)
    return dict(Win=g(k[0], D, d), bin=jnp.zeros(d), Wq=g(k[1], d, d), Wk=g(k[2], d, d), Wv=g(k[3], d, d),
                Wo=0.5 * g(k[4], d, d), W1=g(k[5], d, m), b1=jnp.zeros(m), W2=0.5 * g(k[6], m, d), b2=jnp.zeros(d),
                g1=jnp.ones(d), c1=jnp.zeros(d), g2=jnp.ones(d), c2=jnp.zeros(d), gf=jnp.ones(d), cf=jnp.zeros(d),
                wout=0.1 * jax.random.normal(k[7], (d,)), bout=jnp.zeros(()))


def ln(x, g, c):
    mu = x.mean(-1, keepdims=True); v = x.var(-1, keepdims=True)
    return g * (x - mu) / jnp.sqrt(v + 1e-5) + c


def lin_state(params, tok, T, Tmax, Z0=None):
    X = tok[:, :P_DIM]
    K = (X @ params["A"] @ X[:-1].T) / N_CTX                      # (N+1, N): query row included, query column excluded
    def step(y, t):
        yn = y - K @ y[:-1]
        return jnp.where(t < T, yn, y), None
    y, _ = lax.scan(step, tok[:, P_DIM] if Z0 is None else Z0, jnp.arange(Tmax))
    return y


def soft_state(params, tok, T, Tmax, Z0=None):
    p = params
    Pe = tok @ p["Win"] + p["bin"]
    def step(Z, t):
        U = Z + Pe
        h = ln(U, p["g1"], p["c1"])
        A = jax.nn.softmax((h @ p["Wq"]) @ (h @ p["Wk"]).T / jnp.sqrt(D_MODEL), axis=-1)
        H = U + (A @ (h @ p["Wv"])) @ p["Wo"]
        Zn = H + jax.nn.gelu(ln(H, p["g2"], p["c2"]) @ p["W1"] + p["b1"]) @ p["W2"] + p["b2"]
        return jnp.where(t < T, Zn, Z), None
    Z, _ = lax.scan(step, jnp.zeros_like(Pe) if Z0 is None else Z0, jnp.arange(Tmax))
    return Z


def predict(arch, params, tok, T, Tmax):
    if arch.startswith("lin"):
        return -lin_state(params, tok, T, Tmax)[-1]
    Z = soft_state(params, tok, T, Tmax)
    return ln(Z[-1], params["gf"], params["cf"]) @ params["wout"] + params["bout"]


def tree_norms(g):
    return jnp.sqrt(sum(jnp.sum(jnp.reshape(l, (l.shape[0], -1)) ** 2, axis=1) for l in jax.tree_util.tree_leaves(g)))


# ----------------------------------------------------------------------------- accountant
def rdp_epsilon(q, sigma, steps, delta, orders=range(2, 129)):
    """epsilon for `steps` compositions of the Poisson-subsampled Gaussian mechanism (integer Renyi orders)."""
    best = float("inf")
    for a in orders:
        logs = [math.lgamma(a + 1) - math.lgamma(k + 1) - math.lgamma(a - k + 1) + (a - k) * math.log1p(-q)
                + (k * math.log(q) if k else 0.0) + (k * k - k) / (2 * sigma ** 2) for k in range(a + 1)]
        mx = max(logs); loga = mx + math.log(sum(math.exp(l - mx) for l in logs))
        best = min(best, steps * loga / (a - 1) + math.log(1 / delta) / (a - 1))
    return best


# ----------------------------------------------------------------------------- training
def build(arch, T, arm, randT, C, sigma, lr, qn):
    Tmax = T
    def loss_one(params, tok, y, Tcur):
        return (predict(arch, params, tok, Tcur, Tmax) - y) ** 2
    psg = jax.vmap(jax.grad(loss_one), in_axes=(None, 0, 0, None))

    @jax.jit
    def step(params, m, v, i, tok, y, wts, key, Tcur):
        g = psg(params, tok, y, Tcur)
        nbad = jnp.sum(wts * (~jnp.isfinite(tree_norms(g))))        # v1.1: logging only
        if arm == "np":
            fac = wts
        else:
            g = jax.tree_util.tree_map(lambda l: jnp.nan_to_num(l, nan=0.0, posinf=0.0, neginf=0.0), g)
            nrm = tree_norms(g)
            fac = wts * jnp.minimum(1.0, C / (nrm + 1e-12))
        keys = jax.random.split(key, len(jax.tree_util.tree_leaves(params)))
        leaves, treedef = jax.tree_util.tree_flatten(g)
        out = []
        for l, kk in zip(leaves, keys):
            s = jnp.tensordot(fac, l, axes=1)
            if arm == "dp":
                s = s + sigma * C * jax.random.normal(kk, s.shape)
            out.append(s / qn)
        gbar = jax.tree_util.tree_unflatten(treedef, out)
        m = jax.tree_util.tree_map(lambda a, b: 0.9 * a + 0.1 * b, m, gbar)
        v = jax.tree_util.tree_map(lambda a, b: 0.999 * a + 0.001 * b * b, v, gbar)
        mh = jax.tree_util.tree_map(lambda a: a / (1 - 0.9 ** i), m)
        vh = jax.tree_util.tree_map(lambda a: a / (1 - 0.999 ** i), v)
        params = jax.tree_util.tree_map(lambda p_, a, b: p_ - lr * a / (jnp.sqrt(b) + 1e-8), params, mh, vh)
        return params, m, v, nbad

    return step, psg


def evaluate(arch, params, tok, y, T):
    out = {}; errs = {}
    L0 = float(jnp.mean(y ** 2))
    for name, Te in (("T", T), ("2T", 2 * T)):
        f = jax.jit(jax.vmap(lambda t_: predict(arch, params, t_, Te, Te)))
        e = np.array(jnp.concatenate([(f(tok[i:i + 2048]) - y[i:i + 2048]) ** 2 for i in range(0, tok.shape[0], 2048)]))
        e = np.where(np.isfinite(e), e, np.inf)
        out[f"mean_{name}"] = float(np.mean(e) / L0) if np.all(np.isfinite(e)) else float("inf")
        out[f"med_{name}"] = float(np.median(e) / L0)
        out[f"fcat_{name}"] = float(np.mean(e > 10 * L0))
        errs[name] = e
    with np.errstate(all="ignore"):
        out["f_err_grow"] = float(np.mean(errs["2T"] > errs["T"]))      # v1.1: error larger at 2T than at T
        out["f_err_grow10"] = float(np.mean(errs["2T"] > 10 * errs["T"]))
    return out


def diagnostics(arch, params, tok, y, T, psg, C, tok_big):
    """finite-time expansion rate rho_i (two random tangents) and per-prompt gradient norms on the diagnostic prompts"""
    state = lin_state if arch.startswith("lin") else soft_state
    def rho(t_, key):
        Z0 = t_[:, P_DIM] if arch.startswith("lin") else jnp.zeros((N_CTX + 1, D_MODEL))
        tans = jax.random.normal(key, (2,) + Z0.shape)
        def one(tan):
            _, dz = jax.jvp(lambda z: state(params, t_, T, T, Z0=z), (Z0,), (tan,))
            return jnp.log(jnp.linalg.norm(dz) / jnp.linalg.norm(tan)) / T
        return jnp.exp(jnp.mean(jax.vmap(one)(tans)))
    keys = jax.random.split(jax.random.PRNGKey(7), tok.shape[0])
    r = np.array(jax.jit(jax.vmap(rho))(tok, keys))
    gn = np.array(tree_norms(psg(params, tok, y, T)))
    ok = np.isfinite(gn) & np.isfinite(r)
    rk = lambda a: np.argsort(np.argsort(a))
    sp = float(np.corrcoef(rk(gn[ok]), rk(r[ok]))[0, 1]) if ok.sum() > 10 else float("nan")
    out = dict(f_unstable=float(np.mean(~np.isfinite(r) | (r > 1.0))), rho_med=float(np.nanmedian(r)),
               het_q99=float(np.nanquantile(gn, 0.99) / np.nanmedian(gn)), spearman_g_rho=sp,
               clip_frac_end=float(np.mean(~np.isfinite(gn) | (gn > C))))
    if arch.startswith("lin"):
        # exact criterion: prompt i is unstable iff the spectral radius of I - A Sigma_i exceeds 1
        X = np.array(tok_big[:, :-1, :P_DIM], dtype=np.float64)
        S = np.einsum("nki,nkj->nij", X, X) / N_CTX
        ev = np.linalg.eigvals(np.array(params["A"], dtype=np.float64)[None] @ S)
        out["f_unstable_exact"] = float(np.mean(np.max(np.abs(1 - ev), axis=1) > 1.0))
        # v1.1: finite-horizon measures for the (non-normal) iteration matrix M_i = I - A Sigma_i
        M = np.eye(P_DIM)[None] - np.array(params["A"], dtype=np.float64)[None] @ S
        MT = M.copy()
        for _ in range(T - 1):
            MT = MT @ M
        with np.errstate(all="ignore"):
            out["f_specnorm_gt1"] = float(np.mean(np.linalg.norm(M, 2, axis=(1, 2)) > 1.0))     # one-step gain > 1
            gT = np.linalg.norm(np.nan_to_num(MT, nan=1e300, posinf=1e300, neginf=-1e300), 2, axis=(1, 2))
            out["f_gainT_gt1"] = float(np.mean(gT > 1.0))                                   # T-step gain > 1
            out["gainT_q99"] = float(np.quantile(gT, 0.99)); out["gainT_max"] = float(np.max(gT))
        out["A"] = np.array(params["A"], dtype=np.float64).round(6).tolist()
    return out


def run(arch, T, arm, seed, lr, randT=False, steps=None, verbose=False, cfg=CFG):
    steps = steps or cfg["steps"]
    norm = arch == "lin_norm"
    tok, y = make_data(1000 + seed, cfg["n_train"], norm)
    tok_te, y_te = make_data(2000 + seed, cfg["n_test"], norm)
    tok_va, y_va = make_data(3000 + seed, cfg["n_val"], norm)
    tok_cal, y_cal = make_data(4000 + seed, cfg["n_diag"], norm)       # public calibration prompts for C
    tok_dg, y_dg = make_data(5000 + seed, cfg["n_diag"], norm)
    params = init_params(arch, seed)
    qn = cfg["q"] * cfg["n_train"]
    step0, psg = build(arch, T, "clip", randT, 1.0, 0.0, lr, qn)
    n0 = np.array(tree_norms(psg(params, tok_cal, y_cal, T)))
    C = float(np.nanmedian(np.where(np.isfinite(n0), n0, np.nan))); nonfinite0 = float(np.mean(~np.isfinite(n0)))
    step, psg = build(arch, T, arm, randT, C, cfg["sigma_dp"], lr, qn)
    m = jax.tree_util.tree_map(jnp.zeros_like, params); v = jax.tree_util.tree_map(jnp.zeros_like, params)
    rng = np.random.default_rng(seed); key = jax.random.PRNGKey(20_000 + seed)
    vfun = jax.jit(jax.vmap(lambda p_, t_: predict(arch, p_, t_, T, T), in_axes=(None, 0)))
    diverged = False; t0 = time.time(); nf_total = 0.0; nf_first = None; nf_maxfrac = 0.0
    for i in range(1, steps + 1):
        idx = np.flatnonzero(rng.random(cfg["n_train"]) < cfg["q"])[: cfg["bmax"]]
        w = np.zeros(cfg["bmax"], np.float32); w[: len(idx)] = 1.0
        pad = np.zeros(cfg["bmax"], np.int32); pad[: len(idx)] = idx
        Tcur = int(rng.integers(T // 2, T + 1)) if randT else T
        key, sub = jax.random.split(key)
        params, m, v, nb = step(params, m, v, i, tok[pad], y[pad], jnp.asarray(w), sub, Tcur)
        nb = float(nb)
        if nb > 0:
            nf_total += nb; nf_first = nf_first or i; nf_maxfrac = max(nf_maxfrac, nb / max(1, len(idx)))
        if i % 100 == 0 or i == steps:
            lv = float(jnp.median((vfun(params, tok_va[:512]) - y_va[:512]) ** 2))
            if verbose:
                print(f"    step {i:5d}  val median loss {lv:.4f}  ({time.time() - t0:.0f}s)", flush=True)
            if not math.isfinite(lv):
                diverged = True; break
    res = dict(arch=arch, T=T, arm=arm, seed=seed, lr=lr, randT=randT, steps=steps, C=C, nonfinite0=nonfinite0, diverged=diverged,
               seconds=round(time.time() - t0, 1), nonfinite_grad_count=nf_total, nonfinite_first_step=nf_first,
               nonfinite_max_step_frac=nf_maxfrac, code_sha256=CODE_SHA, versions=VERSIONS,
               eps=(rdp_epsilon(cfg["q"], cfg["sigma_dp"], steps, cfg["delta"]) if arm == "dp" else None),
               cfg={k_: cfg[k_] for k_ in ("n_train", "q", "bmax", "sigma_dp", "delta")})
    if diverged:
        res.update({f"{a}_{b}": float("inf") for a in ("mean", "med") for b in ("T", "2T")})
        res.update(fcat_T=1.0, fcat_2T=1.0, val_med=float("inf"), f_unstable=1.0, f_unstable_exact=1.0)
    else:
        res.update(evaluate(arch, params, tok_te, y_te, T))
        res["val_med"] = float(jnp.median((vfun(params, tok_va) - y_va) ** 2))
        res.update(diagnostics(arch, params, tok_dg, y_dg, T, psg, C, tok_te))
    return res


# ----------------------------------------------------------------------------- entry points
def cells():
    out = [(a, T, arm, False) for a in ARCHS for T in TS for arm in ARMS]
    return out + [("soft_norm", 16, "clip", True), ("soft_norm", 16, "dp", True)]


def stage0():
    print("G-acc  accountant: q=256/60000, sigma=1.1, 60 epochs, delta=1e-5 ->",
          round(rdp_epsilon(256 / 60000, 1.1, 60 * 60000 // 256, 1e-5), 3), "(commonly quoted value is about 3.0)")
    print("       pilot schedule: q=%.5f, sigma=%.1f, steps=%d, delta=%g -> eps = %.2f" % (
        CFG["q"], CFG["sigma_dp"], CFG["steps"], CFG["delta"], rdp_epsilon(CFG["q"], CFG["sigma_dp"], CFG["steps"], CFG["delta"])))
    # G-grad: per-sample gradient against finite differences (lin block, float32)
    tok, y = make_data(99, 4, False); params = init_params("lin_raw", 99)
    _, psg = build("lin_raw", 4, "clip", False, 1.0, 0.0, 1e-3, 1.0)
    g = psg(params, tok, y, 4)
    f = lambda A: float((predict("lin_raw", dict(A=A), tok[0], 4, 4) - y[0]) ** 2)
    E = np.zeros((5, 5), np.float32); E[1, 2] = 1e-3
    fd = (f(params["A"] + E) - f(params["A"] - E)) / 2e-3
    print(f"G-grad per-sample gradient vs finite difference: {float(g['A'][0, 1, 2]):.5f} vs {fd:.5f}")
    # G-time and G-learn: non-private arm only, seed 100 (not an analysis seed)
    for arch in ARCHS:
        for T in TS:
            r = run(arch, T, "np", 100, 3e-3, steps=300, verbose=False)
            print(f"G-learn {arch:>9} T={T:2d} np 300 steps: test median L/L0 = {r['med_T']:.3f}, diverged={r['diverged']}, "
                  f"{r['seconds'] / 300 * 1000:.0f} ms/step, C(init median grad norm) = {r['C']:.3g}, non-finite at init {r['nonfinite0']:.3f}", flush=True)


def grid(part, out, budget=1e9):
    k, m = (int(s) for s in part.split("/")); t_start = time.time()
    jobs = []
    for (a, T, arm, rT) in cells():
        for lr in CFG["lrs"]:
            jobs.append((a, T, arm, rT, CFG["sel_seed"], lr))
    done = set()
    if os.path.exists(out):
        for line in open(out):
            r = json.loads(line); done.add((r["arch"], r["T"], r["arm"], r["randT"], r["seed"], r["lr"]))
    # evaluation seeds use the lr selected on the selection seed; they are scheduled once selection results exist
    sel = {}
    if os.path.exists(out):
        for line in open(out):
            r = json.loads(line)
            if r["seed"] == CFG["sel_seed"]:
                c = (r["arch"], r["T"], r["arm"], r["randT"])
                if c not in sel or r["val_med"] < sel[c][0]:
                    sel[c] = (r["val_med"], r["lr"])
    n_sel = {c: sum(1 for d in done if d[:4] == c and d[4] == CFG["sel_seed"]) for c in sel}
    for c, (_, lr) in sel.items():
        if n_sel[c] == len(CFG["lrs"]):
            jobs += [(*c, s, lr) for s in CFG["eval_seeds"]]
    jobs = [j for i, j in enumerate(jobs) if i % m == k and j not in done]
    jobs.sort(key=lambda j: (j[0] == "soft_norm", j[1]))            # cheap cells first
    print(f"pending jobs in this pass: {len(jobs)}", flush=True)
    for j in jobs:
        if time.time() - t_start > budget:
            print("budget reached; rerun to continue"); return
        r = run(j[0], j[1], j[2], j[4], j[5], randT=j[3])
        with open(out, "a") as fh:
            fh.write(json.dumps(r) + "\n")
        print(json.dumps({k_: r[k_] for k_ in ("arch", "T", "arm", "randT", "seed", "lr", "med_T", "fcat_T", "seconds")}), flush=True)


def summarize(out):
    """Preregistered analysis (PREREG_PILOT.md). Paired by seed over the evaluation seeds; 95% t-intervals."""
    rows = [json.loads(l) for l in open(out)]
    ev = [r for r in rows if r["seed"] in CFG["eval_seeds"]]
    def cell(a, T, arm, rT=False):
        return {r["seed"]: r for r in ev if (r["arch"], r["T"], r["arm"], r["randT"]) == (a, T, arm, rT)}
    def ci(d):
        d = np.asarray(d, float)
        if len(d) < 2 or not np.all(np.isfinite(d)):
            return (float(d.mean()) if len(d) else float("nan"), float("nan"), float("nan"), len(d))
        tcrit = {2: 12.71, 3: 4.303, 4: 3.182, 5: 2.776}[len(d)]
        h = tcrit * d.std(ddof=1) / math.sqrt(len(d))
        return d.mean(), d.mean() - h, d.mean() + h, len(d)
    def diff(x, y, key):
        return [x[i][key] - y[i][key] for i in sorted(set(x) & set(y))]
    def lograt(x, y, key):
        return [math.log(x[i][key] / y[i][key]) if x[i][key] > 0 and y[i][key] > 0 and math.isfinite(x[i][key] / y[i][key])
                else float("nan") for i in sorted(set(x) & set(y))]
    pp = lambda c: f"{100 * c[0]:+.2f} [{100 * c[1]:+.2f}, {100 * c[2]:+.2f}] pp (n={c[3]})"
    gm = lambda c: f"{math.exp(c[0]):.3f} [{math.exp(c[1]):.3f}, {math.exp(c[2]):.3f}] (n={c[3]})"
    C = {(a, T, arm): cell(a, T, arm) for a in ARCHS for T in TS for arm in ARMS}

    print("V1 baseline validity: np median L/L0 at T per evaluation seed (must be <= 0.3), diverged count")
    for a in ARCHS:
        for T in TS:
            c = C[(a, T, "np")]
            print(f"  {a:>9} T={T:2d}: " + ", ".join(f"{c[s_]['med_T']:.3f}" for s_ in sorted(c)) + f"; diverged {sum(r['diverged'] for r in c.values())}/{len(c)}")

    print("\nE1 stability erosion, lin_raw: f_unstable_exact(clip) - f_unstable_exact(np)")
    e1 = {}
    for a in ("lin_raw", "lin_norm"):
        for T in TS:
            e1[(a, T)] = diff(C[(a, T, "clip")], C[(a, T, "np")], "f_unstable_exact")
            print(f"  {a:>9} T={T:2d}: {pp(ci(e1[(a, T)]))};  fcat@2T clip-np: {pp(ci(diff(C[(a, T, 'clip')], C[(a, T, 'np')], 'fcat_2T')))}")
    print("\nE3 token normalization: [lin_raw erosion] - [lin_norm erosion] at T=16")
    if len(e1[("lin_raw", 16)]) == len(e1[("lin_norm", 16)]):
        print("  " + pp(ci(np.array(e1[("lin_raw", 16)]) - np.array(e1[("lin_norm", 16)]))))

    print("\nE2 softmax block: clip vs np in-horizon")
    for T in TS:
        cl, npc = C[("soft_norm", T, "clip")], C[("soft_norm", T, "np")]
        print(f"  soft_norm T={T:2d}: median-loss ratio clip/np {gm(ci(lograt(cl, npc, 'med_T')))};  fcat@T clip-np {pp(ci(diff(cl, npc, 'fcat_T')))}")

    print("\nE4 extrapolation drift: D = median loss dp / median loss clip; ratio D_2T / D_T")
    def e4(dp, cl):
        s_ = sorted(set(dp) & set(cl))
        return [math.log(dp[i]["med_2T"] / cl[i]["med_2T"]) - math.log(dp[i]["med_T"] / cl[i]["med_T"]) for i in s_]
    for a in ARCHS:
        for T in TS:
            dp, cl = C[(a, T, "dp")], C[(a, T, "clip")]
            if dp and cl:
                print(f"  {a:>9} T={T:2d}: D_T {gm(ci(lograt(dp, cl, 'med_T')))};  D_2T/D_T {gm(ci(e4(dp, cl)))}")
    dp, cl = cell("soft_norm", 16, "dp", True), cell("soft_norm", 16, "clip", True)
    if dp and cl:
        print(f"  soft_norm T=16 random-T: D_T {gm(ci(lograt(dp, cl, 'med_T')))};  D_2T/D_T {gm(ci(e4(dp, cl)))}")

    print("\nDescriptive (means over evaluation seeds): lr, median L/L0 at T and 2T, mean L/L0 at T, fcat at T and 2T, "
          "f_unstable (jvp), clipped fraction at end, q99/median gradient norm, spearman(gradient norm, rho)")
    for (a, T, arm, rT) in cells():
        c = cell(a, T, arm, rT)
        if c:
            f = lambda k_: np.mean([r.get(k_, float("nan")) for r in c.values()])
            print(f"  {a:>9} T={T:2d} {arm:>4}{' rT' if rT else '   '}: lr {list(c.values())[0]['lr']:g}, med {f('med_T'):.4f} / {f('med_2T'):.4g}, "
                  f"mean {f('mean_T'):.4g}, fcat {100 * f('fcat_T'):.2f}% / {100 * f('fcat_2T'):.2f}%, f_unst {100 * f('f_unstable'):.2f}%, "
                  f"clipfrac {f('clip_frac_end'):.3f}, het {f('het_q99'):.1f}, spearman {f('spearman_g_rho'):+.2f}, diverged {sum(r['diverged'] for r in c.values())}/{len(c)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd")
    ap.add_argument("--arch"); ap.add_argument("--T", type=int); ap.add_argument("--arm"); ap.add_argument("--seed", type=int)
    ap.add_argument("--lr", type=float); ap.add_argument("--randT", action="store_true")
    ap.add_argument("--part", default="0/1"); ap.add_argument("--out", default="results.jsonl")
    ap.add_argument("--budget", type=float, default=1e9, help="seconds; stop starting new runs after this (resume by rerunning)")
    a = ap.parse_args()
    if a.cmd == "stage0": stage0()
    elif a.cmd == "run": print(json.dumps(run(a.arch, a.T, a.arm, a.seed, a.lr, randT=a.randT, verbose=True)))
    elif a.cmd == "grid": grid(a.part, a.out, a.budget)
    elif a.cmd == "summarize": summarize(a.out)
