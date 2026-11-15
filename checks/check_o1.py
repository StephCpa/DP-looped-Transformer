"""O1 check for claude/dp-looped-theory.md (v0.2): looped in-context gradient descent under DP.

Model. Prompt i: X_i (N x p, rows N(0,I)), w*_i ~ N(0, I/p), y = X w*, query x_q ~ N(0,I).
Loop (one shared scalar gamma): w_{t+1} = w_t - gamma (S_i w_t - S_i w*_i),  S_i = X^T X / N  [raw]
                                 or with S_i replaced by S_i / tr(S_i), gamma = 2*sigmoid(theta)  [normalized]
Residual after T steps: r_i = sum_j a_ij b_ij (1 - gamma lam_ij)^T, loss 0.5 r_i^2.
Prompt i is unstable iff gamma * lam_max,i > 2.
Run: python3 check_o1.py
"""
import numpy as np

rng = np.random.default_rng(1)


def make(n, p=5, N=10):
    X = rng.normal(size=(n, N, p))
    S = np.einsum('nki,nkj->nij', X, X) / N
    lam, V = np.linalg.eigh(S)
    w = rng.normal(size=(n, p)) / np.sqrt(p); q = rng.normal(size=(n, p))
    a = np.einsum('nij,ni->nj', V, q); b = np.einsum('nij,ni->nj', V, w)
    return lam, a * b


def resid(gam, lam, ab, T):
    return np.sum(ab * (1 - gam * lam) ** T, axis=1)


def loss(gam, lam, ab, T):
    with np.errstate(over='ignore'):
        return 0.5 * np.mean(resid(gam, lam, ab, T) ** 2)


def grads(gam, lam, ab, T):
    """per-prompt d loss / d gamma"""
    with np.errstate(over='ignore', invalid='ignore'):
        r = resid(gam, lam, ab, T)
        dr = -T * np.sum(ab * lam * (1 - gam * lam) ** (T - 1), axis=1)
        g = r * dr
    return np.nan_to_num(g, nan=0.0, posinf=1e300, neginf=-1e300)


def clipped_mean(g, C):
    if np.isinf(C):
        return np.mean(g)
    return np.mean(g * np.minimum(1.0, C / (np.abs(g) + 1e-300)))


def profile():
    print("== Prop 9: sup over u in (0,1) of T u (1-u)^(2T-1)  (contracting-side gradient, units c^2/gamma) ==")
    u = np.linspace(1e-6, 1 - 1e-6, 2_000_001)
    for T in (1, 2, 4, 16, 64, 256):
        print(f"T={T:4d}  sup={np.max(T * u * (1 - u) ** (2 * T - 1)):.4f}   at u=2.2: {T * 2.2 * 1.2 ** (2 * T - 1):.3e}")


def fixed_points(n=4000):
    print("\n== Theorem 2: clipping-only fixed point of the raw loop (no noise), n=%d train / test ==" % n)
    lam, ab = make(n); lam_t, ab_t = make(n)
    lmax, lmax_t = lam.max(axis=1), lam_t.max(axis=1)
    print(f"lam_max over prompts: median {np.median(lmax):.2f}, 99% {np.quantile(lmax, .99):.2f}, max {lmax.max():.2f}")
    grid = np.linspace(0.005, 3.0, 1500)
    print(f"{'T':>3} {'C/med|g|':>9} {'gamma':>7} {'gam*lmax_max':>12} {'unstable train':>14} {'unstable test':>13} "
          f"{'L/L0 @T':>9} {'L/L0 @2T':>10} {'max|g|/C':>9}")
    for T in (1, 4, 16, 64):
        med = np.median(np.abs(grads(0.05, lam, ab, T)))
        L0 = loss(0.0, lam_t, ab_t, T)
        for mult in (np.inf, 100, 10, 1, 0.1):
            C = mult * med
            G = np.array([clipped_mean(grads(g_, lam, ab, T), C) for g_ in grid])
            idx = np.where((G[:-1] < 0) & (G[1:] >= 0))[0]
            if len(idx) == 0:
                print(f"{T:3d} {mult:9.1f}   no sign change on grid"); continue
            lo, hi = grid[idx[0]], grid[idx[0] + 1]
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                if clipped_mean(grads(mid, lam, ab, T), C) < 0: lo = mid
                else: hi = mid
            gam = 0.5 * (lo + hi)
            gmax = np.max(np.abs(grads(gam, lam, ab, T))) / C if np.isfinite(C) else float('nan')
            print(f"{T:3d} {mult:9.1f} {gam:7.3f} {gam * lmax.max():12.3f} {np.mean(gam * lmax > 2):14.4f} "
                  f"{np.mean(gam * lmax_t > 2):13.4f} {loss(gam, lam_t, ab_t, T) / L0:9.3e} "
                  f"{loss(gam, lam_t, ab_t, 2 * T) / loss(0.0, lam_t, ab_t, 2 * T):10.3e} {gmax:9.2e}")


def root(lam, ab, T, C, grid):
    G = np.array([clipped_mean(grads(g_, lam, ab, T), C) for g_ in grid])
    idx = np.where((G[:-1] < 0) & (G[1:] >= 0))[0]
    if len(idx) == 0:
        return grid[-1]  # gradient negative on the whole grid: optimum at the upper boundary
    lo, hi = grid[idx[0]], grid[idx[0] + 1]
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if clipped_mean(grads(mid, lam, ab, T), C) < 0: lo = mid
        else: hi = mid
    return 0.5 * (lo + hi)


def ratios(n=4000, n_big=40000):
    print("\n== Loss ratios at the stationary point, raw vs normalized block (test set of %d prompts) ==" % n_big)
    print("R_erm = L_test(unclipped ERM) / L_test(population-optimal gamma);  R_C = L_test(clipped fixed point) / L_test(unclipped ERM)")
    lam, ab = make(n); lam_b, ab_b = make(n_big)
    print(f"{'block':>11} {'T':>3} {'gam_pop':>8} {'gam_erm':>8} {'R_erm':>10} {'L_erm/L0':>9} {'R_C: C=10med':>13} {'C=med':>10} {'C=0.1med':>10} {'unstable test @C=med':>21}")
    for block in ('raw', 'normalized'):
        if block == 'raw':
            lm, lb, grid = lam, lam_b, np.linspace(0.005, 3.0, 1000)
        else:
            lm, lb = lam / lam.sum(1, keepdims=True), lam_b / lam_b.sum(1, keepdims=True)
            grid = np.linspace(0.005, 1.999, 1000)
        for T in (1, 4, 16, 64):
            cg = grid[::4]; Lb = np.array([loss(g_, lb, ab_b, T) for g_ in cg]); gpop = cg[np.argmin(Lb)]
            med = np.median(np.abs(grads(grid[0] * 10, lm, ab, T)))
            germ = root(lm, ab, T, np.inf, grid)
            Lerm = loss(germ, lb, ab_b, T)
            out = []
            for mult in (10, 1, 0.1):
                gc = root(lm, ab, T, mult * med, grid)
                out.append((loss(gc, lb, ab_b, T) / Lerm, np.mean(gc * lb.max(1) > 2)))
            print(f"{block:>11} {T:3d} {gpop:8.3f} {germ:8.3f} {Lerm / Lb.min():10.3e} {Lerm / loss(0.0, lb, ab_b, T):9.2e} {out[0][0]:13.3e} {out[1][0]:10.3e} {out[2][0]:10.3e} {out[1][1]:21.4f}")


def dynamics(n=4000, T=16, mu=1.0, K=6000, seeds=5):
    print("\n== Dynamics: clipped noisy GD on the raw loop, T=%d, C = median |g|, mu-GDP=%.1f over K=%d steps, n=%d, %d seeds ==" % (T, mu, K, n, seeds))
    lam, ab = make(n); lam_t, ab_t = make(20000)
    med = np.median(np.abs(grads(0.05, lam, ab, T))); C = med
    grid = np.linspace(0.005, 3.0, 3000)
    print(f"clipped fixed point (root): gamma = {root(lam, ab, T, C, grid):.3f};  unclipped ERM: gamma = {root(lam, ab, T, np.inf, grid):.3f}")
    s = C * np.sqrt(K) / (n * mu); eta = 0.05 / C
    marks = (100, 300, 1000, 3000, 6000); rec = {k: [] for k in marks}
    L0 = loss(0.0, lam_t, ab_t, T)
    for sd in range(seeds):
        r2 = np.random.default_rng(500 + sd); gam = 0.05
        for k in range(1, K + 1):
            gam -= eta * (clipped_mean(grads(gam, lam, ab, T), C) + s * r2.normal())
            if k in rec:
                rec[k].append((gam, loss(gam, lam_t, ab_t, T) / L0, np.mean(gam * lam_t.max(1) > 2)))
    print(f"{'step':>6} {'gamma mean':>11} {'gamma sd':>9} {'median L/L0':>12} {'unstable test':>13}")
    for k in marks:
        a = np.array(rec[k])
        print(f"{k:6d} {a[:, 0].mean():11.3f} {a[:, 0].std():9.4f} {np.median(a[:, 1]):12.3e} {a[:, 2].mean():13.4f}")


if __name__ == "__main__":
    profile(); fixed_points(); ratios(); dynamics()
