"""Checks for docs/dp-looped-theory.md v0.3, section 16.

Part A. Softmax-attention loop (retrieval toy).
  Prompt: M stored tokens x_1..x_M in R^p; query q = noisy copy of x_1; target x_1.
  Loop with input injection: z_0 = 0,  z_{t+1} = (1-alpha) z_t + alpha * sum_j softmax_j(beta x_j.(q + z_t)) x_j.
  Without injection (inject=False): z_0 = q, logits beta x_j.z_t.
  One trainable scalar beta. normalized: ||x_j|| = ||q|| = 1.  raw: per-prompt log-normal token scale.
Part B. Matrix preconditioner for the linear loop of check_o1.py, clipping on the Frobenius norm.
Each function reseeds, so functions can be run separately and give the same numbers.
Run: python3 check_softmax.py   (about 7 minutes)
"""
import numpy as np

rng = np.random.default_rng(2)


def seed(k):
    global rng
    rng = np.random.default_rng(k)


# ---------------------------------------------------------------- Part A
def make_tokens(n, M=8, p=8, nu=1.0, normalized=True):
    X = rng.normal(size=(n, M, p))
    X /= np.linalg.norm(X, axis=2, keepdims=True)
    if not normalized:
        X *= np.exp(0.5 * rng.normal(size=(n, 1, 1)))
    tgt = X[:, 0, :].copy()
    scale = np.linalg.norm(tgt, axis=1, keepdims=True)
    q = tgt + nu * scale * rng.normal(size=(n, p)) / np.sqrt(p)
    q *= scale / np.linalg.norm(q, axis=1, keepdims=True)
    return X, tgt, q


def run(beta, X, tgt, q, T, alpha=0.5, inject=True, jac=False):
    """per-prompt loss, d loss / d beta, max Jacobian norm along the trajectory, final state, final Jacobian norm"""
    p = X.shape[2]
    z = np.zeros_like(q) if inject else q.copy()
    dz = np.zeros_like(z); jmax = 0.0; jlast = None
    for t in range(T + (1 if jac else 0)):
        u = q + z if inject else z
        s = np.einsum('nmp,np->nm', X, u); ds = np.einsum('nmp,np->nm', X, dz)
        a = beta * s; a -= a.max(1, keepdims=True); pr = np.exp(a); pr /= pr.sum(1, keepdims=True)
        if jac:
            mean = np.einsum('nm,nmp->np', pr, X)
            cov = np.einsum('nm,nmp,nmq->npq', pr, X, X) - np.einsum('np,nq->npq', mean, mean)
            jlast = np.linalg.norm((1 - alpha) * np.eye(p)[None] + alpha * beta * cov, 2, axis=(1, 2))
            jmax = max(jmax, jlast.max())
            if t == T:
                break
        dlog = s + beta * ds
        dpr = pr * (dlog - np.sum(pr * dlog, 1, keepdims=True))
        z, dz = ((1 - alpha) * z + alpha * np.einsum('nm,nmp->np', pr, X),
                 (1 - alpha) * dz + alpha * np.einsum('nm,nmp->np', dpr, X))
    r = z - tgt
    return 0.5 * np.sum(r * r, 1), np.sum(r * dz, 1), jmax, z, jlast


def clipped_mean(g, C):
    if np.isinf(C):
        return np.mean(g)
    return np.mean(g * np.minimum(1.0, C / (np.abs(g) + 1e-300)))


def root(f, grid):
    """first stationary point reached when ascending from grid[0] (gradient descent from a small initial value)"""
    G = np.array([f(b) for b in grid])
    idx = np.where((G[:-1] < 0) & (G[1:] >= 0))[0]
    if len(idx) == 0:
        return grid[-1] if G[-1] < 0 else grid[0]
    lo, hi = grid[idx[0]], grid[idx[0] + 1]
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0: lo = mid
        else: hi = mid
    return 0.5 * (lo + hi)


def prop10(alpha=0.5):
    seed(10)
    print("== Prop 10 (normalized tokens, R=1, with injection): observed / bound, asserted <= 1 ==")
    print("Jacobian bound: (1-alpha) + alpha*beta.   Gradient bound: 4 * alpha * S_T(rho), rho = 1 - alpha(1-beta), beta < 1")
    X, tgt, q = make_tokens(2000)
    print(f"{'beta':>5} {'T':>4} {'max||J||/bound':>15} {'max|g|/bound':>13} {'max|g|':>9} {'max/med |g|':>12}")
    for beta in (0.3, 0.6, 0.9, 0.99):
        for T in (4, 16, 64, 256):
            _, g, jm, _, _ = run(beta, X, tgt, q, T, alpha, jac=True)
            rho = 1 - alpha * (1 - beta); ST = (1 - rho ** T) / (1 - rho)
            assert jm <= ((1 - alpha) + alpha * beta) * (1 + 1e-6) and np.abs(g).max() <= 4 * alpha * ST * (1 + 1e-6), f"Prop 10 bound violated at beta={beta}, T={T}"
            print(f"{beta:5.2f} {T:4d} {jm / ((1 - alpha) + alpha * beta):15.3f} {np.abs(g).max() / (4 * alpha * ST):13.4f} "
                  f"{np.abs(g).max():9.4f} {np.abs(g).max() / np.median(np.abs(g)):12.1f}")
    print("Beyond the contraction regime (beta > 1), T=64: Jacobian bound holds, state stays in the unit ball, gradients are not bounded a priori")
    for beta in (2.0, 5.0, 20.0):
        l, g, jm, _, _ = run(beta, X, tgt, q, 64, alpha, jac=True)
        print(f"beta={beta:5.1f}  max||J||={jm:7.3f} (bound {(1 - alpha) + alpha * beta:6.2f})  max loss={l.max():.3f} (cap 2.0)  "
              f"max|g|={np.abs(g).max():9.3f}  max/med |g|={np.abs(g).max() / np.median(np.abs(g)):9.1f}")


def forgetting(alpha=0.5):
    seed(11)
    print("\n== Forgetting without injection (Prop 2 / Prop 8): same tokens, two different queries, beta=0.5 ==")
    X, tgt, q = make_tokens(500); q2 = np.roll(q, 1, axis=0)
    for T in (4, 16, 64):
        d_no = np.abs(run(0.5, X, tgt, q, T, alpha, inject=False)[3] - run(0.5, X, tgt, q2, T, alpha, inject=False)[3]).max()
        d_in = np.abs(run(0.5, X, tgt, q, T, alpha, inject=True)[3] - run(0.5, X, tgt, q2, T, alpha, inject=True)[3]).max()
        assert T < 64 or d_no < 1e-8, "forgetting check failed: a contractive loop without injection should forget its query"
        print(f"T={T:3d}  max |z_T(q) - z_T(q')|: no injection {d_no:.2e}   with injection {d_in:.2e}")


def softmax_clip(n=3000, n_test=15000, alpha=0.5):
    seed(12)
    print("\n== Clipping fixed point of the normalized softmax loop vs T (C in units of the median |g| at beta=1) ==")
    print("R_C = L_test(clipped fixed point) / L_test(unclipped ERM);  het = max|g| / median|g| at the unclipped ERM")
    print(f"{'T':>3} {'beta_erm':>8} {'L_erm/L0':>9} {'het':>9} {'R_C: C=10med':>13} {'C=med':>8} {'C=0.1med':>9} {'beta @C=med':>12} {'beta @0.1med':>13}")
    grid = np.linspace(0.05, 40.0, 120)
    X, tgt, q = make_tokens(n); Xt, tt, qt = make_tokens(n_test)
    for T in (1, 4, 16, 64):
        gfun = lambda b, C: clipped_mean(run(b, X, tgt, q, T, alpha)[1], C)
        med = np.median(np.abs(run(1.0, X, tgt, q, T, alpha)[1]))
        b_erm = root(lambda b: gfun(b, np.inf), grid)
        l_erm = run(b_erm, Xt, tt, qt, T, alpha)[0].mean()
        g_erm = np.abs(run(b_erm, X, tgt, q, T, alpha)[1])
        L0 = run(0.0, Xt, tt, qt, T, alpha)[0].mean()
        out = []
        for mult in (10, 1, 0.1):
            bc = root(lambda b: gfun(b, mult * med), grid)
            out.append((run(bc, Xt, tt, qt, T, alpha)[0].mean() / l_erm, bc))
        print(f"{T:3d} {b_erm:8.2f} {l_erm / L0:9.3f} {g_erm.max() / np.median(g_erm):9.1f} "
              f"{out[0][0]:13.3f} {out[1][0]:8.3f} {out[2][0]:9.3f} {out[1][1]:12.2f} {out[2][1]:13.2f}")


def raw_tokens(alpha=0.5):
    seed(13)
    print("\n== Raw (unnormalized) tokens: a bounded state does not bound sensitivity. beta = 5 ==")
    X, tgt, q = make_tokens(2000, normalized=False)
    print(f"{'T':>4} {'max||J||':>9} {'max|g|':>11} {'max/med |g|':>12} {'max loss':>9}")
    for T in (4, 16, 64):
        l, g, jm, _, _ = run(5.0, X, tgt, q, T, alpha, jac=True)
        print(f"{T:4d} {jm:9.2f} {np.abs(g).max():11.3e} {np.abs(g).max() / np.median(np.abs(g)):12.3e} {l.max():9.2f}")


def critical(n=4000, alpha=0.5, beta=3.0):
    seed(14)
    print("\n== Who carries the gradient at beta=%.1f (normalized tokens)? rho_i = ||J(z_T)|| ==" % beta)
    X, tgt, q = make_tokens(n)
    print(f"{'T':>4} {'frac pushing up':>15} {'max/med |g|':>12} {'median rho':>11} {'spearman(|g|, 1/(1-rho))':>25} {'top-1% share of sum|g|':>23}")
    for T in (4, 16, 64, 256):
        _, g, _, _, rho = run(beta, X, tgt, q, T, alpha, jac=True)
        ra = np.argsort(np.argsort(np.abs(g))); rb = np.argsort(np.argsort(1 / np.maximum(1 - rho, 1e-9)))
        gs = np.sort(np.abs(g))[::-1]
        print(f"{T:4d} {(g < 0).mean():15.3f} {np.abs(g).max() / np.median(np.abs(g)):12.1f} {np.median(rho):11.3f} "
              f"{np.corrcoef(ra, rb)[0, 1]:25.3f} {gs[: n // 100].sum() / gs.sum():23.3f}")


def replicate(draws=5, n=2000, n_test=10000, alpha=0.5):
    seed(15)
    print("\n== Replication over %d independent draws (normalized tokens, n=%d) ==" % (draws, n))
    print(f"{'T':>3} {'draw':>5} {'beta_erm':>9} {'beta @C=med':>12} {'L_erm/L0':>9} {'R_C (C=med)':>12}")
    grid = np.linspace(0.05, 40.0, 120)
    for T in (16, 64):
        rs = []
        for d in range(draws):
            X, tgt, q = make_tokens(n); Xt, tt, qt = make_tokens(n_test)
            med = np.median(np.abs(run(1.0, X, tgt, q, T, alpha)[1]))
            b_erm = root(lambda b: clipped_mean(run(b, X, tgt, q, T, alpha)[1], np.inf), grid)
            b_c = root(lambda b: clipped_mean(run(b, X, tgt, q, T, alpha)[1], med), grid)
            L0 = run(0.0, Xt, tt, qt, T, alpha)[0].mean()
            le = run(b_erm, Xt, tt, qt, T, alpha)[0].mean(); lc = run(b_c, Xt, tt, qt, T, alpha)[0].mean()
            rs.append(lc / le)
            print(f"{T:3d} {d:5d} {b_erm:9.2f} {b_c:12.2f} {le / L0:9.3f} {lc / le:12.3f}")
        print(f"{T:3d} median R_C = {np.median(rs):.3f}, range [{min(rs):.3f}, {max(rs):.3f}]")


def contraction_cost(alpha=0.5):
    seed(17)
    print("\n== Price of a-priori contraction (normalized softmax loop): test L/L0 at beta just inside the contraction regime vs at the trained value ==")
    X, tgt, q = make_tokens(15000)
    print(f"{'T':>4} {'beta=0.99':>10} {'beta=3.3':>9} {'ratio':>7}")
    for T in (4, 16, 64):
        L0 = run(0.0, X, tgt, q, T, alpha)[0].mean()
        a = run(0.99, X, tgt, q, T, alpha)[0].mean() / L0; b = run(3.3, X, tgt, q, T, alpha)[0].mean() / L0
        print(f"{T:4d} {a:10.3f} {b:9.3f} {a / b:7.2f}")


# ---------------------------------------------------------------- Part B
def make_lin(n, p=5, N=10):
    X = rng.normal(size=(n, N, p))
    S = np.einsum('nki,nkj->nij', X, X) / N
    lam, V = np.linalg.eigh(S)
    w = rng.normal(size=(n, p)) / np.sqrt(p); q = rng.normal(size=(n, p))
    return lam, np.einsum('nij,ni->nj', V, q), np.einsum('nij,ni->nj', V, w)


def lin_grads(gam, lam, qt, wt, T):
    """at Gamma = gam*I: residual r, scalar gradient (trace direction), Frobenius norm of the full matrix gradient"""
    with np.errstate(over='ignore', invalid='ignore'):
        a = 1 - gam * lam
        r = np.sum(qt * wt * a ** T, axis=1)
        K = np.zeros((lam.shape[0], lam.shape[1], lam.shape[1]))
        for k in range(T):
            K += a[:, :, None] ** k * a[:, None, :] ** (T - 1 - k)
        D = -qt[:, :, None] * (wt * lam)[:, None, :] * K        # d r / d Gamma in the prompt's eigenbasis
        g_scalar = r * np.einsum('njj->n', D)
        g_frob = np.abs(r) * np.sqrt(np.sum(D * D, axis=(1, 2)))
    return np.nan_to_num(r), np.nan_to_num(g_scalar, posinf=1e300, neginf=-1e300), np.nan_to_num(g_frob, posinf=1e300)


def matrix_clip(n=4000, n_test=40000, draws=3):
    seed(16)
    print("\n== Part B: linear loop, full matrix preconditioner at Gamma = gamma*I; clipping on ||grad||_F; %d draws ==" % draws)
    print("stationarity in the trace direction: mean_i min(1, C/||G_i||_F) tr(G_i) = 0;  C = median norm at gamma=0.05")
    grid = np.linspace(0.005, 3.0, 800)
    print(f"{'T':>3} {'draw':>5} {'gam_erm':>8} {'unstable':>9} | {'scalar clip gam':>15} {'R_C':>10} {'unstable':>9} | {'matrix clip gam':>15} {'R_C':>10} {'unstable':>9}")
    for T in (1, 4, 16):
        for d in range(draws):
            lam, qt, wt = make_lin(n); lam_t, qt_t, wt_t = make_lin(n_test)
            loss = lambda g: 0.5 * np.mean(lin_grads(g, lam_t, qt_t, wt_t, T)[0] ** 2)
            def f(g, C, mode):
                _, gs, gf = lin_grads(g, lam, qt, wt, T)
                nrm = np.abs(gs) if mode == 'scalar' else gf
                w = np.ones_like(gs) if np.isinf(C) else np.minimum(1.0, C / (nrm + 1e-300))
                return np.mean(w * gs)
            _, gs0, gf0 = lin_grads(0.05, lam, qt, wt, T)
            g_erm = root(lambda g: f(g, np.inf, 'scalar'), grid); Lerm = loss(g_erm)
            gsc = root(lambda g: f(g, np.median(np.abs(gs0)), 'scalar'), grid)
            gmx = root(lambda g: f(g, np.median(gf0), 'matrix'), grid)
            un = lambda g: np.mean(g * lam.max(1) > 2)
            print(f"{T:3d} {d:5d} {g_erm:8.3f} {un(g_erm):9.4f} | {gsc:15.3f} {loss(gsc) / Lerm:10.3e} {un(gsc):9.4f} | "
                  f"{gmx:15.3f} {loss(gmx) / Lerm:10.3e} {un(gmx):9.4f}")


if __name__ == "__main__":
    prop10(); forgetting(); raw_tokens(); contraction_cost(); matrix_clip(); critical(); softmax_clip(); replicate()
