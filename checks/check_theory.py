"""Numerical sanity checks for docs/dp-looped-theory.md (DP + looped transformers).

Toy block: F(z, x; theta) = tanh(W z + U x), theta = (W, U), ||W||_2 <= rho.
Checks: Lemma 1 (forward sensitivity), Lemma 2 (gradient norm), Lemma 3 (smoothness),
Prop 3 (coherence factor, tied vs untied), Prop 4 (Cauchy-Schwarz ratio),
Prop 7 (privacy saturation factor), Section 9 toy (equipartition + extrapolation drift).
Run: python3 check_theory.py
"""
import numpy as np

rng = np.random.default_rng(0)
C2 = 4 / (3 * np.sqrt(3))  # max |tanh''|


def S(rho, T, power=1):
    return sum((rho ** power) ** k for k in range(T))


def specnorm(W, rho):
    return W * (rho / np.linalg.norm(W, 2))


def forward(W, U, x, T):
    z = np.zeros(W.shape[0]); zs = [z]
    for _ in range(T):
        z = np.tanh(W @ z + U @ x); zs.append(z)
    return zs


def grad(W, U, x, y, T):
    zs = forward(W, U, x, T)
    lam = zs[-1] - y
    gW = np.zeros_like(W); gU = np.zeros_like(U)
    for t in range(T - 1, -1, -1):
        d = (1 - zs[t + 1] ** 2) * lam
        gW += np.outer(d, zs[t]); gU += np.outer(d, x)
        lam = W.T @ d
    return np.concatenate([gW.ravel(), gU.ravel()]), zs


def lemmas_1_2_3(m=6, p=3, trials=200):
    print("== Lemmas 1-3: max over trials of (observed / bound); asserted <= 1 ==")
    print(f"{'rho':>5} {'T':>5} {'L1 fwd':>9} {'L2 grad':>9} {'L3 smooth':>10} {'|grad| max':>11}")
    for rho in (0.5, 0.9, 0.99):
        for T in (5, 20, 100, 400):
            r1 = r2 = r3 = gmax = 0.0
            for _ in range(trials):
                x = rng.normal(size=p); y = rng.uniform(-1, 1, size=m)
                W = specnorm(rng.normal(size=(m, m)), rho); U = rng.normal(size=(m, p))
                scale = 10 ** rng.uniform(-3, 0)
                W2 = specnorm(W + scale * rng.normal(size=(m, m)), rho)
                U2 = U + scale * rng.normal(size=(m, p))
                dth = np.sqrt(np.sum((W2 - W) ** 2) + np.sum((U2 - U) ** 2))
                Lth = np.sqrt(m + x @ x); G = 2 * np.sqrt(m); H = 1.0
                ST = S(rho, T)
                bzz = C2 * rho ** 2; bzt = C2 * rho * Lth + 1; btz = bzt; btt = C2 * Lth ** 2
                beta = H * Lth**2 * ST**2 + G * (btt * ST + (btz + bzt) * Lth * ST**2 + bzz * Lth**2 * ST**3)
                g1, zs1 = grad(W, U, x, y, T); g2, zs2 = grad(W2, U2, x, y, T)
                r1 = max(r1, np.linalg.norm(zs2[-1] - zs1[-1]) / (Lth * ST * dth))
                r2 = max(r2, np.linalg.norm(g1) / (G * Lth * ST))
                r3 = max(r3, np.linalg.norm(g2 - g1) / (beta * dth))
                gmax = max(gmax, np.linalg.norm(g1))
            assert max(r1, r2, r3) <= 1.0, f"Lemma 1-3 bound violated at rho={rho}, T={T}: {r1}, {r2}, {r3}"
            print(f"{rho:5.2f} {T:5d} {r1:9.4f} {r2:9.4f} {r3:10.5f} {gmax:11.3f}")


def coherence(m=4, p=3):
    print("\n== Prop 3: first-order noise response, tied vs untied ==")
    print(f"{'rho':>5} {'T':>4} {'kappa_T':>8} {'MC/pred tied':>13} {'MC/pred untied':>15} {'grad id err':>12}")
    d = m * m + m * p
    for rho in (0.5, 0.9, 0.99):
        for T in (10, 50):
            x = rng.normal(size=p); y = rng.uniform(-1, 1, size=m)
            W = specnorm(rng.normal(size=(m, m)), rho); U = 0.3 * rng.normal(size=(m, p))
            zs = forward(W, U, x, T)
            A = []
            for t in range(T):
                D = 1 - zs[t + 1] ** 2
                B = np.zeros((m, d))
                for i in range(m):
                    B[i, i * m:(i + 1) * m] = D[i] * zs[t]
                    B[i, m * m + i * p: m * m + (i + 1) * p] = D[i] * x
                Phi = np.eye(m)
                for s_ in range(T - 1, t, -1):
                    Phi = Phi @ (np.diag(1 - zs[s_ + 1] ** 2) @ W)
                A.append(Phi @ B)
            M = sum(A)
            tied_pred = np.sum(M ** 2); untied_pred = sum(np.sum(a ** 2) for a in A)
            g, _ = grad(W, U, x, y, T)
            gid = np.linalg.norm(M.T @ (zs[-1] - y) - g) / (np.linalg.norm(g) + 1e-30)
            s = 1e-5; n = 4000; acc_t = acc_u = 0.0
            for _ in range(n):
                dl = s * rng.normal(size=d)
                z2 = forward(W + dl[:m * m].reshape(m, m), U + dl[m * m:].reshape(m, p), x, T)[-1]
                acc_t += np.sum((z2 - zs[-1]) ** 2)
                z = np.zeros(m)
                for t in range(T):
                    dl = s * rng.normal(size=d)
                    z = np.tanh((W + dl[:m * m].reshape(m, m)) @ z + (U + dl[m * m:].reshape(m, p)) @ x)
                acc_u += np.sum((z - zs[-1]) ** 2)
            assert abs(acc_t / n / s**2 / tied_pred - 1) < 0.1 and abs(acc_u / n / s**2 / untied_pred - 1) < 0.1, "Prop 3: Monte Carlo disagrees with the first-order prediction"
            assert gid < 1e-8, "Prop 3: gradient identity M_T^T v failed"
            print(f"{rho:5.2f} {T:4d} {tied_pred / untied_pred:8.3f} {acc_t / n / s**2 / tied_pred:13.3f} "
                  f"{acc_u / n / s**2 / untied_pred:15.3f} {gid:12.2e}")


def tables():
    print("\n== Prop 4: (sensitivity x sqrt(dim)) tied / untied = S_T / sqrt(T S_T^(2)) ==")
    print("== Prop 7: privacy factor Gamma_T = S_T^2 / S_T^(2) (naive composition gives T) ==")
    print(f"{'rho':>6} {'T':>5} {'tied/untied':>12} {'Gamma_T':>9} {'limit (1+rho)/(1-rho)':>22} {'sched ok':>9}")
    for rho in (0.5, 0.9, 0.99, 1.0):
        for T in (10, 100, 1000):
            s1, s2 = S(rho, T), S(rho, T, 2)
            # optimal shift schedule a_t ∝ rho^(T-t); check feasibility z_t >= 0, z_T = 0 and sum a^2
            a = np.array([s1 / s2 * rho ** (T - t) for t in range(1, T + 1)])
            z = 0.0; ok = True
            for t in range(T):
                z = rho * z + 1.0 - a[t]; ok &= z > -1e-9
            ok &= abs(z) < 1e-6 and abs(np.sum(a ** 2) - s1 ** 2 / s2) < 1e-6 * s1**2 / s2
            assert ok, f"Prop 7: shift schedule infeasible at rho={rho}, T={T}"
            assert s1 / np.sqrt(T * s2) <= 1 + 1e-12 and s1 ** 2 / s2 <= min(T, (1 + rho) / (1 - rho) if rho < 1 else T) * (1 + 1e-9), "Prop 4 / Prop 7 inequality violated"
            lim = (1 + rho) / (1 - rho) if rho < 1 else float('inf')
            print(f"{rho:6.2f} {T:5d} {s1 / np.sqrt(T * s2):12.4f} {s1**2 / s2:9.3f} {lim:22.3f} {str(bool(ok)):>9}")


def toy_valley(T=10, T2=20, c=1.0, eta=0.005, s=0.05, K=4000, R=20000):
    print("\n== Section 9 toy: z_{t+1} = a z_t + b x, loss 0.5 (b S_T(a) - c)^2, noisy GD ==")
    print(f"predicted in-horizon excess loss ~ eta s^2 / 4 = {eta * s * s / 4:.3e}")
    def Sv(a, T_): return sum(a ** k for k in range(T_))
    def dSv(a, T_): return sum(k * a ** (k - 1) for k in range(1, T_))
    print(f"{'a0':>5} {'K':>6} {'excess loss @T':>15} {'var pred @2T':>13} {'mean a - a0':>12} {'sd a':>8}")
    for a0 in (0.3, 0.9):
        a = np.full(R, a0); b = np.full(R, c / Sv(a0, T))
        for k in range(1, K + 1):
            r = b * Sv(a, T) - c
            ga = r * b * dSv(a, T); gb = r * Sv(a, T)
            a = a - eta * (ga + s * rng.normal(size=R)); b = b - eta * (gb + s * rng.normal(size=R))
            if k in (K // 8, K // 4, K // 2, K):
                ex = 0.5 * np.mean((b * Sv(a, T) - c) ** 2)
                print(f"{a0:5.2f} {k:6d} {ex:15.3e} {np.var(b * Sv(a, T2)):13.3e} {np.mean(a) - a0:12.2e} {np.std(a):8.4f} (SE {np.std(a) / np.sqrt(R):.1e})")


if __name__ == "__main__":
    lemmas_1_2_3(); coherence(); tables(); print("\nALL ASSERTIONS PASSED (Lemmas 1-3, Props 3, 4, 7); the toy runs below are statistical and not asserted"); toy_valley()
    toy_valley(s=0.15, K=16000)  # longer, noisier run: drift of mean a along the valley
