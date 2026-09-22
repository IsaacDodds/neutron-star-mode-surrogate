# Does predicting the ladder beat predicting 182 slots?
#
# Same stars, same split, same architecture, same metric as the seats: the median of
# |f_hat - f|/f over the slots a star fills. The only change is the target. Instead of
# 182 frequencies the network predicts twelve numbers: the two ladder parameters for g
# and for s, the interface mode and the fundamental, and four overtone slots, each with
# its presence. The spectrum is then rebuilt from them and scored rung by rung.
# Usage: python ladder_seat.py DIR [DIR ...]   (each DIR holds sample_*.npz)
import glob, sys, os, importlib.util as iu
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", False)

VPATH = os.environ.get("FM3_VARIANTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "variants.py"))
sp = iu.spec_from_file_location("v", VPATH)
v = iu.module_from_spec(sp); sp.loader.exec_module(v)
DIRS = sys.argv[1:]
GCUT, SCUT = 4, 3          # the first rung of each family that is on its asymptotic line


def ladder_fit(k, f, period):
    y = 1.0 / f if period else f
    A = np.polyfit(k, y, 1)
    return A[0], A[1]


def build():
    X, T, MASK, TRUE = [], [], [], []
    seen = set()
    for dd in DIRS:
        for fn in sorted(glob.glob(dd + "/sample_*.npz")):
            try: d = np.load(fn, allow_pickle=True)
            except Exception: continue
            key = (int(d["index"]), int(d["seed"]))
            if key in seen: continue
            seen.add(key)
            names = [str(x) for x in d["mode_names"]]; fr = np.asarray(d["mode_row_freqs"], float)
            g = sorted((int(n[1:]), f) for n, f in zip(names, fr) if n[:1] == "g" and n[1:].isdigit())
            s = sorted((int(n[1:]), f) for n, f in zip(names, fr) if n[:1] == "s" and n[1:].isdigit())
            gk = np.array([a for a, _ in g], float); gf = np.array([b for _, b in g], float)
            sk = np.array([a for a, _ in s], float); sf = np.array([b for _, b in s], float)
            gm = gk >= GCUT; sm = sk >= SCUT
            if gm.sum() < 4 or sm.sum() < 3: continue
            gsl, gic = ladder_fit(gk[gm], gf[gm], True)
            ssl, sic = ladder_fit(sk[sm], sf[sm], False)
            if gsl <= 0 or ssl <= 0: continue
            fam = np.zeros(5); fam[int(d["family"]) - 1] = 1
            X.append(np.concatenate([np.asarray(d["theta"], float), fam,
                                     [float(d["nt1"]), float(d["mass_target"])]]))
            # twelve numbers: g anchor at GCUT and g spacing, s anchor at SCUT and s spacing,
            # then i and f as frequencies, all in the log so an error is fractional
            gA = 1.0 / (gsl * GCUT + gic); sA = ssl * SCUT + sic
            fi = fr[names.index("i")] if "i" in names else np.nan
            ff = fr[names.index("f")] if "f" in names else np.nan
            T.append([np.log(gA), np.log(gsl), np.log(sA), np.log(ssl),
                      np.log(fi) if np.isfinite(fi) else 0.0,
                      np.log(ff) if np.isfinite(ff) else 0.0])
            MASK.append([1, 1, 1, 1, float(np.isfinite(fi)), float(np.isfinite(ff))])
            TRUE.append((gk[gm], gf[gm], sk[sm], sf[sm], fi, ff))
    return np.stack(X), np.stack(T), np.stack(MASK), TRUE


def train(X, T, M, seed=0, hidden=(256,)*4, epochs=3000, drop=0.2, lr=3e-4):
    n = len(X); o = np.random.default_rng(0).permutation(n)
    nt = max(30, int(n * 0.15)); te, va, tr = o[:nt], o[nt:2*nt], o[2*nt:]
    xm, xs = X[tr].mean(0), X[tr].std(0) + 1e-8
    den = np.maximum(M[tr].sum(0), 1.0)
    tm = (T[tr]*M[tr]).sum(0)/den
    ts = np.maximum(np.sqrt(((T[tr]-tm)**2*M[tr]).sum(0)/den), 0.05)
    Xs = (X-xm)/xs; Tn = np.where(M > 0, (T-tm)/ts, 0.0)
    key = jax.random.key(seed); P = []
    sizes = [X.shape[1]] + list(hidden)
    for a, b in zip(sizes[:-1], sizes[1:]):
        key, k = jax.random.split(key)
        P.append([jax.random.normal(k, (a, b))*np.sqrt(2/(a+b)), jnp.zeros(b)])
    key, k = jax.random.split(key)
    P.append([jax.random.normal(k, (hidden[-1], T.shape[1]))*np.sqrt(1/hidden[-1]), jnp.zeros(T.shape[1])])

    def fwd(P, x, kk=None, train=False):
        h = x
        for W, b in P[:-1]:
            h = jax.nn.silu(h @ W + b)
            if train and drop > 0:
                kk, k2 = jax.random.split(kk)
                h = h*jax.random.bernoulli(k2, 1-drop, h.shape)/(1-drop)
        W, b = P[-1]; return h @ W + b

    def loss(P, Xa, Ta, Ma, kk):
        pr = jax.vmap(lambda x, k3: fwd(P, x, k3, True))(Xa, jax.random.split(kk, len(Xa)))
        return jnp.sum(Ma*(pr-Ta)**2)/jnp.maximum(jnp.sum(Ma), 1.0)

    flat, tree = jax.tree_util.tree_flatten(P)
    m = [jnp.zeros_like(a) for a in flat]; vv = [jnp.zeros_like(a) for a in flat]

    @jax.jit
    def step(P, m, vv, t, Xa, Ta, Ma, kk):
        l, gr = jax.value_and_grad(loss)(P, Xa, Ta, Ma, kk)
        pf, tr_ = jax.tree_util.tree_flatten(P); gf, _ = jax.tree_util.tree_flatten(gr)
        nm = [0.9*a+0.1*b for a, b in zip(m, gf)]; nv = [0.999*a+0.001*b*b for a, b in zip(vv, gf)]
        new = [p-lr*((a/(1-0.9**t))/(jnp.sqrt(b/(1-0.999**t))+1e-8)+1e-5*p) for p, a, b in zip(pf, nm, nv)]
        return jax.tree_util.tree_unflatten(tr_, new), nm, nv, l

    A = (jnp.asarray(Xs[tr]), jnp.asarray(Tn[tr]), jnp.asarray(M[tr]))
    best, bestP = np.inf, P
    for ep in range(1, epochs+1):
        key, k = jax.random.split(key)
        P, m, vv, l = step(P, m, vv, ep, *A, k)
        if ep % 100 == 0:
            pv = np.asarray(jax.vmap(lambda x: fwd(P, x))(jnp.asarray(Xs[va])))
            e = float(np.sum(M[va]*(pv-Tn[va])**2)/max(M[va].sum(), 1))
            if e < best: best, bestP = e, P
    pred = np.asarray(jax.vmap(lambda x: fwd(bestP, x))(jnp.asarray(Xs[te])))*ts+tm
    return te, pred


if __name__ == "__main__":
    X, T, M, TRUE = build()
    print(f"{len(X)} stars with both ladders fittable", flush=True)
    errs_all, errs_g, errs_s, errs_i = [], [], [], []
    for seed in range(3):
        te, pred = train(X, T, M, seed)
        for j, idx in enumerate(te):
            gk, gf, sk, sf, fi, ff = TRUE[idx]
            gA, gsl = np.exp(pred[j, 0]), np.exp(pred[j, 1])
            sA, ssl = np.exp(pred[j, 2]), np.exp(pred[j, 3])
            gh = 1.0/(1.0/gA + (gk-GCUT)*gsl)
            sh = sA + (sk-SCUT)*ssl
            e = list(np.abs(gh-gf)/gf) + list(np.abs(sh-sf)/sf)
            errs_g += list(np.abs(gh-gf)/gf); errs_s += list(np.abs(sh-sf)/sf)
            if np.isfinite(fi):
                ei = abs(np.exp(pred[j, 4])-fi)/fi; e.append(ei); errs_i.append(ei)
            if np.isfinite(ff): e.append(abs(np.exp(pred[j, 5])-ff)/ff)
            errs_all.append(np.median(e))
        print(f"  seed {seed}: per-star median {100*np.median(errs_all[-len(te):]):.2f}%", flush=True)
    print(f"\nLADDER TARGET, {len(errs_all)} held-out stars over 3 seeds")
    print(f"  all modes   median {100*np.median(errs_all):.2f}%")
    print(f"  g rungs     median {100*np.median(errs_g):.2f}%   ({len(errs_g)} rungs)")
    print(f"  s rungs     median {100*np.median(errs_s):.2f}%   ({len(errs_s)} rungs)")
    print(f"  i mode      median {100*np.median(errs_i):.2f}%   ({len(errs_i)} stars)")
    print(f"  compare: the 182-slot one-shot seat is 6.88% all modes, 30.5% on i")
