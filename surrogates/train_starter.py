# ============================================================
# FM3 NN starter: one-shot surrogate  theta -> mode-slot spectrum
#
# First cut of the NN-placement experiment.  Trains a masked-MSE
# MLP on the fixed ML slot vector (g1..g12, i, s1..s16, f) from the
# fm3_dataset NPZ samples.  Pure JAX; runs in the Hex fm3env or any venv
# with jax+numpy.  Split is by draw index, so no leakage between sets.
#
# Usage:
#   python train_starter.py /path/to/fm3_ds [more dirs ...]
#
# The same loader exposes the intermediate stages, so the other placement
# cuts (theta->EOS, post-TOV->spectrum) reuse build_dataset with a
# different X_of() function.
# ============================================================
import sys, glob, json
import numpy as np
import jax, jax.numpy as jnp

jax.config.update("jax_enable_x64", False)  # training in fp32 is fine

SEED = 0
HIDDEN = (256, 256, 256)
LR = 3e-4
EPOCHS = 3000
VAL_FRAC = 0.15
TEST_FRAC = 0.15


SEAT = "c2"   # set by --seat {c2,c5,c6}: c2 theta->spectrum, c5 theta->star (front half), c6 EOS->spectrum
PGRID = np.logspace(np.log10(1e-4), np.log10(300.0), 64)

def _eos_curve(d):
    eps = np.r_[np.asarray(d["crust_eps"]), d["outer_core"][1], d["inner_core"][1]]
    P = np.r_[np.asarray(d["crust_P"]), d["outer_core"][2], d["inner_core"][2]]
    m = (P > 1e-12) & (eps > 1e-12); eps, P = eps[m], P[m]
    u, idx = np.unique(P, return_index=True); e = eps[idx]
    keep = e >= np.maximum.accumulate(e) * (1 - 1e-12)
    return np.interp(np.log10(PGRID), np.log10(u[keep]), np.log10(e[keep]))


def load_sample(fn):
    d = np.load(fn, allow_pickle=True)
    fam = np.zeros(5); fam[int(d["family"]) - 1] = 1.0
    x = np.concatenate([np.asarray(d["theta"], float), fam,
                        [float(d["nt1"]), float(d["mass_target"])]])
    if SEAT == "c6":
        x = np.concatenate([_eos_curve(d), [float(d["mass_target"]), float(d["nt1"])]])
    if SEAT == "c5":
        y = np.concatenate([[float(d["M_msun"]), float(d["R_km"]), float(d["Rcc_km"]), float(d["Mmax"])],
                            _eos_curve(d)])
        m = np.ones_like(y)
        extras = dict(i_hz=float(d["i_hz"]), M=float(d["M_msun"]), R=float(d["R_km"]), index=int(d["index"]))
        return x, y, m, extras
    y = np.asarray(d["slot_freqs"], float)          # (30,) Hz, NaN where absent
    m = np.asarray(d["slot_mask"], float)           # (30,) 1 = slot filled
    extras = dict(i_hz=float(d["i_hz"]), M=float(d["M_msun"]), R=float(d["R_km"]),
                  index=int(d["index"]))
    return x, y, m, extras


def build_dataset(dirs):
    X, Y, M, meta = [], [], [], []
    for dd in dirs:
        for fn in sorted(glob.glob(dd.rstrip("/") + "/sample_*.npz")):
            try:
                x, y, m, e = load_sample(fn)
            except Exception as exc:
                print(f"skip {fn}: {exc}"); continue
            X.append(x); Y.append(y); M.append(m); meta.append(e)
    X = np.stack(X); Y = np.stack(Y); M = np.stack(M)
    if SEAT == "c5":
        Ylog = Y     # star targets are already smooth scalars/log-curves; standardised downstream
    else:
        Ylog = np.where(M > 0, np.log(np.maximum(Y, 1e-3)), 0.0)
    return X, Ylog, M, meta


def standardise(A, mean=None, std=None):
    if mean is None:
        mean = A.mean(0); std = A.std(0) + 1e-8
    return (A - mean) / std, mean, std


def init_mlp(key, sizes):
    params = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        key, k1, k2 = jax.random.split(key, 3)
        params.append((jax.random.normal(k1, (a, b)) * np.sqrt(2.0 / a),
                       jnp.zeros(b)))
    return params


def mlp(params, x):
    for W, b in params[:-1]:
        x = jax.nn.gelu(x @ W + b)
    W, b = params[-1]
    return x @ W + b


def masked_mse(params, X, Y, M):
    P = jax.vmap(lambda x: mlp(params, x))(X)
    return jnp.sum(M * (P - Y) ** 2) / jnp.maximum(jnp.sum(M), 1.0)


@jax.jit
def step(params, opt, X, Y, M):
    loss, g = jax.value_and_grad(masked_mse)(params, X, Y, M)
    new_params, new_opt = [], []
    for (W, b), (mW, mb) in zip(params, opt):
        gW, gb = g[len(new_params)]
        mW = 0.9 * mW + 0.1 * gW; mb = 0.9 * mb + 0.1 * gb
        new_params.append((W - LR * mW, b - LR * mb))
        new_opt.append((mW, mb))
    return new_params, new_opt, loss


def main(dirs):
    X, Y, M, meta = build_dataset(dirs)
    n = len(X)
    print(f"dataset: {n} samples, {X.shape[1]} inputs, {Y.shape[1]} slots")
    if n < 40:
        print("NOTE: fewer than ~40 samples; this run only checks the plumbing.")
    rng = np.random.default_rng(SEED)
    order = rng.permutation(n)
    n_test = max(1, int(n * TEST_FRAC)); n_val = max(1, int(n * VAL_FRAC))
    te, va, tr = order[:n_test], order[n_test:n_test + n_val], order[n_test + n_val:]
    Xs, xm, xs = standardise(X[tr])
    Mtr = M[tr]; den = np.maximum(Mtr.sum(0), 1.0)
    ym = (Y[tr] * Mtr).sum(0) / den
    ys = np.sqrt(((Y[tr] - ym) ** 2 * Mtr).sum(0) / den) + 1e-3
    Y = np.where(M > 0, (Y - ym) / ys, 0.0)
    Xva = (X[va] - xm) / xs; Xte = (X[te] - xm) / xs
    key = jax.random.key(SEED)
    params = init_mlp(key, (X.shape[1],) + HIDDEN + (Y.shape[1],))
    opt = [(jnp.zeros_like(W), jnp.zeros_like(b)) for W, b in params]
    Xtr, Ytr, Mtr = map(jnp.asarray, (Xs, Y[tr], M[tr]))
    best = np.inf; best_params = params
    for ep in range(EPOCHS):
        params, opt, loss = step(params, opt, Xtr, Ytr, Mtr)
        if ep % 200 == 0 or ep == EPOCHS - 1:
            vl = float(masked_mse(params, jnp.asarray(Xva), jnp.asarray(Y[va]), jnp.asarray(M[va])))
            if vl < best: best, best_params = vl, params
            print(f"epoch {ep:5d}  train {float(loss):.4f}  val {vl:.4f}")
    # report: per-slot MAE in per cent of frequency, on the held-out test set
    P = np.asarray(jax.vmap(lambda x: mlp(best_params, x))(jnp.asarray(Xte)))
    Ph = P * ys + ym; Yh = Y[te] * ys + ym
    err = np.abs(np.exp(Ph) - np.exp(Yh)) / np.maximum(np.exp(Yh), 1e-9)
    mask = M[te] > 0
    slots = [f"g{i}" for i in range(1, 13)] + ["i"] + [f"s{i}" for i in range(1, 17)] + ["f"]
    print("\nheld-out per-slot median |df|/f where the slot exists:")
    for j, name in enumerate(slots):
        if mask[:, j].sum():
            print(f"  {name:4s}: {np.median(err[mask[:, j], j]) * 100:6.2f}%   (n={int(mask[:, j].sum())})")
    overall = np.median(err[mask]) * 100
    print(f"\noverall median frequency error: {overall:.2f}%  (test n={len(te)})")
    np.savez("fm3_surrogate_oneshot.npz",
             **{f"W{i}": np.asarray(W) for i, (W, b) in enumerate(best_params)},
             **{f"b{i}": np.asarray(b) for i, (W, b) in enumerate(best_params)},
             x_mean=xm, x_std=xs, y_mean=ym, y_std=ys, slots=np.asarray(slots))
    print("weights saved to fm3_surrogate_oneshot.npz")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--seat" in args:
        k = args.index("--seat"); SEAT = args[k + 1]
        globals()["SEAT"] = SEAT
        args = args[:k] + args[k + 2:]
    main(args or ["fm3_ds"])
