# Accept/reject classifier on the prior draws, from the batch manifests, with a maximum-mass head.
# Usage: python gatekeeper.py DIR [DIR ...]   (each DIR holds a manifest.csv, or node folders that do)
import sys, os, glob, csv, numpy as np, jax, jax.numpy as jnp, collections
jax.config.update("jax_enable_x64", False)
def cls(status):
    s=status.replace("reject:","")
    if s.startswith("ok"): return "accept"
    if "symmetry energy" in s: return "soft symmetry"
    if "could not evaluate extrapolated" in s: return "no extrapolated n_cc"
    if "no forward crossing" in s: return "no forward crossing"
    if s.startswith("mmax_lt_2"): return "Mmax < 2"
    if "sound speed" in s: return "negative c_s^2"
    if s.startswith("mass_gt_mmax"): return "mass > Mmax"
    if "u=1/8" in s: return "no pasta onset"
    if "neutron drip" in s: return "no drip"
    if "trapz" in s: return None          # code bug, excluded
    return "other"
X,Y,MM=[],[],[]
for fn in [f for d in sys.argv[1:] for f in sorted(glob.glob(os.path.join(d, "**", "manifest.csv"), recursive=True))]:
    lines=[l for l in open(fn).read().splitlines() if l.strip()]
    hdr=[i for i,l in enumerate(lines) if l.startswith("index,")]
    if not hdr: continue
    import io
    for r in csv.DictReader(io.StringIO("\n".join([lines[hdr[0]]]+[l for l in lines if not l.startswith("index,")]))):
        st=r.get("status") or r.get("result") or list(r.values())[1]
        if st is None or "theta" not in r or r["theta"] in (None,""): continue
        c=cls(st); 
        if c is None: continue
        try:
            th=[float(x) for x in r["theta"].split("|")]; fam=np.zeros(5); fam[int(float(r["family"]))-1]=1
            X.append(np.concatenate([th,fam,[float(r["nt1"]),float(r["mass_target"])]])); Y.append(c)
            MM.append(float(r["Mmax"]) if r["Mmax"] not in ("nan","") else np.nan)
        except Exception: pass
X=np.stack(X); classes=sorted(set(Y)); yi=np.array([classes.index(c) for c in Y]); MM=np.array(MM); n=len(X)
print("draws:",n,"| classes:",{c:int((yi==i).sum()) for i,c in enumerate(classes)})
rng=np.random.default_rng(0); o=rng.permutation(n); nt=int(n*0.2); te,tr=o[:nt],o[nt:]
xm,xs=X[tr].mean(0),X[tr].std(0)+1e-8; Xs=(X-xm)/xs; K=len(classes)
key=jax.random.key(0); sizes=(X.shape[1],256,256,K); P=[]
for a,b in zip(sizes[:-1],sizes[1:]):
    key,k=jax.random.split(key); P.append((jax.random.normal(k,(a,b))*np.sqrt(2/(a+b)),jnp.zeros(b)))
def net(P,x):
    for W,b in P[:-1]: x=jax.nn.gelu(x@W+b)
    W,b=P[-1]; return x@W+b
cw=jnp.asarray(len(tr)/(K*np.bincount(yi[tr],minlength=K)+1.0))
def loss(P,Xa,ya):
    lg=jax.vmap(lambda x: net(P,x))(Xa); lp=jax.nn.log_softmax(lg)
    return -jnp.mean(cw[ya]*lp[jnp.arange(len(ya)),ya])
@jax.jit
def step(P,m,v,t,Xa,ya):
    l,g=jax.value_and_grad(loss)(P,Xa,ya)
    nm=[(0.9*a[0]+0.1*b[0],0.9*a[1]+0.1*b[1]) for a,b in zip(m,g)]
    nv=[(0.999*a[0]+0.001*b[0]**2,0.999*a[1]+0.001*b[1]**2) for a,b in zip(v,g)]
    P2=[(W-1e-3*(mw/(1-0.9**t))/(jnp.sqrt(vw/(1-0.999**t))+1e-8), b-1e-3*(mb/(1-0.9**t))/(jnp.sqrt(vb/(1-0.999**t))+1e-8)) for (W,b),(mw,mb),(vw,vb) in zip(P,nm,nv)]
    return P2,nm,nv,l
m=[(jnp.zeros_like(W),jnp.zeros_like(b)) for W,b in P]; v=[(jnp.zeros_like(W),jnp.zeros_like(b)) for W,b in P]
A=(jnp.asarray(Xs[tr]),jnp.asarray(yi[tr]))
for ep in range(1,1501): P,m,v,l=step(P,m,v,ep,*A)
pred=np.asarray(jnp.argmax(jax.vmap(lambda x: net(P,x))(jnp.asarray(Xs[te])),-1)); yt=yi[te]
print(f"\nGATEKEEPER held-out (n={nt}): overall accuracy {100*(pred==yt).mean():.1f}%")
acc_i=classes.index("accept"); print(f"binary accept-vs-reject accuracy: {100*((pred==acc_i)==(yt==acc_i)).mean():.1f}%")
print(f"{'class':22s} {'n':>5s} {'recall':>7s} {'precision':>9s}")
for i,c in enumerate(classes):
    ti=yt==i; pi=pred==i
    if ti.sum()==0: continue
    print(f"{c:22s} {int(ti.sum()):5d} {100*(pred[ti]==i).mean():6.1f}% {100*(yt[pi]==i).mean() if pi.sum() else 0:8.1f}%")
# Mmax head on rows where Mmax is known
ok=np.isfinite(MM); Xr=Xs[ok]; yr=MM[ok]; nr=len(Xr); o2=rng.permutation(nr); t2=o2[:int(nr*0.2)]; r2=o2[int(nr*0.2):]
key,k=jax.random.split(key); Q=[]
for a,b in zip((X.shape[1],256,256,1),(256,256,1)):
    key,k=jax.random.split(key); Q.append((jax.random.normal(k,(a,b))*np.sqrt(2/(a+b)),jnp.zeros(b)))
ym_,ys_=yr[r2].mean(),yr[r2].std()
def lossr(Q,Xa,ya): return jnp.mean((jax.vmap(lambda x: net(Q,x))(Xa)[:,0]-ya)**2)
@jax.jit
def stepr(Q,m,v,t,Xa,ya):
    l,g=jax.value_and_grad(lossr)(Q,Xa,ya)
    nm=[(0.9*a[0]+0.1*b[0],0.9*a[1]+0.1*b[1]) for a,b in zip(m,g)]
    nv=[(0.999*a[0]+0.001*b[0]**2,0.999*a[1]+0.001*b[1]**2) for a,b in zip(v,g)]
    Q2=[(W-1e-3*(mw/(1-0.9**t))/(jnp.sqrt(vw/(1-0.999**t))+1e-8), b-1e-3*(mb/(1-0.9**t))/(jnp.sqrt(vb/(1-0.999**t))+1e-8)) for (W,b),(mw,mb),(vw,vb) in zip(Q,nm,nv)]
    return Q2,nm,nv,l
m=[(jnp.zeros_like(W),jnp.zeros_like(b)) for W,b in Q]; v=[(jnp.zeros_like(W),jnp.zeros_like(b)) for W,b in Q]
A=(jnp.asarray(Xr[r2]),jnp.asarray((yr[r2]-ym_)/ys_))
for ep in range(1,2001): Q,m,v,l=stepr(Q,m,v,ep,*A)
pm=np.asarray(jax.vmap(lambda x: net(Q,x))(jnp.asarray(Xr[t2]))[:,0])*ys_+ym_
print(f"\nMMAX HEAD (n={len(t2)} held-out draws with Mmax known): MAE {np.mean(np.abs(pm-yr[t2])):.3f} Msun, median |err| {np.median(np.abs(pm-yr[t2])):.3f}; threshold-at-2 agreement {100*((pm>=2.0)==(yr[t2]>=2.0)).mean():.1f}%")
