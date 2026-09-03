# Front-half seats: C1 (theta -> the six EOS curves on the fixed pressure grid) and C5 (theta -> the solved star:
# M, R, R_cc, M_max and the EOS curve). Flat, unrelated inputs -> MLP grid under the five-step protocol.
# Score: median relative error over curve points (C1) / over the four scalars and the curve (C5), 3 seeds, plus
# inference time per star. Usage: python front_seats.py SEAT OUT.csv DIR [DIR ...] [--budget-min 20]
import sys, os, csv, time, glob, importlib.util, numpy as np, jax, jax.numpy as jnp
VPATH=os.environ.get("FM3_VARIANTS",os.path.join(os.path.dirname(os.path.abspath(__file__)),"variants.py"))
spec=importlib.util.spec_from_file_location("v",VPATH); v=importlib.util.module_from_spec(spec); spec.loader.exec_module(v)
jax.config.update("jax_enable_x64", False)

def load(dirs,seat):
    X=[];Y=[];seen=set()
    for dd in dirs:
        for fn in sorted(glob.glob(dd.rstrip('/')+'/sample_*.npz')):
            try:
                d=np.load(fn,allow_pickle=True); k=(int(d['index']),int(d['seed']))
                if k in seen: continue
                seen.add(k); fam=np.zeros(5); fam[int(d['family'])-1]=1
                x=np.concatenate([np.asarray(d['theta'],float),fam,[float(d['nt1']),float(d['mass_target'])]])
                curve=v.eos_curve(d)   # log10 eps on the fixed log-P grid (64)
                if seat=="c1": y=curve
                else: y=np.concatenate([[float(d['M_msun']),float(d['R_km']),float(d['Rcc_km']),float(d['Mmax'])],curve])
                if np.all(np.isfinite(y)): X.append(x); Y.append(y)
            except Exception: pass
    return np.stack(X),np.stack(Y)

def train(cfg,X,Y,seed,epochs):
    n=len(X); rng=np.random.default_rng(0); o=rng.permutation(n); nt=max(30,int(n*0.15)); te,va,tr=o[:nt],o[nt:2*nt],o[2*nt:]
    xm,xs=X[tr].mean(0),X[tr].std(0); xs=np.where(xs<1e-6*xs.max(),1.0,xs); Xs=(X-xm)/xs
    ym,ys=Y[tr].mean(0),np.maximum(Y[tr].std(0),1e-3); Yn=(Y-ym)/ys
    key=jax.random.key(seed); sizes=[X.shape[1]]+cfg["hidden"]; P=[]
    for a,b in zip(sizes[:-1],sizes[1:]):
        key,k=jax.random.split(key); P.append([jax.random.normal(k,(a,b))*np.sqrt(2/(a+b)),jnp.zeros(b)])
    key,k=jax.random.split(key); P.append([jax.random.normal(k,(sizes[-1],Y.shape[1]))*np.sqrt(1/sizes[-1]),jnp.zeros(Y.shape[1])])
    act={"gelu":jax.nn.gelu,"silu":jax.nn.silu}[cfg["act"]]
    def fwd(P,x,k=None):
        h=x
        for W,b in P[:-1]:
            h=act(h@W+b)
            if k is not None and cfg["dropout"]>0: k,kk=jax.random.split(k); h=h*jax.random.bernoulli(kk,1-cfg["dropout"],h.shape)/(1-cfg["dropout"])
        return h@P[-1][0]+P[-1][1]
    def loss(P,Xa,Ya,k): return jnp.mean((jax.vmap(lambda x,kk: fwd(P,x,kk))(Xa,jax.random.split(k,len(Xa)))-Ya)**2)
    lr=3e-4; wd=1e-5; flat,tree=jax.tree_util.tree_flatten(P); m=[jnp.zeros_like(a) for a in flat]; vv=[jnp.zeros_like(a) for a in flat]
    @jax.jit
    def step(P,m,vv,t,Xa,Ya,k):
        l,g=jax.value_and_grad(loss)(P,Xa,Ya,k); pf,tr_=jax.tree_util.tree_flatten(P); gf,_=jax.tree_util.tree_flatten(g)
        nm=[0.9*a+0.1*b for a,b in zip(m,gf)]; nv=[0.999*a+0.001*b*b for a,b in zip(vv,gf)]
        return jax.tree_util.tree_unflatten(tr_,[p-lr*((a/(1-0.9**t))/(jnp.sqrt(b/(1-0.999**t))+1e-8)+wd*p) for p,a,b in zip(pf,nm,nv)]),nm,nv,l
    A=(jnp.asarray(Xs[tr]),jnp.asarray(Yn[tr])); Va=(jnp.asarray(Xs[va]),jnp.asarray(Yn[va])); best=np.inf; bestP=P
    evalj=jax.jit(lambda P,Xa,Ya: jnp.mean((jax.vmap(lambda x: fwd(P,x))(Xa)-Ya)**2))
    for ep in range(1,epochs+1):
        key,k=jax.random.split(key); P,m,vv,l=step(P,m,vv,ep,*A,k)
        if ep%100==0:
            vl=float(evalj(P,*Va))
            if vl<best: best,bestP=vl,P
    def score(idx):
        pred=np.asarray(jax.vmap(lambda x: fwd(bestP,x))(jnp.asarray(Xs[idx])))*ys+ym; Yt=Y[idx]
        # curves are log10 eps: error in eps = |10^pred - 10^true| / 10^true; scalars relative
        if Yt.shape[1]==64: rel=np.abs(10**pred-10**Yt)/10**Yt; return float(np.median(rel)),float(np.median(rel))
        sc=np.abs(pred[:,:4]-Yt[:,:4])/np.abs(Yt[:,:4]); cv=np.abs(10**pred[:,4:]-10**Yt[:,4:])/10**Yt[:,4:]
        return float(np.median(np.concatenate([sc.ravel(),cv.ravel()]))),float(np.median(sc))
    vmed,_=score(va); tmed,tsc=score(te)
    f=jax.jit(jax.vmap(lambda x: fwd(bestP,x))); Xj=jnp.asarray(Xs); f(Xj).block_until_ready(); t1=time.time()
    for _ in range(5): f(Xj).block_until_ready()
    return vmed,tmed,tsc,(time.time()-t1)/5/len(X)

def main():
    a=sys.argv[1:]; seat=a[0]; out=a[1]; dirs=[]; budget=20.0; i=2
    while i<len(a):
        if a[i]=="--budget-min": budget=float(a[i+1]); i+=2
        else: dirs.append(a[i]); i+=1
    X,Y=load(dirs,seat); v.SEAT="c2"; v.load(dirs); fp=v.DATA_FP; t0=time.time()
    print(f"[front] seat {seat} data {fp} n={len(X)} inputs {X.shape[1]} outputs {Y.shape[1]} device {jax.devices()[0].platform}",flush=True)
    space=[(d,w,p,act) for d in (2,3,4) for w in (128,256,512) for p in (0.0,0.1) for act in ("gelu","silu")]
    rows=[]; new=not os.path.exists(out)
    def rec(name,seed,cfg,ep):
        vmed,tmed,tsc,per=train(cfg,X,Y,seed,ep); r=dict(config=name,data=fp,seed=seed,val_median=vmed,median=tmed,scalars_median=tsc,inference_s_per_star=per,secs=round(time.time()-t0,1))
        nonlocal new
        with open(out,"a",newline="") as fh:
            w=csv.DictWriter(fh,fieldnames=list(r));
            if new: w.writeheader(); new=False
            w.writerow(r)
        rows.append((name,cfg,r)); print(f"[front] {name} seed{seed}: val {vmed*100:.3f}% test {tmed*100:.3f}% (scalars {tsc*100:.3f}%) {per*1e3:.3f} ms/star",flush=True); return r
    for d,w,p,act in space:
        if (time.time()-t0)/60>budget*0.6: print("[front] candidate budget reached",flush=True); break
        rec(f"{seat}_d{d}_w{w}_p{p}_{act}",0,dict(hidden=[w]*d,dropout=p,act=act),1500)
    best=sorted(rows,key=lambda t: t[2]["val_median"])[:3]
    for name,cfg,r in best:
        for seed in (1,2): rec(name+"_final",seed,cfg,3000)
    print(f"[front] DONE {seat}: top-3 {[(n,round(r['val_median']*100,3)) for n,_,r in best]} ({(time.time()-t0)/60:.1f} min)",flush=True)

if __name__=="__main__": main()
