# For the Results chapter: (1) training and validation loss curves of each seat's winning network, (2) permutation
# feature importance on held-out stars (drop in accuracy when one input is shuffled), grouped physically, and
# (3) learning curves (held-out error against training-set size) for C2 and C3.
# Usage: python curves_and_importance.py OUTDIR DIR [DIR ...]
import sys, os, json, time, importlib.util, numpy as np, jax, jax.numpy as jnp
VPATH=os.environ.get("FM3_VARIANTS",os.path.join(os.path.dirname(os.path.abspath(__file__)),"variants.py"))
spec=importlib.util.spec_from_file_location("v",VPATH); v=importlib.util.module_from_spec(spec); spec.loader.exec_module(v)
OUT=sys.argv[1]; DIRS=sys.argv[2:]; os.makedirs(OUT,exist_ok=True)
WIN={"c2":dict(hidden=[256]*4,act="gelu",dropout=0.2),"c3":dict(hidden=[512]*3,act="silu",dropout=0.2),"c6":dict(hidden=[512]*2,act="silu",dropout=0.2)}
BASE=dict(residual=False,wd=1e-5,loss="mse",logt=True,std=True,init="xavier",ln=False,wm=1.0)
THETA=["n_sat","E_0","K_0","Q_0","Z_0","J","L","K_sym","Q_sym","Z_sym"]
C2_NAMES=THETA+["family 1","family 2","family 3","family 4","family 5","n_t1","M"]
C3_GROUPS={"log P(r)":range(0,24),"log eps(r)":range(24,48),"log n(r)":range(48,72),"Gamma_1(r)":range(72,96),"Gamma_eq(r)":range(96,120),"nu(r)":range(120,144),"lambda(r)":range(144,168),"m(r)/M":range(168,192),"M":[192],"R":[193],"R_cc/R":[194],"R_t1/R":[195],"inner core entered":[196],"log P_c":[197],"M_max":[198]}

def train_logged(cfg,X,Y,M,seed,epochs,log_every=50):
    """Mirror of v.run's training with the loss recorded, returning the best params and the curves."""
    n=len(X); rng=np.random.default_rng(0); o=rng.permutation(n); nt=max(30,int(n*0.15)); te,va,tr=o[:nt],o[nt:2*nt],o[2*nt:]
    xm,xs=X[tr].mean(0),X[tr].std(0)+1e-8; Xs=(X-xm)/xs; T=np.log(Y)
    den=np.maximum(M[tr].sum(0),1.0); tm=(T[tr]*M[tr]).sum(0)/den; ts=np.maximum(np.sqrt(((T[tr]-tm)**2*M[tr]).sum(0)/den),0.05); Tn=np.where(M>0,(T-tm)/ts,0.0)
    W=1.0-(v.UNK if v.UNK is not None and len(v.UNK)==n else np.zeros_like(M))
    key=jax.random.key(seed); P=v.build(cfg,X.shape[1],key); lr=3e-4
    def loss(P,Xa,Ta,Ma,Wa,k,train):
        fr,lg=jax.vmap(lambda x,kk: v.forward(P,x,cfg,kk,train))(Xa,jax.random.split(k,len(Xa))) if train else jax.vmap(lambda x: v.forward(P,x,cfg))(Xa)
        Lf=jnp.sum(Ma*(fr-Ta)**2)/jnp.maximum(jnp.sum(Ma),1.0); Lm=jnp.sum(Wa*(jnp.maximum(lg,0)-lg*Ma+jnp.log1p(jnp.exp(-jnp.abs(lg)))))/jnp.maximum(jnp.sum(Wa),1.0)
        return Lf+0.5*Lm
    flat,tree=jax.tree_util.tree_flatten(P); m=[jnp.zeros_like(a) for a in flat]; vv=[jnp.zeros_like(a) for a in flat]
    @jax.jit
    def step(P,m,vv,t,Xa,Ta,Ma,Wa,k):
        l,g=jax.value_and_grad(lambda P: loss(P,Xa,Ta,Ma,Wa,k,True))(P); pf,tr_=jax.tree_util.tree_flatten(P); gf,_=jax.tree_util.tree_flatten(g)
        nm=[0.9*a+0.1*b for a,b in zip(m,gf)]; nv=[0.999*a+0.001*b*b for a,b in zip(vv,gf)]
        return jax.tree_util.tree_unflatten(tr_,[p-lr*((a/(1-0.9**t))/(jnp.sqrt(b/(1-0.999**t))+1e-8)+cfg["wd"]*p) for p,a,b in zip(pf,nm,nv)]),nm,nv,l
    ev=jax.jit(lambda P,Xa,Ta,Ma,Wa: loss(P,Xa,Ta,Ma,Wa,jax.random.key(0),False))
    A=(jnp.asarray(Xs[tr]),jnp.asarray(Tn[tr]),jnp.asarray(M[tr]),jnp.asarray(W[tr])); Va=(jnp.asarray(Xs[va]),jnp.asarray(Tn[va]),jnp.asarray(M[va]),jnp.asarray(W[va]))
    curve=[]; best=np.inf; bestP=P
    for ep in range(1,epochs+1):
        key,k=jax.random.split(key); P,m,vv,l=step(P,m,vv,ep,*A,k)
        if ep%log_every==0:
            vl=float(ev(P,*Va)); tl=float(ev(P,*A)); curve.append((ep,tl,vl))
            if vl<best: best,bestP=vl,P
    def err(Pp,idx,Xin):
        fr,lg=jax.vmap(lambda x: v.forward(Pp,x,cfg))(jnp.asarray(Xin)); pred=np.exp(np.asarray(fr)*ts+tm); e=np.abs(pred-Y[idx])/Y[idx]; msk=M[idx]>0
        return float(np.median(e[msk])), float(np.median(e[msk[:,v.I_IDX],v.I_IDX])) if msk[:,v.I_IDX].sum() else np.nan
    return bestP,curve,(te,Xs,err)

def main():
    res={}
    for seat,cfg in WIN.items():
        v.SEAT=seat; X,Y,M=v.load(DIRS); fp=v.DATA_FP; cfg=dict(BASE,**cfg); t0=time.time()
        P,curve,(te,Xs,err)=train_logged(cfg,X,Y,M,0,4000)
        base_med,base_i=err(P,te,Xs[te]); print(f"[{seat}] winner retrained: test {base_med*100:.2f}% i {base_i*100:.1f}% ({time.time()-t0:.0f}s)",flush=True)
        res[seat]={"data":fp,"curve":curve,"test_median":base_med,"test_imode":base_i}
        # permutation importance: shuffle one input (or one physical group) across the test set, 5 repeats
        groups={n:[i] for i,n in enumerate(C2_NAMES)} if seat=="c2" else (C3_GROUPS if seat=="c3" else {f"curve {k}":range(k*64,(k+1)*64) for k in range(6)}|{"M":[384],"n_t1":[385]})
        rng=np.random.default_rng(0); imp={}
        for name,cols in groups.items():
            ds=[];dis=[]
            for r in range(5):
                Xp=Xs[te].copy(); perm=rng.permutation(len(te))
                for c in cols: Xp[:,c]=Xp[perm,c]
                mm,ii=err(P,te,Xp); ds.append(mm-base_med); dis.append(ii-base_i)
            imp[name]=(float(np.mean(ds)),float(np.std(ds)),float(np.nanmean(dis)))
        res[seat]["importance"]=imp
        top=sorted(imp.items(),key=lambda kv:-kv[1][0])[:6]; print(f"[{seat}] importance (drop in median error when shuffled): "+", ".join(f"{k} +{d*100:.2f}" for k,(d,s,di) in top),flush=True)
        # learning curve (C2, C3): subsample the training set, keep validation and test fixed
        if seat in ("c2","c3"):
            lc=[]
            n=len(X); rng2=np.random.default_rng(0); o=rng2.permutation(n); nt=max(30,int(n*0.15)); tr_full=o[2*nt:]
            for ntr in (100,200,400,800,len(tr_full)):
                meds=[]
                for s in (0,1):
                    sub=np.random.default_rng(s).choice(tr_full,ntr,replace=False); keep=np.concatenate([o[:2*nt],sub])
                    Xk,Yk,Mk=X[keep],Y[keep],M[keep]; uk=v.UNK[keep]; old=v.UNK; v.UNK=uk
                    Pk,_,(tek,Xsk,errk)=train_logged(cfg,Xk,Yk,Mk,s,2500,log_every=500); v.UNK=old
                    meds.append(errk(Pk,tek,Xsk[tek])[0])
                lc.append((ntr,float(np.mean(meds)),float(np.std(meds)))); print(f"[{seat}] learning curve n={ntr}: {np.mean(meds)*100:.2f} +- {np.std(meds)*100:.2f}%",flush=True)
            res[seat]["learning_curve"]=lc
        json.dump(res,open(f"{OUT}/curves_importance.json","w"),indent=1)
    print("[curves] DONE",flush=True)

if __name__=="__main__": main()
