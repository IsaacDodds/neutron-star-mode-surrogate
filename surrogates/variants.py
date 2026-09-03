# Variant sweep for the one-shot seat (C2): every ablation-grid row as a config.
# Pure JAX. Appends one CSV row per (config, seed): held-out median |df|/f with a
# 95% bootstrap interval, i-mode median, mask accuracy.
# Usage: python variants.py OUT.csv DIR [DIR ...] [--configs name1,name2] [--seeds 2] [--epochs 3000]
import sys, glob, csv, os, time
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", False)

CONFIGS = {
 "baseline":      dict(hidden=[256,256,256], act="gelu", residual=False, dropout=0.0, wd=1e-5, loss="mse", logt=True, std=True, init="he", ln=False, wm=1.0),
 "shallow":       dict(hidden=[256],         act="gelu", residual=False, dropout=0.0, wd=1e-5, loss="mse", logt=True, std=True, init="he", ln=False, wm=1.0),
 "wide":          dict(hidden=[512,512,512], act="gelu", residual=False, dropout=0.0, wd=1e-5, loss="mse", logt=True, std=True, init="he", ln=False, wm=1.0),
 "residual":      dict(hidden=[256]*4,       act="gelu", residual=True,  dropout=0.0, wd=1e-5, loss="mse", logt=True, std=True, init="he", ln=False, wm=1.0),
 "july_recipe":   dict(hidden=[256]*4,       act="silu", residual=True,  dropout=0.05,wd=1e-4, loss="huber",logt=True, std=True, init="he", ln=True,  wm=1.0),
 "no_log":        dict(hidden=[256,256,256], act="gelu", residual=False, dropout=0.0, wd=1e-5, loss="mse", logt=False,std=True, init="he", ln=False, wm=1.0),
 "raw_inputs":    dict(hidden=[256,256,256], act="gelu", residual=False, dropout=0.0, wd=1e-5, loss="mse", logt=True, std=False,init="he", ln=False, wm=1.0),
 "dropout01":     dict(hidden=[256,256,256], act="gelu", residual=False, dropout=0.1, wd=1e-5, loss="mse", logt=True, std=True, init="he", ln=False, wm=1.0),
 "layernorm":     dict(hidden=[256,256,256], act="gelu", residual=False, dropout=0.0, wd=1e-5, loss="mse", logt=True, std=True, init="he", ln=True,  wm=1.0),
 "xavier":        dict(hidden=[256,256,256], act="gelu", residual=False, dropout=0.0, wd=1e-5, loss="mse", logt=True, std=True, init="xavier", ln=False, wm=1.0),
 "huber_alone":   dict(hidden=[256,256,256], act="gelu", residual=False, dropout=0.0, wd=1e-5, loss="huber",logt=True, std=True, init="he", ln=False, wm=1.0),
 "no_mask_head":  dict(hidden=[256,256,256], act="gelu", residual=False, dropout=0.0, wd=1e-5, loss="mse", logt=True, std=True, init="he", ln=False, wm=0.0),
 "dropout02":     dict(hidden=[256,256,256], act="gelu", residual=False, dropout=0.2, wd=1e-5, loss="mse", logt=True, std=True, init="he", ln=False, wm=1.0),
 "drop_xav":      dict(hidden=[256,256,256], act="gelu", residual=False, dropout=0.1, wd=1e-5, loss="mse", logt=True, std=True, init="xavier", ln=False, wm=1.0),
 "wide_drop_xav": dict(hidden=[512,512,512], act="gelu", residual=False, dropout=0.1, wd=1e-5, loss="mse", logt=True, std=True, init="xavier", ln=False, wm=1.0),
 "july_drop_xav": dict(hidden=[256]*4,       act="silu", residual=True,  dropout=0.1, wd=1e-4, loss="huber",logt=True, std=True, init="xavier", ln=True,  wm=1.0),
 "wd1e-4":        dict(hidden=[256,256,256], act="gelu", residual=False, dropout=0.0, wd=1e-4, loss="mse", logt=True, std=True, init="he", ln=False, wm=1.0),
 "drop_xav_nolog":dict(hidden=[256,256,256], act="gelu", residual=False, dropout=0.1, wd=1e-5, loss="mse", logt=False,std=True, init="xavier", ln=False, wm=1.0),
 "mild_a":        dict(hidden=[256,256,256], act="gelu", residual=False, dropout=0.05,wd=1e-4, loss="mse", logt=True, std=True, init="xavier", ln=False, wm=1.0, noise=0.02),
 "mild_b":        dict(hidden=[256,256,256], act="gelu", residual=False, dropout=0.05,wd=5e-5, loss="mse", logt=True, std=True, init="xavier", ln=True,  wm=1.0, noise=0.05),
 "mild_c":        dict(hidden=[256,256,256], act="gelu", residual=False, dropout=0.03,wd=1e-4, loss="huber",logt=True, std=True, init="xavier", ln=False, wm=1.0, noise=0.03),
 "mild_d":        dict(hidden=[256]*4,       act="gelu", residual=True,  dropout=0.05,wd=1e-4, loss="mse", logt=True, std=True, init="xavier", ln=False, wm=1.0, noise=0.02),
 "hybrid_targets":dict(hidden=[256,256,256], act="gelu", residual=False, dropout=0.1, wd=1e-5, loss="mse", logt="hybrid", std=True, init="xavier", ln=False, wm=1.0),
 "hybrid_mild":   dict(hidden=[256,256,256], act="gelu", residual=False, dropout=0.05,wd=1e-4, loss="mse", logt="hybrid", std=True, init="xavier", ln=False, wm=1.0, noise=0.02),
 "mild_e":        dict(hidden=[256,256,256], act="gelu", residual=False, dropout=0.1, wd=1e-4, loss="mse", logt=True, std=True, init="xavier", ln=False, wm=1.0, noise=0.02),
 # winner architecture (54-candidate grid) under the full slot scheme, with and without
 # family-balanced frequency loss: 40 g rungs would otherwise outvote the single i slot.
 "winner":        dict(hidden=[128]*4,       act="silu", residual=False, dropout=0.2, wd=1e-5, loss="mse", logt=True, std=True, init="xavier", ln=False, wm=1.0),
 "winner_fambal": dict(hidden=[128]*4,       act="silu", residual=False, dropout=0.2, wd=1e-5, loss="mse", logt=True, std=True, init="xavier", ln=False, wm=1.0, fam_balance=True),
}

SEAT="c2"   # c2: theta -> spectrum (one shot); c3: the solved star (radial profiles + scalars) -> spectrum
            # c6: the EOS curve + mass -> spectrum (post-EOS seat)
XGRID=np.linspace(0.02,1.0,24)
PGRID=np.logspace(np.log10(1e-4),np.log10(300.0),64)
def _lg(a): return np.log10(np.maximum(np.asarray(a,float),1e-50))
EOS_TRIM=0   # crust rows dropped at the seam, summed over the loaded set
def eos_curve(d):
    """The equation of state as the six curves the design specifies, each sampled on the
    same fixed log-pressure grid: log10 eps, Gamma_1, Gamma_eq, log10 c_s^2, log10 mu_sh,
    and log10 n. The segments join in DENSITY, not in pressure, and the pressure steps
    down across the crust-core seam on most stars, so assemble in density order and trim
    the crust rows that reach above the core's first pressure."""
    global EOS_TRIM
    cp=np.asarray(d["crust_P"]); ce=np.asarray(d["crust_eps"])
    oc=d["outer_core"]; ic=d["inner_core"]
    cut=cp<oc[2][0]; EOS_TRIM+=int((~cut).sum())
    P=np.r_[cp[cut],oc[2],ic[2]]
    cols=[np.r_[ce[cut],oc[1],ic[1]],                                   # eps
          np.r_[np.asarray(d["crust_G1"])[cut],oc[3],ic[3]],            # Gamma_1
          np.r_[np.asarray(d["crust_Geq"])[cut],oc[4],ic[4]],           # Gamma_eq
          np.r_[np.asarray(d["crust_mu"])[cut],oc[5],ic[5]],            # mu_sh
          np.r_[np.asarray(d["crust_n"])[cut],oc[0],ic[0]]]             # n
    m=(P>1e-12)&np.isfinite(P)
    P=P[m]; cols=[c[m] for c in cols]
    keep=P>=np.maximum.accumulate(P)*(1-1e-12)
    P=P[keep]; lp=np.log10(P); g=np.log10(PGRID)
    out=[]
    for k,c in enumerate(cols):
        c=np.asarray(c,float)[keep]
        c=np.where(np.isfinite(c),c,0.0)
        out.append(np.interp(g, lp, _lg(c) if k in (0,3,4) else c))
    cs2=out[1]*np.power(10.0,g-out[0])            # Gamma_1 * P / eps, in the log
    out.append(np.log10(np.maximum(cs2,1e-12)))
    return np.concatenate(out)

def star_feats(d):
    r=np.asarray(d['star_r'],float); R=r[-1]; x=r/R
    prof=[_lg(d['star_P']),_lg(d['star_eps']),_lg(d['star_n']),np.asarray(d['star_G1'],float),
          np.asarray(d['star_Geq'],float),np.asarray(d['star_nu'],float),np.asarray(d['star_lam'],float),
          np.asarray(d['star_m'],float)/float(np.asarray(d['star_m'])[-1])]
    cols=[np.interp(XGRID,x,p) for p in prof]
    rt1=float(d['Rt1_km'])/float(d['R_km']); ent=bool(np.isfinite(rt1)); rt1=rt1 if ent else 0.0
    sc=[float(d['M_msun']),float(d['R_km']),float(d['Rcc_km'])/float(d['R_km']),rt1,float(ent),
        np.log10(float(d['Pc'])),float(d['Mmax'])]
    return np.concatenate(cols+[sc])

# Full slot scheme: every labelled root of every star gets a slot. Slots are indexed by
# RUNG NUMBER (the classifier's node count), not by position in a sorted list, and rung
# number is not bounded by the root cap: a star whose scan is truncated from the top can
# carry g45 while g1-g6 were never solved. Observed maxima after relabel_v4 are g97 and s17, and p is now ranked within the star
# so it stops at p3; the 160-root shards may go higher, so the caps carry headroom; load() counts anything that still
# falls outside rather than dropping it silently.
G_MAX=128; S_MAX=48; P_MAX=4
SLOT_NAMES=([f"g{i}" for i in range(1,G_MAX+1)]+["i"]+[f"s{i}" for i in range(1,S_MAX+1)]
            +["f"]+[f"p{i}" for i in range(1,P_MAX+1)])
SLOT_INDEX={n:i for i,n in enumerate(SLOT_NAMES)}
NSLOT=len(SLOT_NAMES); I_IDX=SLOT_INDEX["i"]; N_G=G_MAX

def slot_vectors(d):
    """Rebuild the target vector from the star's own labelled roots (mode_names /
    mode_row_freqs), so nothing the labeller named is dropped. Unclassified roots have
    no name and stay out."""
    names=[str(x) for x in d['mode_names']]; freqs=np.asarray(d['mode_row_freqs'],float)
    y=np.full(NSLOT,np.nan); m=np.zeros(NSLOT); off=0; coll=0
    for nm,f in zip(names,freqs):
        j=SLOT_INDEX.get(nm)
        if j is None:
            if nm[:1] in "gsp" and nm[1:].isdigit(): off+=1   # rung numbered above its cap
            continue
        if not np.isfinite(f): continue
        if m[j]: coll+=1; continue                            # same rung number labelled twice
        y[j]=f; m[j]=1.0
    return y,m,off,coll

I_RULE=None  # per-star: 'strict' | 'entangled' | 'none' - which rule identified the interface mode
DATA_FP=""   # fingerprint of the exact files a run loaded (labels included), stamped into every CSV row
UNK=None   # (n,NSLOT) 1 where a slot's absence is UNKNOWN (star hit the solver's root cap: unsolved, not absent); excluded from the mask loss
def load(dirs):
    global UNK
    global DATA_FP, I_RULE
    import hashlib; h=hashlib.blake2b(digest_size=4)
    X,Y,M,U=[],[],[],[]; seen=set(); dup=0; n_off=0; n_coll=0; IR=[]; n_unres=0
    for dd in dirs:
        for fn in sorted(glob.glob(dd.rstrip('/')+'/sample_*.npz')):
            try:
                d=np.load(fn,allow_pickle=True); fam=np.zeros(5); fam[int(d['family'])-1]=1
                key_=(int(d['index']),int(d['seed']))
                if key_ in seen: dup+=1; continue   # same draw solved by two nodes: keep one copy (else train/test leakage)
                seen.add(key_)
                if SEAT=="c3": X.append(star_feats(d))
                elif SEAT=="c6": X.append(np.concatenate([eos_curve(d),[float(d['mass_target']),float(d['nt1'])]]))
                else: X.append(np.concatenate([np.asarray(d['theta'],float),fam,[float(d['nt1']),float(d['mass_target'])]]))
                y,m,off,coll=slot_vectors(d); n_off+=off; n_coll+=coll
                v=np.asarray(d['mode_valid'],bool); capped=bool(v.sum()==len(v))
                ir=str(d['i_rule']) if 'i_rule' in d.files else 'strict'; IR.append(ir)
                u=(m==0).astype(float) if capped else np.zeros_like(m)
                # 'none' on a star the solver finished means no root carried the seam signature,
                # so the interface mode is unresolved, not absent: unknown, never a negative.
                if ir=='none' and not capped: u[I_IDX]=1.0; n_unres+=1
                Y.append(y); M.append(m); U.append(u)
                h.update(np.array([key_[0],key_[1]],dtype=np.int64).tobytes())
                h.update("|".join(str(x) for x in d['mode_names']).encode())
                h.update(np.round(np.nan_to_num(np.asarray(d['mode_row_freqs'],float)),6).tobytes())
                h.update(ir.encode())
            except Exception: pass
    X=np.stack(X); Y=np.stack(Y); M=np.stack(M); UNK=np.stack(U); DATA_FP=h.hexdigest(); I_RULE=np.array(IR)
    Y=np.where(M>0,np.maximum(Y,1e-3),1.0)
    print(f"[load] data {DATA_FP} | {len(X)} unique stars ({dup} duplicates dropped), {NSLOT} slots, {M.sum(1).mean():.1f} filled per star; "
          f"{int(np.sum([r=='none' or True for r in []]) or (UNK.sum(1)>0).sum())} stars with unknown slots "
          f"({int(UNK.sum())} absences: capped scans plus {n_unres} unresolved i); "
          f"dropped {n_off} roots above a cap and {n_coll} repeated rung numbers; "
          f"i rules {dict(zip(*np.unique(I_RULE,return_counts=True)))}, {n_unres} unresolved i marked unknown",flush=True)
    return X,Y,M

def build(cfg, n_in, key):
    sizes=[n_in]+cfg["hidden"]; P=[]
    for a,b in zip(sizes[:-1],sizes[1:]):
        key,k=jax.random.split(key)
        scale=np.sqrt(2/a) if cfg["init"]=="he" else np.sqrt(2/(a+b))
        P.append([jax.random.normal(k,(a,b))*scale, jnp.zeros(b), jnp.ones(b), jnp.zeros(b)])
    key,k1=jax.random.split(key); key,k2=jax.random.split(key)
    w=cfg["hidden"][-1]
    P.append([jax.random.normal(k1,(w,NSLOT))*np.sqrt(1/w), jnp.zeros(NSLOT)])   # freq head
    P.append([jax.random.normal(k2,(w,NSLOT))*np.sqrt(1/w), jnp.zeros(NSLOT)])   # mask head
    return P

def act_fn(name):
    return {"gelu":jax.nn.gelu,"silu":jax.nn.silu,"relu":jax.nn.relu}[name]

def forward(P, x, cfg, key=None, train=False):
    f=act_fn(cfg["act"]); h=x
    for i,(W,b,g,beta) in enumerate(P[:-2]):
        z=h@W+b
        if cfg["ln"]:
            mu=z.mean(-1,keepdims=True); var=z.var(-1,keepdims=True); z=(z-mu)/jnp.sqrt(var+1e-5)*g+beta
        z=f(z)
        if train and cfg["dropout"]>0 and key is not None:
            key,k=jax.random.split(key); keep=jax.random.bernoulli(k,1-cfg["dropout"],z.shape); z=z*keep/(1-cfg["dropout"])
        h = h+z if (cfg["residual"] and i>0 and z.shape==h.shape) else z
    Wf,bf=P[-2]; Wm,bm=P[-1]
    return h@Wf+bf, h@Wm+bm

def run(cfg, X,Y,M, seed, epochs, out, name, dump=None):
    n=len(X); rng=np.random.default_rng(0); o=rng.permutation(n)
    nt=max(30,int(n*0.15)); nv=max(30,int(n*0.15)); te,va,tr=o[:nt],o[nt:nt+nv],o[nt+nv:]
    xm,xs=(X[tr].mean(0),X[tr].std(0)+1e-8) if cfg["std"] else (np.zeros(X.shape[1]),np.ones(X.shape[1]))
    Xs=(X-xm)/xs
    if cfg["logt"]=="hybrid":
        T=np.where(np.arange(NSLOT)[None,:]>=N_G, np.log(Y), Y)
    else:
        T=np.log(Y) if cfg["logt"] else Y
    den=np.maximum(M[tr].sum(0),1.0); tm=(T[tr]*M[tr]).sum(0)/den; ts=np.maximum(np.sqrt(((T[tr]-tm)**2*M[tr]).sum(0)/den),0.05)   # 5% floor in log f: rare slots with one or two
    # training fills have std ~0, and 1e-3 sent their standardised targets to thousands, which then dominated the loss
    Tn=np.where(M>0,(T-tm)/ts,0.0)
    fam=np.array([n[0] for n in SLOT_NAMES]); SW=np.ones(NSLOT)
    if cfg.get("fam_balance"):
        for ch in set(fam): SW[fam==ch]=1.0/float((fam==ch).sum())
        SW=SW/SW.mean()
    SWj=jnp.asarray(SW)
    key=jax.random.key(seed); P=build(cfg,X.shape[1],key)
    lr=3e-4
    def loss(P,Xa,Ta,Ma,Wa,k):
        if cfg.get("noise",0.0)>0:
            k,kn=jax.random.split(k); Xa=Xa+cfg["noise"]*jax.random.normal(kn,Xa.shape)
        fr,lg=jax.vmap(lambda x: forward(P,x,cfg,None,False))(Xa) if cfg["dropout"]==0 else jax.vmap(lambda x,kk: forward(P,x,cfg,kk,True))(Xa,jax.random.split(k,len(Xa)))
        r=fr-Ta
        if cfg["loss"]=="huber": lf=jnp.where(jnp.abs(r)<1,0.5*r**2,jnp.abs(r)-0.5)
        else: lf=r**2
        Lf=jnp.sum(Ma*SWj*lf)/jnp.maximum(jnp.sum(Ma*SWj),1.0)
        Lm=jnp.sum(Wa*(jnp.maximum(lg,0)-lg*Ma+jnp.log1p(jnp.exp(-jnp.abs(lg)))))/jnp.maximum(jnp.sum(Wa),1.0)
        return Lf+cfg["wm"]*0.5*Lm
    cfg_eval=dict(cfg); cfg_eval["noise"]=0.0; cfg_eval["dropout"]=0.0
    def loss_eval(P,Xa,Ta,Ma,Wa,k):
        fr,lg=jax.vmap(lambda x: forward(P,x,cfg_eval,None,False))(Xa)
        r=fr-Ta; lf=jnp.where(jnp.abs(r)<1,0.5*r**2,jnp.abs(r)-0.5) if cfg["loss"]=="huber" else r**2
        Lf=jnp.sum(Ma*SWj*lf)/jnp.maximum(jnp.sum(Ma*SWj),1.0)
        Lm=jnp.sum(Wa*(jnp.maximum(lg,0)-lg*Ma+jnp.log1p(jnp.exp(-jnp.abs(lg)))))/jnp.maximum(jnp.sum(Wa),1.0)
        return Lf+cfg["wm"]*0.5*Lm
    flat,tree=jax.tree_util.tree_flatten(P)
    m=[jnp.zeros_like(a) for a in flat]; v=[jnp.zeros_like(a) for a in flat]
    @jax.jit
    def step(P,m,v,t,Xa,Ta,Ma,Wa,k):
        l,g=jax.value_and_grad(loss)(P,Xa,Ta,Ma,Wa,k)
        pf,tr_=jax.tree_util.tree_flatten(P); gf,_=jax.tree_util.tree_flatten(g)
        nm=[0.9*a+0.1*b for a,b in zip(m,gf)]; nv=[0.999*a+0.001*b*b for a,b in zip(v,gf)]
        newf=[p-lr*((a/(1-0.9**t))/(jnp.sqrt(b/(1-0.999**t))+1e-8)+cfg["wd"]*p) for p,a,b in zip(pf,nm,nv)]
        return jax.tree_util.tree_unflatten(tr_,newf),nm,nv,l
    W=1.0-(UNK if UNK is not None and len(UNK)==len(X) else np.zeros_like(M))
    A=(jnp.asarray(Xs[tr]),jnp.asarray(Tn[tr]),jnp.asarray(M[tr]),jnp.asarray(W[tr]))
    Va=(jnp.asarray(Xs[va]),jnp.asarray(Tn[va]),jnp.asarray(M[va]),jnp.asarray(W[va]))
    best=np.inf; bestP=P; t0=time.time()
    for ep in range(1,epochs+1):
        key,k=jax.random.split(key)
        P,m,v,l=step(P,m,v,ep,*A,k)
        if ep%100==0:
            vl=float(loss_eval(P,*Va,k))
            if vl<best: best,bestP=vl,P
    frv,_=jax.vmap(lambda x: forward(bestP,x,cfg))(jnp.asarray(Xs[va]))
    frv=np.asarray(frv)*ts+tm; predv=np.exp(frv) if cfg["logt"] is True else (np.where(np.arange(NSLOT)[None,:]>=N_G,np.exp(frv),frv) if cfg["logt"]=="hybrid" else frv)
    errv=np.abs(predv-Y[va])/Y[va]; val_med=float(np.median(errv[M[va]>0]))
    fr,lg=jax.vmap(lambda x: forward(bestP,x,cfg))(jnp.asarray(Xs[te]))
    fr=np.asarray(fr)*ts+tm
    if cfg["logt"]=="hybrid": pred=np.where(np.arange(NSLOT)[None,:]>=N_G, np.exp(fr), fr)
    else: pred=np.exp(fr) if cfg["logt"] else fr
    err=np.abs(pred-Y[te])/Y[te]; msk=M[te]>0
    e=err[msk]; med=float(np.median(e))
    boot=[np.median(np.random.default_rng(s_).choice(e,len(e))) for s_ in range(300)]
    imode=float(np.median(err[msk[:,I_IDX],I_IDX])) if msk[:,I_IDX].sum() else np.nan
    # the entangled interface modes sit on a mixed-character root, so score them apart
    ims={}
    if I_RULE is not None and len(I_RULE)==len(X):
        rte=I_RULE[te]
        for tag in ("strict","entangled"):
            sel=msk[:,I_IDX]&(rte==tag)
            ims[tag]=float(np.median(err[sel,I_IDX])) if sel.sum() else np.nan
    else:
        ims={"strict":np.nan,"entangled":np.nan}
    Wt=W[te]>0; macc=float((((np.asarray(lg)>0)==(M[te]>0))[Wt]).mean())
    # Count metrics: how many modes does the star carry, and how long is its g ladder?
    # Stars that hit the solver's root cap have a censored true count - excluded here.
    pres=np.asarray(lg)>0; true=M[te]>0; keep=~(UNK[te].sum(1)>0) if UNK is not None and len(UNK)==len(X) else np.ones(len(te),bool)
    ctrue=true.sum(1); cpred=pres.sum(1)
    gtrue=true[:,:N_G].sum(1); gpred=pres[:,:N_G].sum(1)
    cnt_mae=float(np.mean(np.abs(cpred-ctrue)[keep])) if keep.sum() else np.nan
    cnt_1=float(np.mean((np.abs(cpred-ctrue)<=1)[keep])) if keep.sum() else np.nan
    g_exact=float(np.mean((gpred==gtrue)[keep])) if keep.sum() else np.nan
    if dump: np.savez(dump,pred=pred,logit=np.asarray(lg),Y=Y[te],M=M[te],te=te,predv=predv,Yv=Y[va],Mv=M[va],va=va)
    row=dict(config=name,data=DATA_FP,seed=seed,n_train=len(tr),val_median=val_med,median=med,ci_lo=float(np.percentile(boot,2.5)),ci_hi=float(np.percentile(boot,97.5)),imode=imode,imode_strict=ims['strict'],imode_entangled=ims['entangled'],mask_acc=macc,cnt_mae=cnt_mae,cnt_within1=cnt_1,g_exact=g_exact,secs=round(time.time()-t0,1))
    new=not os.path.exists(out)
    with open(out,"a",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=list(row)); 
        if new: w.writeheader()
        w.writerow(row)
    print(f"{name:14s} seed{seed}: val {val_med*100:5.2f}% test {med*100:6.2f}% [{row['ci_lo']*100:.1f},{row['ci_hi']*100:.1f}]  i {imode*100:6.2f}%  mask {macc*100:5.1f}%  count MAE {cnt_mae:4.2f} (+-1 {cnt_1*100:4.1f}%, g exact {g_exact*100:4.1f}%)  ({row['secs']}s)", flush=True)

if __name__=="__main__":
    a=sys.argv[1:]; out=a[0]; dirs=[]; names=list(CONFIGS); seeds=2; epochs=3000
    i=1
    while i<len(a):
        if a[i]=="--configs": names=a[i+1].split(","); i+=2
        elif a[i]=="--seeds": seeds=int(a[i+1]); i+=2
        elif a[i]=="--epochs": epochs=int(a[i+1]); i+=2
        elif a[i]=="--seat": SEAT=a[i+1]; i+=2
        else: dirs.append(a[i]); i+=1
    X,Y,M=load(dirs); print("samples:",len(X), flush=True)
    for nm in names:
        for s in range(seeds): run(CONFIGS[nm],X,Y,M,s,epochs,out,nm)
