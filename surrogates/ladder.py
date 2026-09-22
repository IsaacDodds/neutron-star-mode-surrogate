# Ladder-parameter head for the one-shot seat (C2): theta -> the whole labelled spectrum, no slot cap.
#   g family : period(n) = a + b*n            (g-modes are evenly spaced in period; 2-param fit reproduces a ladder to ~0.6%)
#   s family : f(n)      = c + d*n + e*n^2    (shear modes ~evenly spaced in frequency; quadratic fit ~2.7%)
#   i mode   : presence logit + log f
#   f mode   : log f, censored at the 3000 Hz scan edge
# "How many modes" is not a head: a rung exists when the ladder places it inside the 10-3000 Hz scan window.
# Loss on every OBSERVED rung (log-frequency squared error) + censoring hinges at the window for the first unobserved
# rung on each ladder, applied only to stars the solver did NOT truncate (capped stars: unknown, no hinge).
# Usage: python ladder.py OUT.csv DIR [DIR ...] [--seeds 3] [--epochs 3000] [--dump prefix]
import sys, glob, csv, re, time, numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", False)
LO,HI=10.0,3000.0; NG,NS=48,48   # max rungs carried per star in the padded target arrays

def _fam(names,fr,k):
    out=[(int(re.sub(r'\D','',n) or 0),fr[i]) for i,n in enumerate(names) if n.startswith(k) and n[1:2].isdigit() and np.isfinite(fr[i])]
    return sorted(out)

def load(dirs):
    X=[];GN=[];GF=[];SN=[];SF=[];IP=[];IF=[];FP=[];FF=[];CAP=[];GZ=[];SZ=[]; seen=set(); dup=0
    for dd in dirs:
        for fn in sorted(glob.glob(dd.rstrip('/')+'/sample_*.npz')):
            try:
                d=np.load(fn,allow_pickle=True); fam=np.zeros(5); fam[int(d['family'])-1]=1
                key_=(int(d['index']),int(d['seed']))
                if key_ in seen: dup+=1; continue   # same draw solved by two nodes: keep one copy (else train/test leakage)
                seen.add(key_)
                names=[str(x) for x in d['mode_names']]; fr=np.asarray(d['mode_row_freqs'],float)
                v=np.asarray(d['mode_valid'],bool); capped=bool(v.sum()==len(v))
                g=_fam(names,fr,'g')[:NG]; s=_fam(names,fr,'s')[:NS]
                gn=np.zeros(NG); gf=np.ones(NG); sn=np.zeros(NS); sf=np.ones(NS)
                for j,(n,f) in enumerate(g): gn[j]=n; gf[j]=f
                for j,(n,f) in enumerate(s): sn[j]=n; sf[j]=f
                M=np.asarray(d['slot_mask'],float); F=np.asarray(d['slot_freqs'],float)
                X.append(np.concatenate([np.asarray(d['theta'],float),fam,[float(d['nt1']),float(d['mass_target'])]]))
                GN.append(gn); GF.append(gf); SN.append(sn); SF.append(sf); CAP.append(float(capped))
                IP.append(M[12]); IF.append(F[12] if M[12]>0 else 1.0); FP.append(M[29]); FF.append(F[29] if M[29]>0 else 1.0)
                GZ.append(float(len(g)==0)); SZ.append(float(len(s)==0))
            except Exception: pass
    D=dict(X=np.stack(X),GN=np.stack(GN),GF=np.stack(GF),SN=np.stack(SN),SF=np.stack(SF),IP=np.array(IP),IF=np.array(IF),FP=np.array(FP),FF=np.array(FF),CAP=np.array(CAP),GZ=np.array(GZ),SZ=np.array(SZ))
    print(f"[load] {len(D['X'])} unique stars ({dup} duplicates dropped), {int(D['CAP'].sum())} capped (no censoring terms), g rungs total {int((D['GN']>0).sum())}, s rungs total {int((D['SN']>0).sum())}",flush=True)
    return D

def build(cfg,n_in,key):
    sizes=[n_in]+cfg["hidden"]; P=[]
    for a,b in zip(sizes[:-1],sizes[1:]):
        key,k=jax.random.split(key); P.append([jax.random.normal(k,(a,b))*np.sqrt(2/(a+b)),jnp.zeros(b)])
    key,k=jax.random.split(key); w=cfg["hidden"][-1]
    P.append([jax.random.normal(k,(w,9))*np.sqrt(1/w),jnp.zeros(9)])   # a,b | c,d,e | i_logit, log f_i | log f_f | (spare)
    return P

def forward(P,x,cfg,key=None,train=False):
    f={"gelu":jax.nn.gelu,"silu":jax.nn.silu}[cfg["act"]]; h=x
    for W,b in P[:-1]:
        z=f(h@W+b)
        if train and cfg["dropout"]>0 and key is not None:
            key,k=jax.random.split(key); keep=jax.random.bernoulli(k,1-cfg["dropout"],z.shape); z=z*keep/(1-cfg["dropout"])
        h=z
    W,b=P[-1]; o=h@W+b
    # g ladder in seconds: a in [1e-4,1] s, b in [1e-5,0.1] s per rung (typical spacing 4 ms)
    a=1e-3*jnp.exp(o[0]); b=4e-3*jnp.exp(o[1])
    c=300.0*jnp.exp(o[2]); d=300.0*jnp.exp(o[3]); e=10.0*o[4]
    return a,b,c,d,e,o[5],o[6],o[7]

def g_freq(a,b,n): return 1.0/(a+b*n)
def s_freq(c,d,e,n): return c+d*n+e*n*n

def losses(P,cfg,x,gn,gf,sn,sf,ip,if_,fp,ff,cap,gz,sz,key,train):
    a,b,c,d,e,il,ilf,flf=forward(P,x,cfg,key,train)
    gm=gn>0; sm=sn>0
    # observed rungs: squared error in log frequency
    lg=jnp.sum(gm*(jnp.log(jnp.maximum(g_freq(a,b,gn),1e-3))-jnp.log(gf))**2)/jnp.maximum(gm.sum(),1.0)
    ls=jnp.sum(sm*(jnp.log(jnp.maximum(s_freq(c,d,e,sn),1e-3))-jnp.log(sf))**2)/jnp.maximum(sm.sum(),1.0)
    # censoring at the window (uncapped stars only): next g rung below 10 Hz; next s rung above 3000 Hz; zero-ladder stars: first rung outside
    ng=jnp.max(gn); ns_=jnp.max(sn)
    gnext=jnp.where(gz>0,1.0,ng+1.0); snext=jnp.where(sz>0,1.0,ns_+1.0)
    hg=(1-cap)*jnp.maximum(0.0,jnp.log(jnp.maximum(g_freq(a,b,gnext),1e-3))-jnp.log(LO))**2
    hs=(1-cap)*jnp.maximum(0.0,jnp.log(HI)-jnp.log(jnp.maximum(s_freq(c,d,e,snext),1e-3)))**2
    # i mode: presence (unknown when capped and absent) + log f when present
    wi=1.0-cap*(1-ip); li=wi*(jnp.maximum(il,0)-il*ip+jnp.log1p(jnp.exp(-jnp.abs(il))))
    lif=ip*(ilf-jnp.log(if_))**2
    # f mode: log f when present; when absent on an uncapped star, hinge above 3000 Hz
    lff=fp*(flf-jnp.log(ff))**2+(1-fp)*(1-cap)*jnp.maximum(0.0,jnp.log(HI)-flf)**2
    return lg,ls,hg,hs,li,lif,lff

def run(cfg,D,seed,epochs,out,name,dump=None):
    X=D["X"]; n=len(X); rng=np.random.default_rng(0); o=rng.permutation(n)
    nt=max(30,int(n*0.15)); nv=max(30,int(n*0.15)); te,va,tr=o[:nt],o[nt:nt+nv],o[nt+nv:]
    xm,xs=X[tr].mean(0),X[tr].std(0)+1e-8; Xs=(X-xm)/xs
    keys=["GN","GF","SN","SF","IP","IF","FP","FF","CAP","GZ","SZ"]
    def pack(idx): return (jnp.asarray(Xs[idx]),)+tuple(jnp.asarray(D[k][idx]) for k in keys)
    key=jax.random.key(seed); P=build(cfg,X.shape[1],key); lr=3e-4
    def total(P,batch,k,train):
        x=batch[0]; rest=batch[1:]
        ks=jax.random.split(k,len(x))
        L=jax.vmap(lambda xi,gn,gf,sn,sf,ip,if_,fp,ff,cap,gz,sz,kk: losses(P,cfg,xi,gn,gf,sn,sf,ip,if_,fp,ff,cap,gz,sz,kk,train))(x,*rest,ks)
        lg,ls,hg,hs,li,lif,lff=[jnp.mean(t) for t in L]
        return lg+ls+0.5*(hg+hs)+0.5*li+lif+lff,(lg,ls,hg,hs,li,lif,lff)
    flat,tree=jax.tree_util.tree_flatten(P); m=[jnp.zeros_like(t) for t in flat]; v=[jnp.zeros_like(t) for t in flat]
    @jax.jit
    def step(P,m,v,t,batch,k):
        (l,parts),g=jax.value_and_grad(lambda P: total(P,batch,k,True),has_aux=True)(P)
        pf,tr_=jax.tree_util.tree_flatten(P); gf,_=jax.tree_util.tree_flatten(g)
        nm=[0.9*a+0.1*b for a,b in zip(m,gf)]; nv=[0.999*a+0.001*b*b for a,b in zip(v,gf)]
        newf=[p-lr*((a/(1-0.9**t))/(jnp.sqrt(b/(1-0.999**t))+1e-8)+cfg["wd"]*p) for p,a,b in zip(pf,nm,nv)]
        return jax.tree_util.tree_unflatten(tr_,newf),nm,nv,l
    evalf=jax.jit(lambda P,batch,k: total(P,batch,k,False))
    A=pack(tr); Va=pack(va); best=np.inf; bestP=P; t0=time.time()
    for ep in range(1,epochs+1):
        key,k=jax.random.split(key); P,m,v,l=step(P,m,v,ep,A,k)
        if ep%100==0:
            vl,_=evalf(P,Va,k); vl=float(vl)
            if vl<best: best,bestP=vl,P
    def score(idx):
        x=jnp.asarray(Xs[idx]); a,b,c,d,e,il,ilf,flf=jax.vmap(lambda xi: forward(bestP,xi,cfg))(x)
        a,b,c,d,e,il,ilf,flf=[np.asarray(t) for t in (a,b,c,d,e,il,ilf,flf)]
        gn,gf,sn,sf=D["GN"][idx],D["GF"][idx],D["SN"][idx],D["SF"][idx]; cap=D["CAP"][idx]>0
        gp=1/(a[:,None]+b[:,None]*gn); sp=c[:,None]+d[:,None]*sn+e[:,None]*sn**2
        ge=np.abs(gp-gf)/gf; se=np.abs(sp-sf)/sf; gm=gn>0; sm=sn>0
        # counts from the window
        nn=np.arange(1,200)[None,:]; gall=1/(a[:,None]+b[:,None]*nn); sall=c[:,None]+d[:,None]*nn+e[:,None]*nn**2
        gc=((gall>=LO)&(gall<=HI)).sum(1); sc=((sall>=LO)&(sall<=HI)&(sall>0)).sum(1); gt=gm.sum(1); st=sm.sum(1)
        ipred=il>0; it=D["IP"][idx]>0; ie=np.abs(np.exp(ilf)-D["IF"][idx])/D["IF"][idx]
        fpres=np.exp(flf)<=HI; ft=D["FP"][idx]>0; fe=np.abs(np.exp(flf)-D["FF"][idx])/D["FF"][idx]
        u=~cap
        return dict(g_med=float(np.median(ge[gm])),s_med=float(np.median(se[sm])),all_med=float(np.median(np.concatenate([ge[gm],se[sm],ie[it]]))),
            i_med=float(np.median(ie[it])),i_acc=float(np.mean((ipred==it)[u|it])),f_med=float(np.median(fe[ft])) if ft.any() else np.nan,f_acc=float(np.mean((fpres==ft)[u|ft])),
            g_cnt_exact=float(np.mean((gc==gt)[u])),g_cnt_1=float(np.mean((np.abs(gc-gt)<=1)[u])),s_cnt_exact=float(np.mean((sc==st)[u])),s_cnt_1=float(np.mean((np.abs(sc-st)<=1)[u])),
            preds=dict(a=a,b=b,c=c,d=d,e=e,il=il,ilf=ilf,flf=flf))
    sv=score(va); st_=score(te)
    row=dict(config=name,seed=seed,n_train=len(tr),val_all=sv["all_med"],**{k:v for k,v in st_.items() if k!="preds"},secs=round(time.time()-t0,1))
    new=not __import__("os").path.exists(out)
    with open(out,"a",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=list(row));
        if new: w.writeheader()
        w.writerow(row)
    if dump: np.savez(f"{dump}_s{seed}.npz",te=te,**{k:D[k][te] for k in keys},**st_["preds"])
    print(f"{name} seed{seed}: val {sv['all_med']*100:.2f}% | test all {st_['all_med']*100:.2f}%  g {st_['g_med']*100:.2f}%  s {st_['s_med']*100:.2f}%  i {st_['i_med']*100:.1f}% (pres {st_['i_acc']*100:.0f}%)  f {st_['f_med']*100:.1f}% (pres {st_['f_acc']*100:.0f}%) | count g exact {st_['g_cnt_exact']*100:.0f}% ±1 {st_['g_cnt_1']*100:.0f}%  s exact {st_['s_cnt_exact']*100:.0f}% ±1 {st_['s_cnt_1']*100:.0f}%  ({row['secs']}s)",flush=True)
    return row

if __name__=="__main__":
    a=sys.argv[1:]; out=a[0]; dirs=[]; seeds=3; epochs=3000; dump=None; i=1
    while i<len(a):
        if a[i]=="--seeds": seeds=int(a[i+1]); i+=2
        elif a[i]=="--epochs": epochs=int(a[i+1]); i+=2
        elif a[i]=="--dump": dump=a[i+1]; i+=2
        else: dirs.append(a[i]); i+=1
    D=load(dirs)
    cfg=dict(hidden=[128]*4,act="silu",dropout=0.2,wd=1e-5)
    for s in range(seeds): run(cfg,D,s,epochs,out,"ladder_d4_w128_silu",dump)
