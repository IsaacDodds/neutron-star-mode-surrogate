# DENSE-style architecture search (Kasim et al.) for placements whose inputs are RELATED (a curve or profiles on a
# grid): C3 (8 star profiles x 24 radial points + 7 scalars) and C6 (the EOS curve on a 64-point grid + 2 scalars).
# Skeleton fixed: L slots, each slot picks ONE operation from a menu {conv k=3, conv k=5, conv k=7, zero}; every op is a
# 1-D convolution whose units see only a few neighbours of the layer before (shared weights), identity skip always on;
# two fixed fully connected layers and the two heads follow. Search alternates weight steps (train whatever path was
# drawn) and probability steps (rank sampled paths on validation loss); the argmax network is retrained from scratch.
# Loss and metrics mirror fm3_variants.run: masked MSE on standardised log f + 0.5 x BCE on the mask with unknown
# slots at zero weight; median |df|/f on filled test slots; mask accuracy on known slots; i-mode median.
# Usage: python dense_search.py SEAT OUT.csv DIR [DIR ...] [--rounds 12] [--final-epochs 1500] [--slots 4] [--channels 16]
import sys, os, csv, time, importlib.util, numpy as np
VPATH=os.environ.get("FM3_VARIANTS",os.path.join(os.path.dirname(os.path.abspath(__file__)),"variants.py"))
spec=importlib.util.spec_from_file_location("v",VPATH); v=importlib.util.module_from_spec(spec); spec.loader.exec_module(v)
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", False)
KS=(3,5,7); NOPS=4   # ops: conv3, conv5, conv7, zero

def layout(seat,n_in):
    if seat=="c3": return 8,24,n_in-8*24     # channels, grid, scalars
    if seat=="c6": return 6,64,n_in-6*64   # six curves on the 64-point grid + M, n_t1
    raise SystemExit("DENSE is for related inputs: c3 or c6")

def init(key,C,G,S,L,nslot,fc=128):
    ks=jax.random.split(key,3*L+6); P={}
    P["stem"]=[jax.random.normal(ks[0],(3,C,16))*np.sqrt(2/(3*C)),jnp.zeros(16)]; C=16
    for l in range(L):
        for j,k in enumerate(KS): P[f"s{l}k{k}"]=[jax.random.normal(ks[1+3*l+j],(k,C,C))*np.sqrt(2/(k*C)),jnp.zeros(C)]
    P["fc1"]=[jax.random.normal(ks[-5],(C*G+S,fc))*np.sqrt(2/(C*G+S)),jnp.zeros(fc)]
    P["fc2"]=[jax.random.normal(ks[-4],(fc,fc))*np.sqrt(2/fc),jnp.zeros(fc)]
    P["hf"]=[jax.random.normal(ks[-3],(fc,nslot))*np.sqrt(1/fc),jnp.zeros(nslot)]
    P["hm"]=[jax.random.normal(ks[-2],(fc,nslot))*np.sqrt(1/fc),jnp.zeros(nslot)]
    return P

def conv1d(x,W,b):   # x: (G,Cin), W: (k,Cin,Cout), same padding
    k=W.shape[0]; p=k//2; xp=jnp.pad(x,((p,p),(0,0)))
    return sum(xp[i:i+x.shape[0]]@W[i] for i in range(k))+b

def forward(P,x,ops,C,G,S,L,drop_key=None,pdrop=0.0):
    grid=x[:C*G].reshape(C,G).T; sc=x[C*G:]
    h=jax.nn.gelu(conv1d(grid,*P["stem"]))
    for l in range(L):
        o=ops[l]; branch=lambda k: jax.nn.gelu(conv1d(h,*P[f"s{l}k{k}"]))
        z=jax.lax.switch(o,[lambda: branch(3),lambda: branch(5),lambda: branch(7),lambda: jnp.zeros_like(h)])
        h=h+z   # identity skip always on
    f=jnp.concatenate([h.reshape(-1),sc])
    f=jax.nn.gelu(f@P["fc1"][0]+P["fc1"][1])
    if drop_key is not None and pdrop>0: f=f*jax.random.bernoulli(drop_key,1-pdrop,f.shape)/(1-pdrop)
    f=jax.nn.gelu(f@P["fc2"][0]+P["fc2"][1])
    return f@P["hf"][0]+P["hf"][1], f@P["hm"][0]+P["hm"][1]

def main():
    a=sys.argv[1:]; seat=a[0]; out=a[1]; dirs=[]; rounds=12; fepochs=1500; L=4; i=2
    while i<len(a):
        if a[i]=="--rounds": rounds=int(a[i+1]); i+=2
        elif a[i]=="--final-epochs": fepochs=int(a[i+1]); i+=2
        elif a[i]=="--slots": L=int(a[i+1]); i+=2
        else: dirs.append(a[i]); i+=1
    v.SEAT=seat; X,Y,M=v.load(dirs); fp=v.DATA_FP; C,G,S=layout(seat,X.shape[1]); nslot=Y.shape[1]
    print(f"[dense] seat {seat} data {fp} n={len(X)} grid {C}x{G} + {S} scalars, slots {L}, device {jax.devices()[0].platform}",flush=True)
    n=len(X); rng=np.random.default_rng(0); o=rng.permutation(n); nt=max(30,int(n*0.15)); nv=nt; te,va,tr=o[:nt],o[nt:nt+nv],o[nt+nv:]
    xm,xs=X[tr].mean(0),X[tr].std(0); xs=np.where(xs<1e-6*np.max(xs),1.0,xs); Xs=(X-xm)/xs; T=np.log(Y)   # constant columns (shared crust) stay 0, not 1e8
    den=np.maximum(M[tr].sum(0),1.0); tm=(T[tr]*M[tr]).sum(0)/den; ts=np.maximum(np.sqrt(((T[tr]-tm)**2*M[tr]).sum(0)/den),0.05); Tn=np.where(M>0,(T-tm)/ts,0.0)   # floor 5% in log f: rare slots (1-2 fills) must not explode the loss
    W=1.0-(v.UNK if v.UNK is not None and len(v.UNK)==n else np.zeros_like(M))
    pack=lambda idx:(jnp.asarray(Xs[idx]),jnp.asarray(Tn[idx]),jnp.asarray(M[idx]),jnp.asarray(W[idx]))
    A=pack(tr); Va=pack(va); Te=pack(te)
    def loss(P,ops,Xa,Ta,Ma,Wa,key,pdrop):
        keys=jax.random.split(key,len(Xa))
        fr,lg=jax.vmap(lambda x,k: forward(P,x,ops,C,G,S,L,k,pdrop))(Xa,keys)
        Lf=jnp.sum(Ma*(fr-Ta)**2)/jnp.maximum(jnp.sum(Ma),1.0)
        Lm=jnp.sum(Wa*(jnp.maximum(lg,0)-lg*Ma+jnp.log1p(jnp.exp(-jnp.abs(lg)))))/jnp.maximum(jnp.sum(Wa),1.0)
        return Lf+0.5*Lm
    lr=3e-4; wd=1e-5
    def make_step():
        @jax.jit
        def step(P,m,vv,t,ops,Xa,Ta,Ma,Wa,key):
            l,g=jax.value_and_grad(loss)(P,ops,Xa,Ta,Ma,Wa,key,0.1)
            pf,tree=jax.tree_util.tree_flatten(P); gf,_=jax.tree_util.tree_flatten(g)
            nm=[0.9*a_+0.1*b_ for a_,b_ in zip(m,gf)]; nv_=[0.999*a_+0.001*b_*b_ for a_,b_ in zip(vv,gf)]
            newf=[p-lr*((a_/(1-0.9**t))/(jnp.sqrt(b_/(1-0.999**t))+1e-8)+wd*p) for p,a_,b_ in zip(pf,nm,nv_)]
            return jax.tree_util.tree_unflatten(tree,newf),nm,nv_,l
        return step
    step=make_step(); evalj=jax.jit(lambda P,ops,Xa,Ta,Ma,Wa: loss(P,ops,Xa,Ta,Ma,Wa,jax.random.key(0),0.0))
    def metrics(P,ops,idx,batch):
        fr,lg=jax.vmap(lambda x: forward(P,x,ops,C,G,S,L))(batch[0]); fr=np.asarray(fr)*ts+tm; pred=np.exp(fr)
        err=np.abs(pred-Y[idx])/Y[idx]; msk=M[idx]>0; e=err[msk]; Wt=W[idx]>0
        macc=float((((np.asarray(lg)>0)==msk)[Wt]).mean()); im=float(np.median(err[msk[:,v.I_IDX],v.I_IDX])) if msk[:,v.I_IDX].sum() else np.nan
        return float(np.median(e)),macc,im
    # ---- search: alternate weight steps and probability steps ----
    key=jax.random.key(0); P=init(key,C,G,S,L,nslot); flat,_=jax.tree_util.tree_flatten(P); m=[jnp.zeros_like(x) for x in flat]; vv=[jnp.zeros_like(x) for x in flat]
    logits=np.zeros((L,NOPS)); t=0; t0=time.time(); rng=np.random.default_rng(1)
    for r in range(rounds):
        for _ in range(60):   # weight steps on freshly drawn paths
            pr=np.exp(logits)/np.exp(logits).sum(1,keepdims=True); ops=jnp.asarray([rng.choice(NOPS,p=pr[l]) for l in range(L)]); t+=1
            key,k=jax.random.split(key); P,m,vv,l=step(P,m,vv,t,ops,*A,k)
        # probability step: rank sampled paths on validation loss
        pr=np.exp(logits)/np.exp(logits).sum(1,keepdims=True); samples=[]
        for _ in range(8):
            ops=np.array([rng.choice(NOPS,p=pr[l]) for l in range(L)]); vl=float(evalj(P,jnp.asarray(ops),*Va)); samples.append((ops,vl))
        base=np.mean([s[1] for s in samples])
        for ops,vl in samples:
            for l in range(L): logits[l,ops[l]]+=1.5*(base-vl)/max(base,1e-6)
        best_ops=[int(np.argmax(logits[l])) for l in range(L)]; names_=['k3','k5','k7','zero']; bo="-".join(names_[o] for o in best_ops); lo=min(s_[1] for s_ in samples); hi=max(s_[1] for s_ in samples)
        print(f"[dense] round {r+1}/{rounds}: val loss of sampled paths {lo:.4f}..{hi:.4f}; argmax ops {bo} ({(time.time()-t0)/60:.1f} min)",flush=True)
    final_ops=jnp.asarray([int(np.argmax(logits[l])) for l in range(L)]); opname="-".join(['k3','k5','k7','zero'][int(o)] for o in final_ops)
    print(f"[dense] searched architecture: {opname}; slot probabilities {np.round(np.exp(logits)/np.exp(logits).sum(1,keepdims=True),2).tolist()}",flush=True)
    # ---- retrain the argmax network from scratch, 3 seeds, validate for early stopping, test once ----
    new=not os.path.exists(out)
    for seed in (0,1,2):
        key=jax.random.key(100+seed); P=init(key,C,G,S,L,nslot); flat,_=jax.tree_util.tree_flatten(P); m=[jnp.zeros_like(x) for x in flat]; vv=[jnp.zeros_like(x) for x in flat]
        best=np.inf; bestP=P; t1=time.time()
        for ep in range(1,fepochs+1):
            key,k=jax.random.split(key); P,m,vv,l=step(P,m,vv,ep,final_ops,*A,k)
            if ep%100==0:
                vl=float(evalj(P,final_ops,*Va))
                if vl<best: best,bestP=vl,P
        vmed,_,_=metrics(bestP,final_ops,va,Va); med,macc,im=metrics(bestP,final_ops,te,Te)
        row=dict(config=f"dense_{seat}_{opname}",data=fp,seed=seed,n_train=len(tr),val_median=vmed,median=med,imode=im,mask_acc=macc,secs=round(time.time()-t1,1),device=jax.devices()[0].platform)
        with open(out,"a",newline="") as fh:
            w=csv.DictWriter(fh,fieldnames=list(row));
            if new: w.writeheader(); new=False
            w.writerow(row)
        print(f"[dense] {row['config']} seed{seed}: val {vmed*100:.2f}% test {med*100:.2f}% i {im*100:.1f}% mask {macc*100:.1f}% ({row['secs']}s)",flush=True)
    print(f"[dense] DONE {seat} in {(time.time()-t0)/60:.1f} min",flush=True)

if __name__=="__main__": main()
