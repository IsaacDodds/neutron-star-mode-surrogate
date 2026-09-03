# Baselines the Results opener promised: slot-mean predictor and kernel ridge regression, same split/targets as v.run.
import sys, os, glob, importlib.util, numpy as np
spec=importlib.util.spec_from_file_location("v",os.path.join(os.path.dirname(os.path.abspath(__file__)),"variants.py")); v=importlib.util.module_from_spec(spec); spec.loader.exec_module(v)
DIRS=sorted(d for d in glob.glob(os.path.join(sys.argv[1],"*/")) if "capped" not in d)   # usage: python baselines.py DATASET_ROOT
for seat in ("c2","c3"):
    v.SEAT=seat; X,Y,M=v.load(DIRS); n=len(X); rng=np.random.default_rng(0); o=rng.permutation(n); nt=max(30,int(n*0.15)); nv=nt; te,va,tr=o[:nt],o[nt:nt+nv],o[nt+nv:]
    xm,xs=X[tr].mean(0),X[tr].std(0)+1e-8; Xs=(X-xm)/xs; T=np.log(Y); den=np.maximum(M[tr].sum(0),1.0); tm=(T[tr]*M[tr]).sum(0)/den
    msk=M[te]>0
    pred=np.exp(tm)[None,:].repeat(len(te),0); e=np.abs(pred-Y[te])/Y[te]; print(f"[{seat}] slot-mean predictor: test median {np.median(e[msk])*100:.2f}%  i {np.median(e[msk[:,v.I_IDX],v.I_IDX])*100:.1f}%  (n={n}, fp {v.DATA_FP})",flush=True)
    best=None
    for alpha in (1e-3,1e-2,1e-1,1.0):
        for gamma in (0.3/X.shape[1],1.0/X.shape[1],3.0/X.shape[1]):
            Tn=np.where(M>0,T-tm,0.0)   # residual about the slot mean; masked slots contribute zero
            def K(A,B): d=((A[:,None,:]-B[None,:,:])**2).sum(-1); return np.exp(-gamma*d)
            Ktr=K(Xs[tr],Xs[tr]); W=np.linalg.solve(Ktr+alpha*np.eye(len(tr)),Tn[tr])
            pv=np.exp(K(Xs[va],Xs[tr])@W+tm); ev=np.abs(pv-Y[va])/Y[va]; mv=np.median(ev[M[va]>0])
            if best is None or mv<best[0]: best=(mv,alpha,gamma,W)
    mv,alpha,gamma,W=best
    def K(A,B): d=((A[:,None,:]-B[None,:,:])**2).sum(-1); return np.exp(-gamma*d)
    pt=np.exp(K(Xs[te],Xs[tr])@W+tm); e=np.abs(pt-Y[te])/Y[te]
    print(f"[{seat}] kernel ridge (rbf, alpha {alpha}, gamma {gamma:.2e}; chosen on validation {mv*100:.2f}%): test median {np.median(e[msk])*100:.2f}%  i {np.median(e[msk[:,v.I_IDX],v.I_IDX])*100:.1f}%",flush=True)
print("[baselines] DONE")
