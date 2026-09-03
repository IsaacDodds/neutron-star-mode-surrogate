# C4 warm start: the surrogate says where each mode is and how sure it is; the exact solver scans only there.
# Uncertainty = the 3-seed deep ensemble of the C2 winner (the method's ensemble-as-uncertainty). For each true root of
# each held-out star: is it inside the proposed window mean +- k*sd (k = 2, 3, 4, and a floor of 2% in log f)? and how much
# of the 10-3000 Hz band do the windows cover (the scan cost saved)? Also the count of roots the surrogate did not
# propose at all (mask says absent): those the solver would miss unless a full scan is run.
# Usage: python warm_start.py OUT.csv DIR [DIR ...]
import sys, os, csv, importlib.util, numpy as np, jax
VPATH=os.environ.get("FM3_VARIANTS",os.path.join(os.path.dirname(os.path.abspath(__file__)),"variants.py"))
spec=importlib.util.spec_from_file_location("v",VPATH); v=importlib.util.module_from_spec(spec); spec.loader.exec_module(v)
out=sys.argv[1]; dirs=sys.argv[2:]
v.SEAT="c2"; X,Y,M=v.load(dirs); fp=v.DATA_FP
cfg=dict(hidden=[256]*4,act="gelu",residual=False,dropout=0.2,wd=1e-5,loss="mse",logt=True,std=True,init="xavier",ln=False,wm=1.0)
dumps=[]
for seed in (0,1,2):
    dp=f"/tmp/fm3_ws_s{seed}.npz"; v.run(cfg,X,Y,M,seed,3000,out.replace(".csv","_runs.csv"),"c2_winner_ws",dump=dp); dumps.append(np.load(dp))
te=dumps[0]["te"]; P=np.stack([np.log(d["pred"]) for d in dumps]); L=np.stack([d["logit"] for d in dumps])
mu=P.mean(0); sd=P.std(0); pres=(L.mean(0)>0); Yt=np.log(dumps[0]["Y"]); Mt=dumps[0]["M"]>0
band=np.log(3000)-np.log(10); rows=[]
for k in (2.0,3.0,4.0):
    half=np.maximum(k*sd,0.02); lo=mu-half; hi=mu+half
    inside=(Yt>=lo)&(Yt<=hi)&pres
    covered=float(np.mean(inside[Mt]))                       # true roots inside a proposed window
    missed_absent=float(np.mean((~pres)[Mt]))                # true roots the surrogate did not propose at all
    # scan fraction: union of windows per star / band (windows clipped to the band), averaged over stars
    frac=[]
    for i in range(len(te)):
        segs=sorted([(max(lo[i,j],np.log(10)),min(hi[i,j],np.log(3000))) for j in range(Yt.shape[1]) if pres[i,j]])
        tot=0.0; cur=None
        for a,b in segs:
            if b<=a: continue
            if cur is None or a>cur[1]:
                if cur: tot+=cur[1]-cur[0]
                cur=[a,b]
            else: cur[1]=max(cur[1],b)
        if cur: tot+=cur[1]-cur[0]
        frac.append(tot/band)
    frac=float(np.mean(frac))
    rows.append(dict(data=fp,k=k,roots_inside_window=covered,roots_not_proposed=missed_absent,band_fraction_scanned=frac,scan_cost_saved=1-frac,n_test_stars=len(te)))
    print(f"[warmstart] k={k}: {covered*100:.1f}% of true roots inside a proposed window, {missed_absent*100:.1f}% not proposed, windows cover {frac*100:.1f}% of the band -> scan cost saved {(1-frac)*100:.1f}%",flush=True)
with open(out,"w",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("[warmstart] DONE",flush=True)
