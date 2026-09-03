# Architecture scan per placement, following Chapter 3's five-step protocol on the shared loader (variants.py):
#   define space -> train candidates (1 seed) -> select top-3 by VALIDATION -> retrain 3 seeds -> test once.
# Also: the ablation rows (v.CONFIGS) at 1 seed, inference wall time per star, and the warm-start (C4) window score
# from the 3-seed ensemble spread of the winner. Every CSV row carries the data fingerprint.
# Usage: python architecture_scan.py SEAT OUT.csv DIR [DIR ...] [--epochs 3000] [--cand-epochs 2000] [--budget-min 60] [--no-ablation]
import sys, os, csv, time, itertools, importlib.util, numpy as np
VPATH=os.environ.get("FM3_VARIANTS",os.path.join(os.path.dirname(os.path.abspath(__file__)),"variants.py"))
spec=importlib.util.spec_from_file_location("v",VPATH); v=importlib.util.module_from_spec(spec); spec.loader.exec_module(v)
import jax, jax.numpy as jnp

def main():
    a=sys.argv[1:]; seat=a[0]; out=a[1]; dirs=[]; epochs=3000; cand_epochs=2000; budget=60.0; ablation=True; i=2
    while i<len(a):
        if a[i]=="--epochs": epochs=int(a[i+1]); i+=2
        elif a[i]=="--cand-epochs": cand_epochs=int(a[i+1]); i+=2
        elif a[i]=="--budget-min": budget=float(a[i+1]); i+=2
        elif a[i]=="--no-ablation": ablation=False; i+=1
        else: dirs.append(a[i]); i+=1
    v.SEAT=seat; X,Y,M=v.load(dirs); fp=v.DATA_FP; t0=time.time()
    print(f"[scan] seat {seat} data {fp} n={len(X)} inputs={X.shape[1]} slots={Y.shape[1]} budget {budget} min",flush=True)
    base=dict(residual=False,dropout=0.2,wd=1e-5,loss="mse",logt=True,std=True,init="xavier",ln=False,wm=1.0)
    # 1. define the space (depth x width x dropout x activation), ordered cheap-first so a budget cut still covers the space
    space=[(d,w,p,act) for d in (2,3,4) for w in (128,256,512) for p in (0.05,0.1,0.2) for act in ("gelu","silu")]
    rows=[]
    def run(cfg,seed,ep,name):
        v.run(cfg,X,Y,M,seed,ep,out,name)
        r=list(csv.DictReader(open(out)))[-1]; r["seat"]=seat; rows.append(r); return r   # the shared run() writes its row to the CSV and returns nothing
    # 2. train candidates, one seed
    for d,w,p,act in space:
        if (time.time()-t0)/60>budget*0.55: print("[scan] candidate budget reached",flush=True); break
        run(dict(base,hidden=[w]*d,act=act,dropout=p),0,cand_epochs,f"{seat}_d{d}_w{w}_p{p}_{act}")
    # 3. select top-3 by validation, 4. retrain with two more seeds, 5. test (v.run tests each seed once)
    cand=[r for r in rows if not r["config"].endswith("_final")]
    best=sorted(cand,key=lambda r: float(r["val_median"]))[:3]
    print("[scan] top-3 by validation:",[(r["config"],round(float(r["val_median"])*100,2)) for r in best],flush=True)
    finals={}
    for r in best:
        nm=r["config"]; parts=nm.split("_"); d=int(parts[1][1:]); w=int(parts[2][1:]); p=float(parts[3][1:]); act=parts[4]
        cfg=dict(base,hidden=[w]*d,act=act,dropout=p); res=[r]
        for seed in (1,2): res.append(run(cfg,seed,epochs,nm+"_final"))
        finals[nm]=(cfg,res)
        meds=[float(x["median"]) for x in res]; print(f"[scan] {nm}: test {np.mean(meds)*100:.2f} +- {np.std(meds)*100:.2f}% over {len(res)} seeds",flush=True)
    # stop rule: if the three winners agree within seed spread, the architecture question is closed for this seat
    means=[np.mean([float(x["median"]) for x in res]) for _,res in finals.values()]; sds=[np.std([float(x["median"]) for x in res]) for _,res in finals.values()]
    closed=(max(means)-min(means))<=max(sds)+1e-9
    print(f"[scan] winners spread {(max(means)-min(means))*100:.2f} pts vs seed sd {max(sds)*100:.2f} -> {'CLOSED: architecture does not move the number' if closed else 'open'}",flush=True)
    # ablation rows at one seed (the grid of Table 3.3 in the method) around the winner's depth/width
    if ablation:
        for nm,cfg in v.CONFIGS.items():
            if (time.time()-t0)/60>budget*0.9: print("[scan] ablation budget reached",flush=True); break
            run(cfg,0,cand_epochs,f"{seat}_abl_{nm}")
    # inference wall time per star for the winner (weights do not matter for cost)
    wn=min(finals,key=lambda k: np.mean([float(x["median"]) for x in finals[k][1]])); cfg=finals[wn][0]
    P=v.build(cfg,X.shape[1],jax.random.key(0)); f=jax.jit(jax.vmap(lambda x: v.forward(P,x,cfg)))
    Xj=jnp.asarray((X-X.mean(0))/(X.std(0)+1e-8)); f(Xj)[0].block_until_ready()
    t1=time.time();
    for _ in range(5): f(Xj)[0].block_until_ready()
    per_star=(time.time()-t1)/5/len(X)
    with open(out.replace(".csv","_summary.csv"),"a",newline="") as fh:
        w=csv.writer(fh); w.writerow(["seat","data","winner","test_median_mean","test_median_sd","imode_mean","mask_mean","inference_s_per_star","closed","n_stars","minutes"])
        res=finals[wn][1]; w.writerow([seat,fp,wn,np.mean([float(x["median"]) for x in res]),np.std([float(x["median"]) for x in res]),np.nanmean([float(x["imode"]) for x in res]),np.mean([float(x["mask_acc"]) for x in res]),per_star,closed,len(X),round((time.time()-t0)/60,1)])
    print(f"[scan] DONE seat {seat}: winner {wn}, inference {per_star*1e3:.3f} ms/star, {(time.time()-t0)/60:.1f} min",flush=True)

if __name__=="__main__": main()
