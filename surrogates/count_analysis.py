# What happens when the one-shot gets "how many modes" wrong. Reads the test-set dumps (preds_s*.npz) that variants.run writes when given a dump path.
import numpy as np, glob, collections
names=["g%d"%k for k in range(1,13)]+["i"]+["s%d"%k for k in range(1,17)]+["f"]
G=slice(0,12); I=12; S=slice(13,29); Fm=29
LO,HI=10.0,3000.0
P=[];L=[];Y=[];M=[]
for fn in sorted(glob.glob("preds_s*.npz")):
    d=np.load(fn); P.append(d["pred"]); L.append(d["logit"]); Y.append(d["Y"]); M.append(d["M"]>0)
P=np.concatenate(P); L=np.concatenate(L); Y=np.concatenate(Y); M=np.concatenate(M)
pm=L>0; n=len(P); print("test stars x seeds:",n)
print("\n== 1. how often is the count wrong, per family ==")
for nm,sl in [("g",G),("s",S)]:
    tc=M[:,sl].sum(1); pc=pm[:,sl].sum(1); d=pc-tc
    print(f"{nm}: exact {np.mean(d==0)*100:.0f}%  within1 {np.mean(np.abs(d)<=1)*100:.0f}%  over {np.mean(d>0)*100:.0f}%  under {np.mean(d<0)*100:.0f}%  MAE {np.abs(d).mean():.2f}")
for nm,k in [("i",I),("f",Fm)]:
    print(f"{nm}: presence acc {np.mean(pm[:,k]==M[:,k])*100:.0f}%  false-absent {np.mean(M[:,k]&~pm[:,k])*100:.1f}%  false-present {np.mean(~M[:,k]&pm[:,k])*100:.1f}%  (true presence {M[:,k].mean()*100:.0f}%)")
print("\n== 2. where are the slot errors: at the ladder edge or interior? ==")
for nm,sl,edge,ratio in [("g",G,LO,None),("s",S,HI,None)]:
    tm=M[:,sl]; pp=pm[:,sl]; yy=Y[:,sl]; pr=P[:,sl]
    err=(tm!=pp); tot=err.sum()
    tc=tm.sum(1)
    # edge slot = last present (index tc-1) or first absent (index tc), for contiguous ladders
    contig=np.array([np.all(r[:c]) and not np.any(r[c:]) for r,c in zip(tm,tc)])
    at_edge=np.zeros_like(err)
    for i in range(n):
        if not contig[i]: continue
        c=tc[i]
        for j in (c-1,c):
            if 0<=j<tm.shape[1]: at_edge[i,j]=True
    e_edge=(err&at_edge&contig[:,None]).sum(); e_int=(err&~at_edge&contig[:,None]).sum(); e_nc=(err&~contig[:,None]).sum()
    print(f"{nm}: {tot} slot errors over {n} stars: at the edge rung {e_edge} ({e_edge/max(tot,1)*100:.0f}%), interior of a clean ladder {e_int} ({e_int/max(tot,1)*100:.0f}%), on holed/zero ladders {e_nc} ({e_nc/max(tot,1)*100:.0f}%)")
    # for edge errors: what frequency did the net predict for the disputed slot, relative to the edge
    fa=[];fp=[]
    for i in range(n):
        if not contig[i]: continue
        c=tc[i]
        if c<tm.shape[1] and pp[i,c]:  # false present at first-absent slot: predicted freq vs edge
            fp.append(pr[i,c])
        if c>=1 and not pp[i,c-1]:      # false absent at last-present slot: true freq vs edge
            fa.append(yy[i,c-1])
    if fp: print(f"   false-PRESENT at first-absent slot: predicted freq median {np.median(fp):.1f} Hz (edge {edge}); {np.mean((np.array(fp)<edge) if nm=='g' else (np.array(fp)>edge))*100:.0f}% of them lie OUTSIDE the window (net says 'a rung is there but out of band')")
    if fa: print(f"   false-ABSENT at last-present slot: true freq median {np.median(fa):.1f} Hz (edge {edge}); 90% within [{np.percentile(fa,5):.1f},{np.percentile(fa,95):.1f}]")
print("\n== 3. does a count error hurt the frequencies? ==")
rel=np.abs(P-Y)/Y
tcg=M[:,G].sum(1); pcg=pm[:,G].sum(1); tcs=M[:,S].sum(1); pcs=pm[:,S].sum(1)
ok=(tcg==pcg)&(tcs==pcs)&(pm[:,I]==M[:,I])
print(f"stars with every count right: {ok.mean()*100:.0f}%  median freq err {np.median(rel[ok][M[ok]])*100:.1f}%   vs stars with a count error: {np.median(rel[~ok][M[~ok]])*100:.1f}%")
for nm,sl in [("g",G),("s",S)]:
    tm=M[:,sl]; pp=pm[:,sl]; e=rel[:,sl]
    agree=tm&pp; dis=tm&~pp
    print(f"{nm}: freq err on slots the net also calls present {np.median(e[agree])*100:.1f}%  vs on true slots the net calls ABSENT {np.median(e[dis])*100:.1f}%  (n={dis.sum()})")
print("\n== 4. could presence be DERIVED from the predicted frequency and the window instead of a mask head? ==")
der=np.zeros_like(pm)
der[:,G]=(P[:,G]>=LO)&(P[:,G]<=HI); der[:,S]=(P[:,S]>=LO)&(P[:,S]<=HI); der[:,Fm]=P[:,Fm]<=HI; der[:,I]=pm[:,I]
for nm,sl in [("g",G),("s",S),("f",[Fm])]:
    print(f"{nm}: mask-head acc {np.mean(pm[:,sl]==M[:,sl])*100:.1f}%   window-derived acc {np.mean(der[:,sl]==M[:,sl])*100:.1f}%   (always-present baseline {np.mean(M[:,sl])*100:.1f}%)")
print("\n== 5. is the mask head calibrated? (reliability of sigmoid(logit)) ==")
p=1/(1+np.exp(-L)); bins=np.linspace(0,1,6)
for a,b in zip(bins[:-1],bins[1:]):
    sel=(p>=a)&(p<b)
    if sel.sum(): print(f"  p in [{a:.1f},{b:.1f}): n={sel.sum():5d}  mean p {p[sel].mean():.2f}  actual presence {M[sel].mean():.2f}")
print("\n== 6. per-slot presence accuracy (mask head) ==")
print(" ".join(f"{nm}:{np.mean(pm[:,k]==M[:,k])*100:.0f}" for k,nm in enumerate(names)))
