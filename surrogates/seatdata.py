# Data half of variants.py, lifted verbatim so the classical baselines can load the
# exact same arrays without JAX installed. Do not edit here: edit variants.py and re-extract.
import glob
import numpy as np

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

