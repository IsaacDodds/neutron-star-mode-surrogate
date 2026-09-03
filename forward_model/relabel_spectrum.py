# relabel_v4 (was v3): reclassify every stored star from its SAVED eigenfunctions with the v3.3 node counter.
# Why: the solver's node counter used a global amplitude floor (1e-6 x max). On stars whose crust amplitude dwarfs the
# core, every core node of the g-ladder was erased -> whole ladders labelled 's' (9% of stars) and rung numbers collapsed
# onto each other (26% of stars). The eigenfunctions on disk (548 points) are enough to recount and reclassify.
# What it rewrites: mode_names, mode_labels, mode_row_freqs (family/rung), slot_names/slot_freqs/slot_mask (30-slot
# vector, kept for compatibility), i_index, i_hz. Originals preserved once under *_v0 keys. Idempotent.
# Usage: python relabel_v2.py DIR [DIR ...]   (each DIR holds sample_*.npz)
import sys, glob, os, numpy as np, csv
MODE_LABELS={0:'invalid',1:'g',2:'i',3:'s',4:'f',5:'p',6:'g_disc',7:'unclassified'}
SLOTS=[f"g{i}" for i in range(1,13)]+["i"]+[f"s{i}" for i in range(1,17)]+["f"]

def node_count(x, rel_floor=1e-3, win=25):
    x=np.asarray(x,float); x=np.where(np.isfinite(x),x,0.0); scale=np.max(np.abs(x)) if x.size else 0.0
    if not np.isfinite(scale) or scale<=0: return 0
    env=np.sqrt(np.convolve(x*x,np.ones(2*win+1)/(2*win+1),mode='same')); env=np.maximum(env,1e-300)
    xn=np.where(np.abs(x)>1e-12*scale,x/env,0.0); s=np.sign(np.where(np.abs(xn)>rel_floor,xn,0.0))
    last=0.0; n=0
    for si in s:
        if si==0: continue
        if last!=0 and si!=last: n+=1
        last=si
    return int(n)

def relabel(d):
    f=np.asarray(d['mode_freqs'],float); v=np.asarray(d['mode_valid'],bool); m=np.asarray(d['mode_metrics'],float).copy()
    er=np.asarray(d['eig_r'],float); ex=np.asarray(d['eig_xr'],float)
    R=float(d['R_km']); rcc=float(d['Rcc_km'])/R          # eig_r is r/R
    fam=int(d['family']); old_labels=np.asarray(d['mode_labels'],int)
    ids=np.where(v)[0]
    core_nodes=np.zeros(len(f),int); global_nodes=np.zeros(len(f),int)
    for j in ids:
        ok=np.isfinite(er[j])&np.isfinite(ex[j]); r=er[j][ok]; x=ex[j][ok]
        global_nodes[j]=node_count(x); core_nodes[j]=node_count(x[r<rcc]) if (r<rcc).sum()>5 else 0
    labels=np.zeros(len(f),int); labels[v]=7
    # i: lowest-frequency zero-core-node root with cusp and transverse jump at Rcc (Neill 2026 rule)
    # v3: strict paper rule with a 30 Hz floor guard (no i-mode in 1,141 clean stars sits below 30 Hz; floor picks were
    # deep g rungs whose core nodes the old counter had erased); then, ONLY if nothing qualifies, the entangled-pair
    # fallback: one core node allowed (a g1/i avoided crossing shares character), same cusp/jump thresholds, below s1.
    i_idx=None; i_rule='none'
    icand=ids[(core_nodes[ids]==0)&(f[ids]>=30.0)&(f[ids]<=1000.0)]   # 30 Hz-1 kHz: the clean population's range; kHz seam-localised roots are a different mode
    if len(icand):
        qual=icand[(m[icand,8]>=0.3)&(m[icand,9]>=0.3)]
        if len(qual): i_idx=int(qual[np.argsort(f[qual])][0]); i_rule='strict'
    if i_idx is None:
        s_pre=[j for j in ids if core_nodes[j]==0 and (int(round(m[j,4]))+int(round(m[j,5])))>0]
        s1=float(np.min(f[s_pre])) if s_pre else np.inf
        fb=[j for j in ids if core_nodes[j]<=1 and m[j,8]>=0.3 and m[j,9]>=0.3 and 30.0<=f[j]<=min(s1,1000.0)]
        if fb: i_idx=int(min(fb,key=lambda j:f[j])); i_rule='entangled'
    if i_idx is not None: labels[i_idx]=2
    # s: zero core nodes with elastic crust nodes
    for j in ids:
        if labels[j]==7 and core_nodes[j]==0 and (int(round(m[j,4]))+int(round(m[j,5])))>0: labels[j]=3
    # g_disc (family 1 only): keep the solver's own choice, it needs the phase-boundary localisation we do not store
    if fam==1:
        for j in ids:
            if old_labels[j]==6 and labels[j]==7: labels[j]=6
    # f: node-free global branch, least crust-confined
    # v4: the fundamental sits at 1.5-3 kHz in every clean star (5th percentile 1530 Hz); a node-free root far below is not f
    fcand=[j for j in ids if labels[j]==7 and global_nodes[j]==0 and core_nodes[j]==0 and f[j]>=1500.0]; f_idx=None
    if fcand: f_idx=int(min(fcand,key=lambda q:m[q,1])); labels[f_idx]=4
    # g below f with core nodes; p above f with global nodes
    if f_idx is not None:
        for j in ids:
            if labels[j]!=7: continue
            if f[j]<f[f_idx] and core_nodes[j]>0: labels[j]=1
            elif f[j]>f[f_idx] and global_nodes[j]>0: labels[j]=5
    # gravity-ladder fallback: core-node roots below every zero-core-node root
    zero=[j for j in ids if core_nodes[j]==0]; floor=float(np.min(f[zero])) if zero else np.inf
    for j in ids:
        if labels[j]==7 and core_nodes[j]>0 and f[j]<floor: labels[j]=1
    # remaining core-node roots below the highest g are ladder members too (interleaved with i/s band edge)
    gl=[j for j in ids if labels[j]==1]
    if gl:
        top=max(f[j] for j in gl)
        for j in ids:
            if labels[j]==7 and core_nodes[j]>0 and f[j]<=top: labels[j]=1
    # names
    gl=sorted([j for j in ids if labels[j]==1],key=lambda j:-f[j]); gcn=[core_nodes[j] for j in gl]
    grung={j:max(c,1) for j,c in zip(gl,gcn)} if len(set(gcn))==len(gcn) else {j:k+max(gcn[0],1) for k,j in enumerate(gl)}
    sids=sorted([j for j in ids if labels[j]==3],key=lambda j:f[j]); srank={j:k+1 for k,j in enumerate(sids)}
    pids=sorted([j for j in ids if labels[j]==5],key=lambda j:f[j]); prank={j:k+1 for k,j in enumerate(pids)}
    dids=sorted([j for j in ids if labels[j]==6],key=lambda j:f[j]); drank={j:k+1 for k,j in enumerate(dids)}
    names=[]; rowf=[]
    for j in ids:
        k=labels[j]
        nm={1:lambda:f'g{grung[j]}',2:lambda:'i',3:lambda:f's{srank[j]}',4:lambda:'f',5:lambda:f'p{prank[j]}',6:lambda:f'g_disc{drank[j]}'}.get(k,lambda:f'unclassified_{j}')()   # v4: p numbered by rank
        names.append(nm); rowf.append(float(f[j]))
    m[:,3]=core_nodes
    sf=np.full(len(SLOTS),np.nan); sm=np.zeros(len(SLOTS),np.uint8)
    for nm,ff in zip(names,rowf):
        if nm in SLOTS:
            q=SLOTS.index(nm)
            if not sm[q]: sf[q]=ff; sm[q]=1
    return dict(mode_names=np.asarray(names,dtype='U16'),mode_row_freqs=np.asarray(rowf),mode_labels=labels.astype(np.int32),mode_metrics=m,
                slot_names=np.asarray(SLOTS,dtype='U12'),slot_freqs=sf,slot_mask=sm,
                i_index=np.asarray(-1 if i_idx is None else i_idx),i_hz=np.asarray(np.nan if i_idx is None else float(f[i_idx])),
                mode_global_nodes=global_nodes.astype(np.int32),i_rule=np.asarray(i_rule),relabel_version=np.asarray('relabel_v4/v3.3-localnodes+entangled+fguard+prank'))

if __name__=="__main__":
    tot=0; changed=0; stats=dict(zero_g_before=0,zero_g_after=0,dup_before=0,dup_after=0,i_before=0,i_after=0)
    for D in sys.argv[1:]:
        for fn in sorted(glob.glob(os.path.join(D,'sample_*.npz'))):
            d=dict(np.load(fn,allow_pickle=True)); tot+=1
            if 'mode_names_v0' not in d:
                for k in ('mode_names','mode_row_freqs','mode_labels','slot_freqs','slot_mask','i_index','i_hz'): d[k+'_v0']=d[k]
            old=[str(x) for x in d['mode_names_v0']]
            new=relabel(d); nn=[str(x) for x in new['mode_names']]
            stats['zero_g_before']+=int(not any(x.startswith('g') and x[1:2].isdigit() for x in old)); stats['zero_g_after']+=int(not any(x.startswith('g') and x[1:2].isdigit() for x in nn))
            stats['dup_before']+=int(len(set(old))<len(old)); stats['dup_after']+=int(len(set(nn))<len(nn))
            stats['i_before']+=int('i' in old); stats['i_after']+=int('i' in nn)
            if nn!=[str(x) for x in d['mode_names']]: changed+=1
            d.update(new); np.savez_compressed(fn,**d)
    print(f"relabel_v2: {tot} stars, {changed} changed | zero-g {stats['zero_g_before']} -> {stats['zero_g_after']} | duplicate names {stats['dup_before']} -> {stats['dup_after']} | i present {stats['i_before']} -> {stats['i_after']}")
