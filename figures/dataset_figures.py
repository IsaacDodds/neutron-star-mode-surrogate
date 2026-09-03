# Regenerates the four Results data-figures + timing table from harvested samples.
# Usage: .venv/bin/python dataset_figures.py [harvest_dir] [out_dir]
# Re-run after every harvest (rsync node fm3_ds dirs into harvest_dir/<node>/ first,
# and relabel any node that missed the on-node relabel_slots pass).
import sys, glob, os
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

H = sys.argv[1] if len(sys.argv) > 1 else 'data'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'figures_out'
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.size': 8, 'axes.linewidth': 0.6, 'figure.dpi': 150})
fam_col = {1: '#5c8ec9', 2: '#eab84a', 3: '#71c498', 4: '#ce7e7e', 5: '#9b7fc9'}
fam_name = {1: 'F1 FOPT', 2: 'F2 two-poly', 3: 'F3 three-poly',
            4: 'F4 cs2 segments', 5: 'F5 peaked conformal'}
F = [f for f in glob.glob(H + '/*/sample_*.npz') if 'aching_wrap' not in f]
D = [dict(np.load(f, allow_pickle=True)) for f in F]
print('loaded', len(D), 'samples')

# A: EOS by family (subsampled for legibility; overlap backtracks trimmed)
fig, ax = plt.subplots(figsize=(3.5, 2.9)); seen = set()
for d in D[::max(1, len(D) // 140)]:
    fam = int(d['family'])
    eps = np.r_[np.asarray(d['crust_eps']), d['outer_core'][1], d['inner_core'][1]]
    P = np.r_[np.asarray(d['crust_P']), d['outer_core'][2], d['inner_core'][2]]
    m = (eps > 1e-3) & (P > 1e-6); eps, P = eps[m], P[m]
    keep = eps >= np.maximum.accumulate(eps) * (1 - 1e-12)
    ax.plot(eps[keep], P[keep], color=fam_col[fam], alpha=0.25, lw=0.5,
            label=fam_name[fam] if fam not in seen else None); seen.add(fam)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'$\varepsilon$ [MeV fm$^{-3}$]'); ax.set_ylabel(r'$P$ [MeV fm$^{-3}$]')
h,l=ax.get_legend_handles_labels()
order=sorted(range(len(l)),key=lambda i:l[i])
leg = ax.legend([h[i] for i in order],[l[i] for i in order],fontsize=6, loc='upper left', framealpha=0.9)
for lh in leg.legend_handles: lh.set_alpha(1); lh.set_linewidth(1.5)
fig.tight_layout(); fig.savefig(f'{OUT}/fig_eos_families.pdf'); plt.close(fig)

# B: M-R curves, 3 per family, from the stored tables (independent TOV re-integration)
K=1.32379e-6; MSUN=1.476625
def _tab(d):
    eps=np.r_[np.asarray(d['crust_eps']),d['outer_core'][1],d['inner_core'][1]]
    P=np.r_[np.asarray(d['crust_P']),d['outer_core'][2],d['inner_core'][2]]
    m=(P>1e-12)&(eps>1e-12); eps,P=eps[m]*K,P[m]*K
    u,idx=np.unique(P,return_index=True); e=eps[idx]
    keep=e>=np.maximum.accumulate(e)*(1-1e-12)
    return np.log10(e[keep]),np.log10(u[keep])
def _mr(lge,lgP,Pc):
    Pf=10**lgP[0]
    epsof=lambda P: 10**np.interp(np.log10(np.maximum(P,Pf)),lgP,lge)
    n=len(Pc); r=1e-4; h=0.004
    m=4/3*np.pi*r**3*epsof(Pc); P=Pc.copy(); alive=np.ones(n,bool)
    Ro=np.zeros(n); Mo=np.zeros(n)
    def f(r,m,P):
        e=epsof(P)
        return 4*np.pi*r*r*e, -(e+P)*(m+4*np.pi*r**3*P)/np.maximum(r*(r-2*m),1e-14)
    while r<26 and alive.any():
        k1m,k1p=f(r,m,P); k2m,k2p=f(r+h/2,m+h/2*k1m,np.maximum(P+h/2*k1p,Pf))
        k3m,k3p=f(r+h/2,m+h/2*k2m,np.maximum(P+h/2*k2p,Pf)); k4m,k4p=f(r+h,m+h*k3m,np.maximum(P+h*k3p,Pf))
        m2=m+h/6*(k1m+2*k2m+2*k3m+k4m); P2=P+h/6*(k1p+2*k2p+2*k3p+k4p)
        died=alive&(P2<=Pf*2); Ro[died]=r+h; Mo[died]=m2[died]; alive&=~died
        m=np.where(alive,m2,m); P=np.where(alive,np.maximum(P2,Pf),P); r+=h
    Ro[alive]=r; Mo[alive]=m[alive]
    return Ro,Mo/MSUN
fig,ax=plt.subplots(figsize=(3.5,2.9))
ax.scatter([float(d['R_km']) for d in D],[float(d['M_msun']) for d in D],s=4,color='0.78',alpha=0.5,edgecolors='none',zorder=1)
for fam in range(1,6):
    for j,d in enumerate([x for x in D if int(x['family'])==fam][:3]):
        lge,lgP=_tab(d); Pc=np.logspace(np.log10(4*K),lgP[-1]-np.log10(1.4),44)
        R,M=_mr(lge,lgP,Pc); k=int(np.argmax(M))
        ax.plot(R[:k+1],M[:k+1],color=fam_col[fam],lw=1.0,alpha=0.9,
                label=fam_name[fam] if j==0 else None,zorder=2)
        ax.scatter([float(d['R_km'])],[float(d['M_msun'])],s=14,color=fam_col[fam],zorder=3,edgecolors='k',linewidths=0.3)
ax.set_xlim(8.5,16); ax.set_ylim(0.4,3.0)
ax.set_xlabel(r'$R$ [km]'); ax.set_ylabel(r'$M$ [$M_\odot$]')
ax.legend(fontsize=6,loc='lower left',framealpha=0.9); fig.tight_layout()
fig.savefig(f'{OUT}/fig_mr_families.pdf'); plt.close(fig)

# C: spectrum atlas
fig, ax = plt.subplots(figsize=(3.5, 3.1))
mode_col = {'g': '#71c498', 'i': '#c44a3c', 's': '#eab84a', 'f': '#5c8ec9', 'p': '#9b7fc9'}
seen = set()
for d in D:
    Mst = float(d['M_msun'])
    for n, fr in zip(d['mode_names'], d['mode_row_freqs']):
        fam = str(n).rstrip('0123456789')
        if fam not in mode_col: continue
        ax.scatter(Mst, fr, s=1.5, color=mode_col[fam], alpha=0.5, edgecolors='none',
                   label=fam if fam not in seen else None); seen.add(fam)
ax.set_yscale('log'); ax.set_xlabel(r'$M$ [$M_\odot$]'); ax.set_ylabel('frequency [Hz]')
ax.legend(fontsize=6, loc='lower right', markerscale=6, framealpha=0.9)
fig.tight_layout(); fig.savefig(f'{OUT}/fig_spectrum_atlas.pdf'); plt.close(fig)

# D: i-mode vs R
fig, ax = plt.subplots(figsize=(3.5, 2.9))
for fam in range(1, 6):
    sel = [d for d in D if int(d['family']) == fam and np.isfinite(float(d['i_hz']))]
    ax.scatter([float(d['R_km']) for d in sel], [float(d['i_hz']) for d in sel],
               s=9, color=fam_col[fam], label=fam_name[fam], alpha=0.85, edgecolors='none')
ax.set_yscale('log'); ax.set_xlabel(r'$R$ [km]'); ax.set_ylabel(r'$f_i$ [Hz]')
ax.legend(fontsize=6, loc='upper right'); fig.tight_layout()
fig.savefig(f'{OUT}/fig_imode_vs_R.pdf'); plt.close(fig)

# E: mode type vs frequency bands
mode_col2={'g':'#71c498','i':'#c44a3c','s':'#eab84a','f':'#5c8ec9','p':'#9b7fc9'}
order=['g','i','s','f','p']; ypos={m:k for k,m in enumerate(order)}
freqs={m:[] for m in order}
for d in D:
    for n,fr in zip(d['mode_names'],d['mode_row_freqs']):
        fam=str(n).rstrip('0123456789')
        if fam in freqs: freqs[fam].append(float(fr))
rng=np.random.default_rng(0)
fig,ax=plt.subplots(figsize=(3.5,2.5))
for m in order:
    x=np.asarray(freqs[m]); y=ypos[m]+rng.uniform(-0.28,0.28,len(x))
    ax.scatter(x,y,s=2,color=mode_col2[m],alpha=0.35,edgecolors='none')
    md=np.median(x); ax.plot([md,md],[ypos[m]-0.34,ypos[m]+0.34],color='k',lw=1.1)
    ax.text(2.2e3 if m=='g' else 11,ypos[m]+0.02,f'n={len(x)}',fontsize=6,va='center',ha='left',color='0.35')
ax.set_xscale('log'); ax.set_xlim(9,3.4e3)
ax.set_yticks(range(len(order)))
ax.set_yticklabels(['g ladder','i interface','s shear','f fundamental','p pressure'])
ax.set_xlabel('frequency [Hz]'); fig.tight_layout()
fig.savefig(f'{OUT}/fig_mode_bands.pdf'); plt.close(fig)

# timing table
t = np.array([d['timings'] for d in D if 'timings' in d]); tot = t.sum(1)
q = lambda x: (np.median(x), np.percentile(x, 90))
rows = [('equation of state', *q(t[:, 0])), ('structure and star', *q(t[:, 1])),
        ('oscillations and labelling', *q(t[:, 2])), ('complete pass', *q(tot))]
with open(f'{OUT}/timing_table.tex', 'w') as fh:
    fh.write('\\begin{tabular}{lrr}\n\\hline\nStage & median [s] & 90th pct [s] \\\\\n\\hline\n')
    for n, md, p90 in rows: fh.write(f'{n} & {md:.1f} & {p90:.1f} \\\\\n')
    fh.write('\\hline\n\\end{tabular}\n')
for n, md, p90 in rows: print(f'{n:28s} median {md:7.1f}s  90th {p90:7.1f}s')
ni = [float(d['i_hz']) for d in D if np.isfinite(float(d['i_hz']))]
print(f'stars {len(D)}  with i {len(ni)}  i median {np.median(ni):.0f} Hz  '
      f'IQR {np.percentile(ni,25):.0f}-{np.percentile(ni,75):.0f}')
