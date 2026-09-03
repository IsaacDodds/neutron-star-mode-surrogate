# ============================================================
# NEILL / NEWTON 2026 FORWARD MODEL PARITY SOLVER
# EOS -> target-mass TOV -> relativistic Cowling elastic modes
#
# Scientific target:
#   Neill et al. (2026), arXiv:2606.09621, as documented in the
#   dissertation forward-model chapter v22 (Sept 2026) and the
#   Yoshida & Lee (2002) relativistic-Cowling elastic equations.
#
# Main API:
#   result = forward_model(
#       theta10, mass_msun, family, inner, nt1,
#       plots=True, solve_modes=True,
#   )
#
# Important design choices (v22-doc parity):
#   * ONE CLDM cell describes the crust from the table floor at
#     n = 1e-5 fm^-3 to n_cc.  Nothing is attached below drip; the
#     BPS 1971 table is carried as a DIAGNOSTIC check only
#     (drip density band, Gamma1 plateau), never as production rows.
#   * neutron drip is located by the cell's own condition
#     d eps_cell / dn = m_n (scan + bisection); it is a diagnostic
#     of the draw, checked against 2.0--3.5e-4 fm^-3.
#   * n_cc is the Neill-2026 tangent crossing extrapolated from
#     u = 1/8 (production rule); the marched true crossing is
#     reported as the extrapolation-error diagnostic.
#   * inner-core family priors/forms follow the 2026 supplement;
#     family 5 imposes VALUE-only seam continuity (c1 solved) with
#     c2 sampled, the v22-doc recorded deviation from the source.
#   * the EOS is extended until the TOV maximum-mass turnover is
#     interior to the pressure grid.
#   * stellar mass is an input; central pressure is inverted.
#   * family-1 sharp FOPT uses the rapid-conversion fluid junction.
#   * final root gate is the locked V3 3e-5 singular-ratio AND
#     boundary-residual cut on the high-resolution operator.
#   * the mode band is 10--3000 Hz (v22 doc); i-mode identity is
#     morphology/topology based; 100--600 Hz is only a soft prior.
#   * extra unclassified roots never invalidate an otherwise valid star.
#
# This file is standalone.  No production/dataset loop runs on import.
# ============================================================

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import jax.scipy as jsp

jax.config.update("jax_enable_x64", True)

SOLVER_VERSION = "FULL-SPECTRUM-PARITY-v3.4-WIDE160-LOCALNODES-REJECTS"

_HBARC,_MN,_MP,_ME,_MMU,_E2=197.32698,939.565,938.272,0.511,105.658,1.4400
_SIGMA_S0,_SIGMA_C0,_BS,_P_SURF,_BETA_C,_ALPHA_C=1.1,0.1,29.9,3.0,0.7,5.5
_CEX=3/4*(3/jnp.pi)**(1/3)*_E2
_A0=(3*jnp.pi**2)**(1/3)*_HBARC

def _imp_channel(n,c,n_s):
    b=10*jnp.log(2.); X,y,ee=(n-n_s)/(3*n_s),-(n-n_s)/n_s,jnp.exp(-b*n/n_s); E=dE=d2E=jnp.array(0.,dtype=jnp.float64)
    for K in range(5):
        u=1-y**(5-K)*ee; du=ee/n_s*((5-K)*y**(4-K)+b*y**(5-K)); t=(5-K)*(4-K)*y**(3-K) if K<4 else 0.
        d2u=-ee/n_s**2*(t+2*b*(5-K)*y**(4-K)+b**2*y**(5-K))
        E+=c[K]*X**K*u; dE+=c[K]*((K*X**(K-1)*u/(3*n_s) if K else 0.)+X**K*du)
        d2E+=c[K]*((K*(K-1)*X**(K-2)*u/(9*n_s**2) if K>=2 else 0.)+(2*K*X**(K-1)*du/(3*n_s) if K else 0.)+X**K*d2u)
    return E,dE,d2E

def _imp_nuclear(n,d,c_snm,c_sym,n_s):
    a,s=_imp_channel(n,c_snm,n_s),_imp_channel(n,c_sym,n_s); return a[0]+d*d*s[0],a[1]+d*d*s[1],a[2]+d*d*s[2],s[0]

def _imp_eps_e(ne):
    x=_HBARC*(3*jnp.pi**2*ne)**(1/3)/_ME
    return _ME**4/(8*jnp.pi**2*_HBARC**3)*(x*(1+2*x*x)*jnp.sqrt(1+x*x)-jnp.log(x+jnp.sqrt(1+x*x)))

def _imp_branch(n,q4,c_snm,c_sym,n_s,inner):
    q=q4 if inner else q4[:3]; ni,xi,rN=jnp.exp(q[0]),1/(1+jnp.exp(-q[1])),jnp.exp(q[2]); ng=jnp.exp(q[3]) if inner else jnp.array(0.,dtype=jnp.float64)
    u,delta_i=(n-ng)/(ni-ng),1-2*xi; Ei,dEi,_,Esymi=_imp_nuclear(ni,delta_i,c_snm,c_sym,n_s); Eg,dEg,d2Eg,_=_imp_nuclear(ng,1.,c_snm,c_sym,n_s) if inner else (0.,0.,0.,0.)
    Bi,Bg=Ei+(1-xi)*_MN+xi*_MP,Eg+_MN; ne=u*xi*ni; Fe=jnp.sqrt((_HBARC*(3*jnp.pi**2*ne)**(1/3))**2+_ME**2); mu_e=Fe-4*_CEX*ne**(1/3)/3
    D=xi**(-_P_SURF)+_BS+(1-xi)**(-_P_SURF); sigma_s=_SIGMA_S0*(2**(_P_SURF+1)+_BS)/D; sigma_c=_SIGMA_C0*_ALPHA_C*sigma_s*(_BETA_C-xi)/_SIGMA_S0
    sigma_sp=-sigma_s*(-_P_SURF*xi**(-_P_SURF-1)+_P_SURF*(1-xi)**(-_P_SURF-1))/D; sigma_cp=_SIGMA_C0*_ALPHA_C/_SIGMA_S0*(sigma_sp*(_BETA_C-xi)-sigma_s)
    f,gu=1-1.5*u**(1/3)+.5*u,1-2*u**(1/3)+u; C0=4*jnp.pi*_E2*xi**2*ni**2*rN**2/5
    epsilon=u*ni*Bi+(1-u)*ng*Bg+3*u*sigma_s/rN+6*u*sigma_c/rN**2+C0*u*f-_CEX*xi**(4/3)*ni**(4/3)*u*(1+u**(1/3))+_imp_eps_e(ne)
    Lambda=ni*Bi-ng*Bg+3*sigma_s/rN+6*sigma_c/rN**2+C0*gu-_CEX*xi**(4/3)*ni**(4/3)+mu_e*xi*ni
    d_ni=u*(Bi+ni*dEi+mu_e*xi+2*C0*f/ni-4*_CEX*xi**(4/3)*ni**(1/3)/3)-Lambda*u/(ni-ng)
    d_xi=u*(ni*(_MP-_MN-4*Esymi*delta_i+mu_e)+3*sigma_sp/rN+6*sigma_cp/rN**2+2*C0*f/xi-4*_CEX*xi**(1/3)*ni**(4/3)/3)
    d_rN=u*(-3*sigma_s/rN**2-12*sigma_c/rN**3+2*C0*f/rN)
    if inner:
        d_ng=(1-u)*(Bg+ng*dEg)+Lambda*(u-1)/(ni-ng); grad=jnp.array([ni*d_ni,xi*(1-xi)*d_xi,rN*d_rN,ng*d_ng])
    else:grad=jnp.array([ni*d_ni,xi*(1-xi)*d_xi,rN*d_rN,q4[3]])
    return grad,n*Lambda/(ni-ng)-epsilon

def _imp_grad_pressure(n,q4,inner,c_snm,c_sym,n_s):
    return jax.lax.cond(inner,lambda _: _imp_branch(n,q4,c_snm,c_sym,n_s,True),lambda _: _imp_branch(n,q4,c_snm,c_sym,n_s,False),operand=None)

def _implicit_one_fixed(n,q4,inner,c_snm,c_sym,n_s):
    g=lambda nn,qq:_imp_grad_pressure(nn,qq,inner,c_snm,c_sym,n_s)[0]; pf=lambda nn,qq:_imp_grad_pressure(nn,qq,inner,c_snm,c_sym,n_s)[1]
    bq,A=jax.jacfwd(g,argnums=(0,1))(n,q4); dqdn=-jnp.linalg.solve(A,bq); P,(dPn,dPq)=jax.value_and_grad(pf,argnums=(0,1))(n,q4); G=n*(dPn+jnp.dot(dPq,dqdn))/P
    return G,jnp.max(jnp.abs(g(n,q4))),A

@jax.jit
def implicit_crust_kernel(n,q4,inner,c_snm,c_sym,n_s):
    return jax.vmap(_implicit_one_fixed,in_axes=(0,0,0,None,None,None))(n,q4,inner,c_snm,c_sym,n_s)


def _jax_crust_energy(n,q4,inner,c_snm,c_sym,n_s):
    ni,xi,rN=jnp.exp(q4[0]),1/(1+jnp.exp(-q4[1])),jnp.exp(q4[2]); ng=jnp.where(inner,jnp.exp(q4[3]),0.)
    u,delta_i=(n-ng)/(ni-ng),1-2*xi; Ei,_,_,_=_imp_nuclear(ni,delta_i,c_snm,c_sym,n_s); Eg,_,_,_=_imp_nuclear(ng,1.,c_snm,c_sym,n_s)
    Bi,Bg=Ei+(1-xi)*_MN+xi*_MP,Eg+_MN; ne=u*xi*ni
    D=xi**(-_P_SURF)+_BS+(1-xi)**(-_P_SURF); sigma_s=_SIGMA_S0*(2**(_P_SURF+1)+_BS)/D; sigma_c=_SIGMA_C0*_ALPHA_C*sigma_s*(_BETA_C-xi)/_SIGMA_S0
    f=1-1.5*u**(1/3)+.5*u; C0=4*jnp.pi*_E2*xi**2*ni**2*rN**2/5
    return u*ni*Bi+(1-u)*ng*Bg+3*u*sigma_s/rN+6*u*sigma_c/rN**2+C0*u*f-_CEX*xi**(4/3)*ni**(4/3)*u*(1+u**(1/3))+_imp_eps_e(ne)

def _jax_crust_bounds(n,inner,n_s):
    qxmin=jnp.log(1e-6/(1-1e-6)); qxmax=jnp.log((_BETA_C-1e-6)/(1-_BETA_C+1e-6))
    # A cluster is matter sitting near its own saturation point (v22 doc), so the
    # cluster density is bounded on BOTH sides.  The ceiling 2 n_s stops the
    # expansion being read at super-saturation garbage; the floor stops the
    # no-gas OUTER branch collapsing onto a dilute artifact minimum past
    # physical drip (a fake cluster at n_i << n_s whose drip condition never
    # crosses).  The inner branch keeps the natural n_i > n bound only, since
    # real clusters approach dissolution from n_i ~ 0.5 n_s near u -> 1.
    ni_floor=jnp.where(inner,n*(1+1e-8),jnp.maximum(n*(1+1e-8),0.5*n_s))
    lo=jnp.array([jnp.log(ni_floor),qxmin,jnp.log(.5),jnp.where(inner,jnp.log(jnp.maximum(1e-16,n*1e-10)),0.)])
    hi=jnp.array([jnp.log(2*n_s),qxmax,jnp.log(50.),jnp.where(inner,jnp.log(n*(1-1e-6)),0.)])
    return lo,hi

def _jax_crust_local_one(n,q0,inner,c_snm,c_sym,n_s):
    lo,hi=_jax_crust_bounds(n,inner,n_s); q=jnp.clip(q0,lo,hi); eye=jnp.eye(4,dtype=jnp.float64); alphas=2.**(-jnp.arange(10,dtype=jnp.float64))
    def body(_,carry):
        q=carry; g=lambda qq:_imp_grad_pressure(n,qq,inner,c_snm,c_sym,n_s)[0]; gv=g(q); H=jax.jacfwd(g)(q); H=(H+H.T)/2
        scale=jnp.maximum(1.,jnp.max(jnp.abs(jnp.diag(H)))); Hreg=H+1e-10*scale*eye; d=jnp.linalg.solve(Hreg,-gv); d=d/jnp.maximum(1.,jnp.max(jnp.abs(d))/0.15)
        cand=jnp.clip(q[None,:]+alphas[:,None]*d[None,:],lo,hi); evals=jax.vmap(lambda z:_jax_crust_energy(n,z,inner,c_snm,c_sym,n_s))(cand); e0=_jax_crust_energy(n,q,inner,c_snm,c_sym,n_s)
        k=jnp.argmin(evals); qn=cand[k]; en=evals[k]
        # If damped Newton cannot lower E, take a small diagonally-scaled gradient step.
        dg=-gv/(jnp.maximum(jnp.abs(jnp.diag(H)),1e-8)); dg=dg/jnp.maximum(1.,jnp.max(jnp.abs(dg))/0.10); qg=jnp.clip(q+dg,lo,hi); eg=_jax_crust_energy(n,qg,inner,c_snm,c_sym,n_s)
        qn=jnp.where((en<e0),qn,jnp.where(eg<e0,qg,q)); return qn
    q=jax.lax.fori_loop(0,48,body,q); g=_imp_grad_pressure(n,q,inner,c_snm,c_sym,n_s)[0]; e=_jax_crust_energy(n,q,inner,c_snm,c_sym,n_s)
    ni,xi,rN=jnp.exp(q[0]),1/(1+jnp.exp(-q[1])),jnp.exp(q[2]); ng=jnp.where(inner,jnp.exp(q[3]),0.); u=(n-ng)/(ni-ng)
    return q,e,u,jnp.max(jnp.abs(g))

def _jax_crust_multistart_one(n,warm,inner,c_snm,c_sym,n_s):
    ni=jnp.array([n_s,n_s,1.1*n_s,.9*n_s]); xi=jnp.array([.30,.20,.25,.15]); r=jnp.array([6.,8.,5.,10.]); ngf=jnp.array([.001,.01,.05,.2])
    generic=jnp.stack([jnp.log(jnp.clip(ni,n*(1+1e-6),2*n_s)),jnp.log(xi/(1-xi)),jnp.log(r),jnp.where(inner,jnp.log(jnp.maximum(1e-16,ngf*n)),0.)],axis=1)
    starts=jnp.concatenate([warm[None,:],generic],axis=0)
    q,e,u,g=jax.vmap(lambda z:_jax_crust_local_one(n,z,inner,c_snm,c_sym,n_s))(starts)
    niq=jnp.exp(q[:,0]); xiq=1/(1+jnp.exp(-q[:,1])); rN=jnp.exp(q[:,2]); ng=jnp.where(inner,jnp.exp(q[:,3]),0.)
    valid=(u>0)&(u<1)&(xiq<_BETA_C)&(niq>n*(1+1e-6))&(rN>.5001)&(rN<49.99)&((~inner)|((ng>0)&(ng<n*(1-1e-6))))&(g<1e-5)
    score=jnp.where(valid,e,jnp.inf); k=jnp.argmin(score); anygood=jnp.any(valid)
    P=_imp_grad_pressure(n,q[k],inner,c_snm,c_sym,n_s)[1]; return q[k],e[k],P,u[k],g[k],anygood,k

@jax.jit
def jax_crust_local_kernel(n,q0,inner,c_snm,c_sym,n_s):
    return jax.vmap(_jax_crust_multistart_one,in_axes=(0,0,0,None,None,None))(n,q0,inner,c_snm,c_sym,n_s)

@jax.jit
def jax_crust_point_kernel(n,q0,inner,c_snm,c_sym,n_s):
    return _jax_crust_multistart_one(n,q0,inner,c_snm,c_sym,n_s)


def _core_channel(n,c,n_s):
    b=10*jnp.log(2.); X,y,ee=(n-n_s)/(3*n_s),-(n-n_s)/n_s,jnp.exp(-b*n/n_s); E=dE=d2E=jnp.array(0.,dtype=jnp.float64)
    for K in range(5):
        u=1-y**(5-K)*ee; du=ee/n_s*((5-K)*y**(4-K)+b*y**(5-K)); t=(5-K)*(4-K)*y**(3-K) if K<4 else 0.
        d2u=-ee/n_s**2*(t+2*b*(5-K)*y**(4-K)+b**2*y**(5-K)); d1=(K*X**(K-1)*u/(3*n_s) if K else 0.)+X**K*du
        d2=(K*(K-1)*X**(K-2)*u/(9*n_s**2) if K>=2 else 0.)+(2*K*X**(K-1)*du/(3*n_s) if K else 0.)+X**K*d2u
        E+=c[K]*X**K*u; dE+=c[K]*d1; d2E+=c[K]*d2
    return E,dE,d2E

def _fermi_eps_jax(n,m):
    x=_HBARC*(3*jnp.pi**2*jnp.maximum(n,0.))**(1/3)/m
    return m**4/(8*jnp.pi**2*_HBARC**3)*(x*(1+2*x*x)*jnp.sqrt(1+x*x)-jnp.log(x+jnp.sqrt(1+x*x)))

def _core_point_fixed(n,c_snm,c_sym,n_s):
    E_snm,dE_snm,d2E_snm=_core_channel(n,c_snm,n_s); E_sym,dE_sym,d2E_sym=_core_channel(n,c_sym,n_s)
    def beta_step(_,z):
        a,d=z; delta=(a+d)/2; n_p=n*(1-delta)/2; mu=_MN-_MP+4*delta*E_sym+4*_CEX*n_p**(1/3)/3
        R_e=jnp.sqrt(jnp.maximum(0.,_A0**2*(mu**2-_ME**2)+16*_CEX**2*_ME**2/9)); ne=jnp.where(mu>=.9999975*_ME,((4*_CEX*mu/3+R_e)/(_A0**2-16*_CEX**2/9))**3,0.)
        R_mu=jnp.sqrt(jnp.maximum(0.,_A0**2*(mu**2-_MMU**2)+16*_CEX**2*_MMU**2/9)); nm=jnp.where(mu>=.9999975*_MMU,((4*_CEX*mu/3+R_mu)/(_A0**2-16*_CEX**2/9))**3,0.); g=n_p-ne-nm
        return jnp.where(g>0,delta,a),jnp.where(g>0,d,delta)
    a,d=jax.lax.fori_loop(0,60,beta_step,(jnp.array(0.),jnp.array(1.))); delta=(a+d)/2; n_n,n_p=n*(1+delta)/2,n*(1-delta)/2
    mu=_MN-_MP+4*delta*E_sym+4*_CEX*n_p**(1/3)/3
    R_e=jnp.sqrt(jnp.maximum(0.,_A0**2*(mu**2-_ME**2)+16*_CEX**2*_ME**2/9)); ne=jnp.where(mu>=.9999975*_ME,((4*_CEX*mu/3+R_e)/(_A0**2-16*_CEX**2/9))**3,0.)
    R_mu=jnp.sqrt(jnp.maximum(0.,_A0**2*(mu**2-_MMU**2)+16*_CEX**2*_MMU**2/9)); nm=jnp.where(mu>=.9999975*_MMU,((4*_CEX*mu/3+R_mu)/(_A0**2-16*_CEX**2/9))**3,0.)
    E_N,dE_N,d2E_N=E_snm+delta**2*E_sym,dE_snm+delta**2*dE_sym,d2E_snm+delta**2*d2E_sym; ee,em=_fermi_eps_jax(ne,_ME),_fermi_eps_jax(nm,_MMU)
    ex_p,ex_l=-_CEX*n_p**(4/3),-_CEX*(ne**(4/3)+nm**(4/3)); epsilon=n_n*_MN+n_p*_MP+n*E_N+ee+em+ex_p+ex_l; P=n**2*dE_N+mu*(ne+nm)-ee-em+ex_p/3-ex_l
    F_e=jnp.sqrt((_HBARC*(3*jnp.pi**2*ne)**(1/3))**2+_ME**2); F_mu=jnp.sqrt((_HBARC*(3*jnp.pi**2*nm)**(1/3))**2+_MMU**2); Fms=jnp.where(nm>0,F_mu,1.)
    dP1=2*n*dE_N+n**2*d2E_N+(F_e**2-_ME**2)/(3*F_e)*ne/n+jnp.where(nm>0,(F_mu**2-_MMU**2)/(3*Fms)*nm/n,0.)+4*(ex_p+ex_l)/(9*n); G1=n*dP1/P
    Res,Rms=jnp.maximum(R_e,1e-30),jnp.maximum(R_mu,1e-30); chie=3*ne**(2/3)*(4*_CEX/3+_A0**2*mu/Res)/(_A0**2-16*_CEX**2/9)
    chim=jnp.where(nm>0,3*nm**(2/3)*(4*_CEX/3+_A0**2*mu/Rms)/(_A0**2-16*_CEX**2/9),0.)
    Pie=(F_e**2-_ME**2)/(3*F_e)-4*_CEX*ne**(1/3)/9; Pim=jnp.where(nm>0,(F_mu**2-_MMU**2)/(3*Fms)-4*_CEX*nm**(1/3)/9,0.); pc=Pie*chie+Pim*chim; np23=n_p**(-2/3)
    dmun=4*delta*dE_sym+2*_CEX*(1-delta)*np23/9; dmud=4*E_sym-2*_CEX*n*np23/9
    dPn=2*n*dE_N+n**2*d2E_N+pc*dmun-2*_CEX*n_p**(1/3)*(1-delta)/9; dPd=2*delta*n**2*dE_sym+pc*dmud+2*_CEX*n_p**(1/3)*n/9
    ddn=((1-delta)/2-(chie+chim)*dmun)/(n/2+(chie+chim)*dmud); Geq=n*(dPn+dPd*ddn)/P; cs2=Geq*P/(epsilon+P)
    return jnp.array([epsilon,P,G1,Geq,cs2,0.])

@jax.jit
def core_kernel(n_grid,c_snm,c_sym,n_s):return jax.vmap(_core_point_fixed,in_axes=(0,None,None,None))(n_grid,c_snm,c_sym,n_s)


# Transition bisections as single compiled JAX loops.
@jax.jit
def jax_drip_bisect_kernel(left,right,qleft,c_snm,c_sym,n_s):
    def step(_,carry):
        lo,hi,qlo,ok=carry; mid=(lo+hi)/2
        q,e,P,u,g,good,k=_jax_crust_multistart_one(mid,qlo,jnp.array(False),c_snm,c_sym,n_s)
        drip=_jax_crust_drip_value(mid,q,jnp.array(False),c_snm,c_sym,n_s); take=drip<0
        return jnp.where(take,mid,lo),jnp.where(take,hi,mid),jnp.where(take,q,qlo),ok&good
    lo,hi,qlo,ok=jax.lax.fori_loop(0,35,step,(left,right,qleft,jnp.array(True)))
    return (lo+hi)/2,qlo,ok

@jax.jit
def jax_dripE_bisect_kernel(left,right,qleft,c_snm,c_sym,n_s):
    # Refine the energy-criterion drip: below drip the with-gas cell empties
    # (n_g/n < 1e-4), above it the gas is real.  Bisect that switch.
    def step(_,carry):
        lo,hi,qlo,ok=carry; mid=(lo+hi)/2
        q,e,P,u,g,good,k=_jax_crust_multistart_one(mid,qlo,jnp.array(True),c_snm,c_sym,n_s)
        ngfrac=jnp.exp(q[3])/mid; take=(ngfrac<1e-4)|(~good)
        return jnp.where(take,mid,lo),jnp.where(take,hi,mid),jnp.where(take,q,qlo),ok
    lo,hi,qlo,ok=jax.lax.fori_loop(0,30,step,(left,right,qleft,jnp.array(True)))
    return (lo+hi)/2,qlo,ok

@jax.jit
def jax_u18_bisect_kernel(left,right,qleft,c_snm,c_sym,n_s):
    def step(_,carry):
        lo,hi,qlo,ok=carry; mid=(lo+hi)/2
        q,e,P,u,g,good,k=_jax_crust_multistart_one(mid,qlo,jnp.array(True),c_snm,c_sym,n_s); take=u<1/8
        return jnp.where(take,mid,lo),jnp.where(take,hi,mid),jnp.where(take,q,qlo),ok&good
    lo,hi,qlo,ok=jax.lax.fori_loop(0,25,step,(left,right,qleft,jnp.array(True)))
    return (lo+hi)/2,qlo,ok


# Production crust march (v22 doc): warm-started multistart minimisation at every
# grid density, on either branch.  Also returns the drip condition value and the
# clustered-minus-uniform energy per baryon so drip and the marched crossing can
# be located on the host without re-solving.
def _crust_march_step(q,n,inner,c_snm,c_sym,n_s):
    q1,e,P,u,g,good,k=_jax_crust_multistart_one(n,q,inner,c_snm,c_sym,n_s)
    drip=_jax_crust_drip_value(n,q1,inner,c_snm,c_sym,n_s)
    diff=e/n-_core_point_fixed(n,c_snm,c_sym,n_s)[0]/n
    qnext=jnp.where(good,q1,q)
    return qnext,(q1,e,P,u,g,drip,diff,good)

@jax.jit
def jax_crust_march_kernel(n_grid,q0,inner,c_snm,c_sym,n_s):
    return jax.lax.scan(lambda c,n:_crust_march_step(c,n,inner,c_snm,c_sym,n_s),q0,n_grid)[1]


# Exact local equilibrium parameter sensitivity.  q is already the converged cell solution;
# the response dq/dtheta comes from differentiating grad_q epsilon = 0, not from differentiating optimiser iterations.
def _theta_coeffs(theta):
    E0,K0,Q0,Z0,J,L,Ksym,Qsym,Zsym,n_s=theta
    return jnp.array([E0,0.,K0/2,Q0/6,Z0/24]),jnp.array([J,L,Ksym/2,Qsym/6,Zsym/24]),n_s

def _crust_epu_theta(theta,n,q4,inner):
    cs,cv,ns=_theta_coeffs(theta); e=_jax_crust_energy(n,q4,inner,cs,cv,ns); P=_imp_grad_pressure(n,q4,inner,cs,cv,ns)[1]
    ni=jnp.exp(q4[0]); ng=jnp.where(inner,jnp.exp(q4[3]),0.); u=(n-ng)/(ni-ng)
    return jnp.array([e,P,u])

def _crust_param_jac_one(theta,n,q4,inner):
    cs,cv,ns=_theta_coeffs(theta)
    g=lambda th,q:_imp_grad_pressure(n,q,inner,*_theta_coeffs(th))[0]
    A=jax.jacfwd(g,argnums=1)(theta,q4); B=jax.jacfwd(g,argnums=0)(theta,q4); dq=-jnp.linalg.solve(A,B)
    Yt,Yq=jax.jacfwd(lambda th,q:_crust_epu_theta(th,n,q,inner),argnums=(0,1))(theta,q4)
    return Yt+Yq@dq,jnp.max(jnp.abs(g(theta,q4)))

crust_param_jac_kernel=jax.jit(_crust_param_jac_one)

@jax.jit
def core_param_jac_kernel(theta,n):
    def f(th):
        cs,cv,ns=_theta_coeffs(th); return _core_point_fixed(n,cs,cv,ns)
    return jax.jacfwd(f)(theta)


def _jax_crust_drip_value(n,q4,inner,c_snm,c_sym,n_s):
    ni,xi,rN=jnp.exp(q4[0]),1/(1+jnp.exp(-q4[1])),jnp.exp(q4[2]); ng=jnp.where(inner,jnp.exp(q4[3]),0.)
    u,delta_i=(n-ng)/(ni-ng),1-2*xi; Ei,_,_,_=_imp_nuclear(ni,delta_i,c_snm,c_sym,n_s); Eg,_,_,_=_imp_nuclear(ng,1.,c_snm,c_sym,n_s)
    Bi,Bg=Ei+(1-xi)*_MN+xi*_MP,Eg+_MN; ne=u*xi*ni; Fe=jnp.sqrt((_HBARC*(3*jnp.pi**2*ne)**(1/3))**2+_ME**2); mu_e=Fe-4*_CEX*ne**(1/3)/3
    D=xi**(-_P_SURF)+_BS+(1-xi)**(-_P_SURF); sigma_s=_SIGMA_S0*(2**(_P_SURF+1)+_BS)/D; sigma_c=_SIGMA_C0*_ALPHA_C*sigma_s*(_BETA_C-xi)/_SIGMA_S0
    gu=1-2*u**(1/3)+u; C0=4*jnp.pi*_E2*xi**2*ni**2*rN**2/5
    Lambda=ni*Bi-ng*Bg+3*sigma_s/rN+6*sigma_c/rN**2+C0*gu-_CEX*xi**(4/3)*ni**(4/3)+mu_e*xi*ni
    return Lambda/(ni-ng)-_MN

def _jax_crust_valid(n,q4,inner,u,g,n_s):
    ni,xi,rN=jnp.exp(q4[0]),1/(1+jnp.exp(-q4[1])),jnp.exp(q4[2]); ng=jnp.where(inner,jnp.exp(q4[3]),0.)
    return (u>0)&(u<1)&(xi<_BETA_C)&(ni>n*(1+1e-6))&(rN>.5001)&(rN<49.99)&((~inner)|((ng>0)&(ng<n*(1-1e-6))))&(g<1e-5)

def _crust_props_branch(n,q4,c_snm,c_sym,n_s,inner):
    ni,xi,rN=jnp.exp(q4[0]),1/(1+jnp.exp(-q4[1])),jnp.exp(q4[2]); ng=jnp.exp(q4[3]) if inner else jnp.array(0.,dtype=jnp.float64)
    u,delta_i=(n-ng)/(ni-ng),1-2*xi; Ei,dEi,d2Ei,Esymi=_imp_nuclear(ni,delta_i,c_snm,c_sym,n_s); Eg,dEg,d2Eg,_=_imp_nuclear(ng,1.,c_snm,c_sym,n_s) if inner else (0.,0.,0.,0.)
    Bi,Bg=Ei+(1-xi)*_MN+xi*_MP,Eg+_MN; ne=u*xi*ni; Fe=jnp.sqrt((_HBARC*(3*jnp.pi**2*ne)**(1/3))**2+_ME**2); mu_e=Fe-4*_CEX*ne**(1/3)/3
    D=xi**(-_P_SURF)+_BS+(1-xi)**(-_P_SURF); sigma_s=_SIGMA_S0*(2**(_P_SURF+1)+_BS)/D; sigma_c=_SIGMA_C0*_ALPHA_C*sigma_s*(_BETA_C-xi)/_SIGMA_S0
    f,gu=1-1.5*u**(1/3)+.5*u,1-2*u**(1/3)+u; C0=4*jnp.pi*_E2*xi**2*ni**2*rN**2/5
    epsilon=u*ni*Bi+(1-u)*ng*Bg+3*u*sigma_s/rN+6*u*sigma_c/rN**2+C0*u*f-_CEX*xi**(4/3)*ni**(4/3)*u*(1+u**(1/3))+_imp_eps_e(ne)
    Lambda=ni*Bi-ng*Bg+3*sigma_s/rN+6*sigma_c/rN**2+C0*gu-_CEX*xi**(4/3)*ni**(4/3)+mu_e*xi*ni; P=n*Lambda/(ni-ng)-epsilon
    fp,fpp=-.5*u**(-2/3)+.5,u**(-5/3)/3; dPgas=ng/(1-u)*(2*ng*dEg+ng**2*d2Eg) if inner else 0.; dPe=ne*(Fe**2-_ME**2)/(3*Fe); dPC=C0*u**2*(2*fp+u*fpp); dPX=-4*_CEX*xi**(4/3)*ni**(4/3)*u**(4/3)/9; G1=(dPgas+dPe+dPC+dPX)/P
    A=4*jnp.pi*rN**3*ni/3; Z=xi*A; Xn=(1-u)*ng/n; mu_cryst=0.1106*(4*jnp.pi/3)**(1/3)*(n*(1-Xn)/A)**(4/3)*Z**2*_E2; drip=Lambda/(ni-ng)-_MN
    return jnp.array([epsilon,P,G1,u,mu_cryst,A,Z,Xn,drip])


def _crust_props_one(n,q4,inner,c_snm,c_sym,n_s):
    return jax.lax.cond(inner,lambda _: _crust_props_branch(n,q4,c_snm,c_sym,n_s,True),lambda _: _crust_props_branch(n,q4,c_snm,c_sym,n_s,False),operand=None)


@jax.jit
def crust_props_kernel(n,q4,inner,active,c_snm,c_sym,n_s):
    def one(nn,qq,ii,aa):return jax.lax.cond(aa,lambda _:_crust_props_one(nn,qq,ii,c_snm,c_sym,n_s),lambda _:jnp.zeros(9,dtype=jnp.float64),operand=None)
    return jax.vmap(one)(n,q4,inner,active)


@jax.jit
def crust_props_point_kernel(n,q4,inner,c_snm,c_sym,n_s):
    return _crust_props_one(n,q4,inner,c_snm,c_sym,n_s)


# ============================================================
# Global units and numerical controls
# ============================================================
C_CGS = 2.99792458e10
G_CGS = 6.67430e-8
MSUN_CGS = 1.98847e33
AMU_CGS = 1.66053906660e-24
MEVFM3_TO_ERGCM3 = 1.602176634e33
MEVFM3_TO_GEOM = MEVFM3_TO_ERGCM3 * G_CGS / C_CGS**4
MODE_TINY = jnp.asarray(1e-40, dtype=jnp.float64)
TOV_TINY = jnp.asarray(1e-300, dtype=jnp.float64)

# EOS grids.  The high-density ceiling is extended dynamically.
N_OUTER_CRUST = 160
OUTER_N_MIN = 1.0e-5   # fm^-3; the v22-doc table floor
OUTER_N_MAX = 6.0e-4   # fm^-3; must contain neutron drip (2.0--3.5e-4 band)
N_CLDM = 520
N_OUTER_CORE = 420
N_INNER = 1000

# TOV controls.
TOV_DR = jnp.asarray(1000.0, dtype=jnp.float64)  # 10 m
N_TOV = 4000  # 40 km search ceiling
N_MR = 72
MMAX_MARGIN = 3
MMAX_EXTEND_FACTOR = 1.45
MMAX_MAX_N = 6.0  # fm^-3, numerical emergency ceiling, not a sampled prior.
MASS_TOL_MSUN = 2.0e-5

# Mode controls: locked V3 high-resolution path.
N_SCAN_CORE_LOG = 161
N_SCAN_CORE_RAD = 700
N_SCAN_CRUST = 360
N_EIGEN_CORE_LOG = 3001
N_EIGEN_CORE_RAD = 11032
N_EIGEN_CRUST = 900
N_FREQ = 512
N_CANDIDATES = 96
N_COARSE_MODES = 56
N_MODES = 48
FREQ_MIN_HZ = 10.0
FREQ_MAX_HZ = 3000.0
COARSE_RATIO_MAX = 5.0e-2
FINAL_RATIO_MAX = 3.0e-5
DEDUP_HZ = 0.20
CANDIDATE_BATCH = 4  # lax.map mini-batch; tune on the target GPU.
RECON_BATCH = 4

MODE_LABELS = {
    0: "invalid",
    1: "g",
    2: "i",
    3: "s",
    4: "f",
    5: "p",
    6: "g_disc",
    7: "unclassified",
}
G_MAX = 12
S_MAX = 16
SLOT_NAMES = [f"g{i}" for i in range(1, G_MAX + 1)] + ["i"] + [f"s{i}" for i in range(1, S_MAX + 1)] + ["f"]

DEFAULT_THETA = jnp.array(
    [-16.0, 230.0, -300.0, 500.0, 32.0, 60.0, -100.0, 300.0, -500.0, 0.16],
    dtype=jnp.float64,
)


def enable_persistent_jax_cache(path: str) -> None:
    """Enable JAX's persistent compilation cache before the first heavy solve.

    On Colab, pass a Google Drive directory so restarts can reuse compiled
    executables when JAX/XLA versions and shapes are compatible.
    """
    jax.config.update("jax_compilation_cache_dir", str(path))
    try:
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 2.0)
    except Exception:
        pass


# ============================================================
# BPS 1971 outer crust -- DIAGNOSTIC REFERENCE ONLY (v22 doc).
# The production crust is the CLDM cell from the table floor to n_cc.
# These rows exist to check the computed drip density and the Gamma1
# plateau, and for the comparison plot.  They never enter the EOS table.
# ============================================================
_BPS_RHO = np.array([
    1.044e4,2.622e4,6.587e4,1.654e5,4.156e5,1.044e6,2.622e6,6.588e6,8.293e6,
    1.655e7,3.302e7,6.589e7,1.315e8,2.624e8,3.304e8,5.237e8,8.301e8,1.045e9,
    1.316e9,1.657e9,2.626e9,4.164e9,6.601e9,8.312e9,1.046e10,1.318e10,
    1.659e10,2.090e10,2.631e10,3.313e10,4.172e10,5.254e10,6.617e10,8.332e10,
    1.049e11,1.322e11,1.664e11,1.844e11,2.096e11,2.640e11,3.325e11,4.188e11,
    4.299e11,
], dtype=float)
_BPS_P_CGS = np.array([
    9.744e18,4.968e19,2.431e20,1.151e21,5.266e21,2.318e22,9.755e22,3.911e23,
    5.259e23,1.435e24,3.833e24,1.006e25,2.604e25,6.676e25,8.738e25,1.629e26,
    3.029e26,4.129e26,5.036e26,6.860e26,1.272e27,2.356e27,4.362e27,5.662e27,
    7.702e27,1.048e28,1.425e28,1.938e28,2.503e28,3.404e28,4.628e28,5.949e28,
    8.089e28,1.100e29,1.495e29,2.033e29,2.597e29,2.892e29,3.290e29,4.473e29,
    5.816e29,7.538e29,7.805e29,
], dtype=float)
_BPS_N_CM3 = np.array([
    6.295e27,1.581e28,3.972e28,9.976e28,2.506e29,6.294e29,1.581e30,3.972e30,
    5.000e30,9.976e30,1.990e31,3.972e31,7.924e31,1.581e32,1.990e32,3.155e32,
    5.000e32,6.294e32,7.924e32,9.976e32,1.581e33,2.506e33,3.972e33,5.000e33,
    6.294e33,7.924e33,9.976e33,1.256e34,1.581e34,1.990e34,2.506e34,3.155e34,
    3.972e34,5.000e34,6.294e34,7.924e34,9.976e34,1.105e35,1.256e35,1.581e35,
    1.990e35,2.506e35,2.572e35,
], dtype=float)
_BPS_GAMMA_EQ = np.array([
    1.796,1.744,1.706,1.670,1.631,1.586,1.534,1.482,1.471,1.437,1.408,1.386,
    1.369,1.357,1.355,1.350,1.346,1.344,1.343,1.342,1.340,1.338,1.337,1.336,
    1.336,1.336,1.335,1.335,1.335,1.335,1.334,1.334,1.334,1.334,1.334,1.334,
    1.334,1.334,1.334,1.334,1.334,1.334,1.334,
], dtype=float)

BPS_DRIP_FM3 = float(_BPS_N_CM3[-1] * 1.0e-39)  # 2.572e-4; 1971 reference value


def _bps_reference() -> Dict[str, np.ndarray]:
    """Published BPS rows in MeV/fm^3 units, for the diagnostic overlay."""
    return dict(
        n=_BPS_N_CM3 * 1.0e-39,
        P=_BPS_P_CGS / MEVFM3_TO_ERGCM3,
        eps=_BPS_RHO * C_CGS**2 / MEVFM3_TO_ERGCM3,
        Geq=_BPS_GAMMA_EQ.copy(),
    )


# ============================================================
# Exact 2026 inner-core families
# ============================================================
def validate_inner_params(family: int, inner: np.ndarray, nt1: float) -> np.ndarray:
    p = np.asarray(inner, dtype=float).ravel()
    if not (0.24 < nt1 < 0.80):
        raise ValueError("nt1 must lie inside the source 0.24--0.80 fm^-3 range")
    if family == 1:
        if p.size != 3: raise ValueError("F1 inner = [dn, c2pt, dc2pt]")
        if not (0.0 < p[0] < 0.48 and 0.0 < p[1] < 1.0 and 1e-2 < p[2] < 1e4):
            raise ValueError("F1 outside Neill-2026 source bounds")
    elif family == 2:
        if p.size != 3: raise ValueError("F2 inner = [gamma1, gamma2, break1]")
        if not (0.0 < p[0] < 5.0 and 0.0 < p[1] < 5.0 and nt1 < p[2] < 0.80):
            raise ValueError("F2 outside source bounds/order")
    elif family == 3:
        if p.size != 5: raise ValueError("F3 inner = [gamma1,gamma2,gamma3,break1,break2]")
        if not (np.all((p[:3] > 0) & (p[:3] < 5)) and nt1 < p[3] < p[4] < 0.80):
            raise ValueError("F3 outside source bounds/order")
    elif family == 4:
        if p.size != 6: raise ValueError("F4 inner = [k1,k2,k3,y1,y2,y3]")
        if not (nt1 < p[0] < p[1] < p[2] < 0.80 and np.all((p[3:] > 0) & (p[3:] < 1))):
            raise ValueError("F4 outside source bounds/order")
    elif family == 5:
        # v22 doc: value-only seam continuity; c1 is solved, c2 is SAMPLED.
        if p.size != 6: raise ValueError("F5 inner = [nBL,nP,wP,hP,sP,c2]; c1 is solved from value continuity")
        nBL,nP,wP,hP,sP,c2 = p
        if not (0.01 < nBL < 3.20 and nt1 < nP < 3.20 and 0.08 < wP < 3.20 and 0 < hP < 1 and -50 < sP < 50):
            raise ValueError("F5 outside Neill-2026 source bounds")
        if not (0.01 < c2 < 3.20):
            raise ValueError("F5 deficit centre c2 outside operational 0.01--3.20 fm^-3 range")
        if not (nt1 < 0.48):
            raise ValueError("F5 source prior uses nt1 < 0.48 fm^-3")
    else:
        raise ValueError("family must be 1..5")
    return p


def sample_inner_source(family: int, nt1: float, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Draw one source-prior inner-core parameter vector."""
    rng = np.random.default_rng() if rng is None else rng
    if family == 1:
        return np.array([rng.uniform(1e-6,0.48), rng.uniform(1e-6,1-1e-6), 10**rng.uniform(-2,4)])
    if family == 2:
        return np.array([rng.uniform(1e-4,5), rng.uniform(1e-4,5), rng.uniform(nt1+1e-4,0.80)])
    if family == 3:
        br=np.sort(rng.uniform(nt1+1e-4,0.80,2)); return np.r_[rng.uniform(1e-4,5,3),br]
    if family == 4:
        k=np.sort(rng.uniform(nt1+1e-4,0.80,3)); return np.r_[k,rng.uniform(1e-4,1-1e-4,3)]
    if family == 5:
        if nt1 >= 0.48: raise ValueError("F5 requires nt1<0.48 in source prior")
        return np.array([rng.uniform(.010001,3.2),rng.uniform(nt1+1e-4,3.2),rng.uniform(.080001,3.2),rng.uniform(1e-4,1-1e-4),rng.uniform(-49.999,49.999),rng.uniform(.010001,3.2)])
    raise ValueError("family must be 1..5")


def _peak_term(n, nP, wP, hP, sP):
    x=(n-nP)/wP
    return hP*jnp.exp(-x*x)*(1.0+jsp.special.erf(sP*x))


def _solve_f5_value(nt1, cs21, pars):
    """Value-only seam continuity (v22 doc): solve the deficit depth c1 so that
    cs2(nt1) equals the outer-core seam value, with the sampled centre c2.
    The slope is deliberately NOT matched; the doc records that departure."""
    nBL,nP,wP,hP,sP,c2 = [jnp.asarray(x,dtype=jnp.float64) for x in pars]
    p0=_peak_term(nt1,nP,wP,hP,sP)
    A=1.0/3.0+p0-cs21
    c1=A*jnp.exp(((nt1-c2)/nBL)**2)
    return c1


def _inner_raw_parity(family,n,e,P,p,cs21,nt1):
    if family==1:
        npt=nt1+p[0]
        return jnp.where(n<npt,0.0,p[1]+(n-npt)*p[2])
    if family==2:
        return jnp.where(n<=p[2],p[0],p[1])*P/jnp.maximum(e,1e-30)
    if family==3:
        gam=jnp.where(n<=p[3],p[0],jnp.where(n<=p[4],p[1],p[2]))
        return gam*P/jnp.maximum(e,1e-30)
    if family==4:
        k1,k2,k3,y1,y2,y3=p[:6]
        return jnp.where(n<k1,cs21+(y1-cs21)*(n-nt1)/(k1-nt1),
               jnp.where(n<k2,y1+(y2-y1)*(n-k1)/(k2-k1),
               jnp.where(n<k3,y2+(y3-y2)*(n-k2)/(k3-k2),y3)))
    nBL,nP,wP,hP,sP,c1,c2=p[:7]
    return 1.0/3.0-c1*jnp.exp(-((n-c2)/nBL)**2)+_peak_term(n,nP,wP,hP,sP)


def _inner_kernel_parity(n_grid,mu1,P1,family,p,cs21,nt1):
    h=jnp.diff(n_grid)
    def step(carry,x):
        mu,P=carry; n,hn=x
        def deriv(mu0,P0,n0):
            e0=n0*mu0-P0
            raw=_inner_raw_parity(family,n0,e0,P0,p,cs21,nt1)
            c=jnp.minimum(raw,1.0)
            return jnp.array([c*mu0/n0,c*mu0])
        k1=deriv(mu,P,n)
        k2=deriv(mu+hn*k1[0]/2,P+hn*k1[1]/2,n+hn/2)
        k3=deriv(mu+hn*k2[0]/2,P+hn*k2[1]/2,n+hn/2)
        k4=deriv(mu+hn*k3[0],P+hn*k3[1],n+hn)
        z=jnp.array([mu,P])+hn*(k1+2*k2+2*k3+k4)/6
        return (z[0],z[1]),z
    (_, _),out=jax.lax.scan(step,(mu1,P1),(n_grid[:-1],h))
    mu=jnp.concatenate((mu1[None],out[:,0])); P=jnp.concatenate((P1[None],out[:,1]))
    e=n_grid*mu-P
    raw=jax.vmap(lambda nn,ee,pp:_inner_raw_parity(family,nn,ee,pp,p,cs21,nt1))(n_grid,e,P)
    cs2=jnp.minimum(raw,1.0)
    G=(e+P)*cs2/jnp.maximum(P,1e-30)
    return e,P,G,cs2,raw,mu

inner_kernel_parity=jax.jit(_inner_kernel_parity,static_argnums=(3,))


def _prepare_inner_params(family:int, inner:np.ndarray, nt1:float, cs21:float) -> jnp.ndarray:
    p=validate_inner_params(family,inner,nt1)
    if family==1: return jnp.array([p[0],p[1],p[2],0,0,0,0,0],dtype=jnp.float64)
    if family==2: return jnp.array([p[0],p[1],p[2],0,0,0,0,0],dtype=jnp.float64)
    if family==3: return jnp.array([p[0],p[1],p[2],p[3],p[4],0,0,0],dtype=jnp.float64)
    if family==4: return jnp.array([*p,0,0],dtype=jnp.float64)
    c2=p[5]
    c1=_solve_f5_value(jnp.asarray(nt1),jnp.asarray(cs21),p)
    if not np.isfinite(float(c1)): raise ValueError("F5 value-continuity solve is singular")
    return jnp.array([p[0],p[1],p[2],p[3],p[4],float(c1),c2,0],dtype=jnp.float64)


def _make_inner_parity(family,inner,nt1,nmax,mu1,P1,cs21):
    p=_prepare_inner_params(family,inner,nt1,cs21)
    base=jnp.linspace(nt1,nmax,N_INNER)
    special=[]
    if family==1: special=[nt1+float(inner[0])]
    elif family==2: special=[float(inner[2])]
    elif family==3: special=[float(inner[3]),float(inner[4])]
    elif family==4: special=list(np.asarray(inner[:3],float))
    grid=jnp.sort(jnp.concatenate((base,jnp.asarray(special,dtype=jnp.float64)))) if special else base
    e,P,G,cs2,raw,mu=inner_kernel_parity(grid,jnp.asarray(mu1),jnp.asarray(P1),family,p,jnp.asarray(cs21),jnp.asarray(nt1))
    a=np.asarray(raw)
    # v22 doc: the only F5 shape rejection is a negative squared sound speed;
    # causality is patched (min against 1) rather than rejected.
    if not np.all(np.isfinite(a)) or np.nanmin(a)<-1e-8:
        raise ValueError("inner-core sound speed squared became negative/non-finite")
    causal=np.asarray(raw)>=1.0
    nac=float(np.asarray(grid)[np.argmax(causal)]) if np.any(causal) else np.nan
    table=np.vstack([np.asarray(grid),np.asarray(e),np.asarray(P),np.asarray(G),np.asarray(G),np.asarray(cs2),np.zeros_like(np.asarray(grid))])
    return table,nac,np.asarray(p)


# ============================================================
# Production crust: one CLDM cell from the table floor to n_cc (v22 doc)
# ============================================================
def _solve_cldm_sequence(theta10: np.ndarray):
    """Solve the full production crust on compiled fixed grids.

    Outer crust (no gas) from OUTER_N_MIN up to neutron drip, located by the
    cell's own condition d eps_cell/dn = m_n; inner crust (with gas) from drip
    to the pasta onset u=1/8; tangent-extrapolated n_cc (production rule) with
    the marched true crossing kept as a diagnostic; pasta-band shear taper.
    """
    theta=jnp.asarray(theta10,dtype=jnp.float64)
    c_snm,c_sym,n_s=_theta_coeffs(theta)
    qguess=jnp.array([jnp.log(n_s),jnp.log(.30/.70),jnp.log(6.0),0.0],dtype=jnp.float64)

    # ---- neutron drip from the WITH-gas branch (energy criterion) ----
    # "The composition is whatever is cheapest" (v22 doc): below drip the
    # 4-coordinate solve empties its gas (n_g collapses toward the coordinate
    # floor); above drip the converged gas density is real and grows with the
    # BBP square-root cusp.  Drip is the first density where the with-gas cell
    # keeps a genuine gas, detected as n_g > 1e-4 n sustained over three grid
    # points.  The derivative criterion d eps_cell/dn = m_n (eq. drip of the
    # doc) is kept as a diagnostic below; the two coincide whenever the no-gas
    # branch keeps its interior minimum up to drip, and the energy route also
    # covers draws where that branch destabilises exactly at drip.
    ndrip=np.nan
    n_lo_try=1.2e-4
    for _ in range(2):
        dgrid=np.geomspace(n_lo_try,OUTER_N_MAX,220)
        q0o,e0o,P0o,u0o,g0o,ok0o,k0o=jax_crust_point_kernel(jnp.asarray(dgrid[0]),qguess,jnp.asarray(False),c_snm,c_sym,n_s)
        if not bool(ok0o): raise ValueError("outer-crust CLDM failed to converge below drip")
        qseed0=jnp.asarray(np.asarray(q0o)).at[3].set(jnp.log(dgrid[0]*1e-6))
        qd,ed,Pd,ud,gd,dripd,diffd,goodd=jax_crust_march_kernel(jnp.asarray(dgrid),qseed0,jnp.asarray(True),c_snm,c_sym,n_s)
        jax.block_until_ready(ed)
        qd=np.asarray(qd); goodd=np.asarray(goodd)
        ngfrac=np.exp(qd[:,3])/dgrid
        dripped=goodd&(ngfrac>1e-4)
        run3=dripped[:-2]&dripped[1:-1]&dripped[2:]
        hits=np.where(run3)[0]
        if hits.size and hits[0]==0 and n_lo_try>7e-5:
            n_lo_try=6e-5; continue
        if hits.size:
            k=int(hits[0])
            lo=dgrid[max(k-1,0)]; hi=dgrid[k]
            nd_j,q_j,ok_j=jax_dripE_bisect_kernel(jnp.asarray(lo),jnp.asarray(hi),jnp.asarray(qd[max(k-1,0)]),c_snm,c_sym,n_s)
            ndrip=float(nd_j)
            q_at_drip=np.asarray(q_j)
        break
    if not np.isfinite(ndrip):
        raise ValueError("no neutron drip below the outer-crust search ceiling (named rejection)")
    if ndrip<=OUTER_N_MIN*3:
        raise ValueError("neutron drip at the table floor; unphysical draw")

    # ---- outer crust march below the detected drip ----
    n_out=np.geomspace(OUTER_N_MIN,ndrip*(1-1e-6),N_OUTER_CRUST)
    qo,eo,Po,uo,go,dripo,diffo,goodo=jax_crust_march_kernel(jnp.asarray(n_out),qguess,jnp.asarray(False),c_snm,c_sym,n_s)
    jax.block_until_ready(eo)
    qo=np.asarray(qo); dripo=np.asarray(dripo); goodo=np.asarray(goodo)
    if not bool(goodo[0]): raise ValueError("outer-crust CLDM failed to converge at the table floor")
    keep_out=np.where(goodo)[0]
    if keep_out.size<24: raise ValueError("insufficient converged outer-crust points")
    n_o=n_out[keep_out]; q_o=qo[keep_out]
    # diagnostic: the derivative-criterion drip (first Lambda crossing), NaN if absent
    dcross=np.where(goodo&(dripo>=0))[0]
    ndrip_lambda=float(n_out[dcross[0]]) if dcross.size else np.nan

    # ---- inner crust march, pasta onset, tangent n_cc ----
    ngrid=np.geomspace(ndrip*(1+1e-5),0.13,N_CLDM)
    qseed=jnp.asarray(q_at_drip)
    q0,e0,P0,u0,g0,ok0,k0=jax_crust_point_kernel(jnp.asarray(ngrid[0]),qseed,jnp.asarray(True),c_snm,c_sym,n_s)
    if not bool(ok0): raise ValueError("could not seed inner-crust CLDM branch")
    qi,ei,Pi,ui,gi,dripi,diffi,goodi=jax_crust_march_kernel(jnp.asarray(ngrid),q0,jnp.asarray(True),c_snm,c_sym,n_s)
    jax.block_until_ready(ei)
    qs=np.asarray(qi); good=np.asarray(goodi); u_scan=np.asarray(ui); diffs=np.asarray(diffi)
    cross=np.where(good&(u_scan>=1/8))[0]
    if cross.size==0: raise ValueError("CLDM never reaches u=1/8")
    ihi=int(cross[0]); ilo=max(0,ihi-1)
    n18,q18,ok18=jax_u18_bisect_kernel(jnp.asarray(ngrid[ilo]),jnp.asarray(ngrid[ihi]),jnp.asarray(qs[ilo]),c_snm,c_sym,n_s)
    if not bool(ok18): raise ValueError("u=1/8 continuation failed")
    n18=float(n18); q18=np.asarray(q18)
    c18=np.asarray(crust_props_point_kernel(jnp.asarray(n18),jnp.asarray(q18),jnp.asarray(True),c_snm,c_sym,n_s))
    core18=np.asarray(_core_point_fixed(jnp.asarray(n18),c_snm,c_sym,n_s))
    ecell,pcell=c18[0]/n18,c18[1]; eunif,punif=core18[0]/n18,core18[1]
    denom=punif-pcell
    if abs(denom)<1e-12: raise ValueError("tangent n_cc denominator is singular")
    # Sign convention (v22 doc): below the crossing the clusters are CHEAPER
    # (ecell<eunif) but carry the LARGER pressure, so numerator and denominator
    # are both negative and the extrapolated shift is positive.  The named
    # rejection is a non-positive shift, not a negative denominator.
    shift=n18*n18*(ecell-eunif)/denom
    if shift<=0: raise ValueError("tangent n_cc has no forward crossing (named rejection)")
    ncc=n18+shift
    if not (n18<ncc<0.16): raise ValueError(f"unphysical tangent n_cc={ncc:.6g}")
    # Marched true-crossing diagnostic (v22 doc): first density at which the
    # clustered energy per baryon meets the uniform one past the pasta onset.
    mc=np.where(good&(u_scan>=1/8)&(diffs>=0))[0]
    if mc.size==0:
        # The branch can rail (u->1, rN bound) before the energies formally
        # cross; the first non-negative energy difference still marks the
        # marched-crossing diagnostic even on such a row.
        mc=np.where(np.isfinite(diffs)&(u_scan>=1/8)&(diffs>=0))[0]
    ncc_marched=float(ngrid[mc[0]]) if mc.size else float('nan')

    # ---- assemble crust rows: outer + drip + inner(<ncc) + exact ncc ----
    idx=np.where((ngrid<ncc)&good)[0]
    if len(idx)<12: raise ValueError("insufficient valid CLDM points below n_cc")
    nsel=ngrid[idx]; qsel=qs[idx]
    qcc,ecc,Pcc,ucc,gcc,okcc,kcc=jax_crust_point_kernel(jnp.asarray(ncc),jnp.asarray(qsel[-1]),jnp.asarray(True),c_snm,c_sym,n_s)
    if not bool(okcc): raise ValueError("CLDM continuation could not evaluate extrapolated n_cc")
    nfinal=np.r_[n_o,nsel,ncc]
    qfinal=np.vstack([q_o,qsel,np.asarray(qcc)])
    inner_flags=np.r_[np.zeros(len(n_o),dtype=bool),np.ones(len(nsel)+1,dtype=bool)]
    props=np.asarray(crust_props_kernel(jnp.asarray(nfinal),jnp.asarray(qfinal),jnp.asarray(inner_flags),jnp.ones(len(nfinal),dtype=bool),c_snm,c_sym,n_s))
    Geq,Simp,Aimp=implicit_crust_kernel(jnp.asarray(nfinal),jnp.asarray(qfinal),jnp.asarray(inner_flags),c_snm,c_sym,n_s)
    jax.block_until_ready(Geq)
    Geq=np.asarray(Geq); Simp=np.asarray(Simp)
    eps,P,G1,u,muc,A,Z,Xn,drip=props.T
    cs2=Geq*P/np.maximum(eps+P,1e-30)

    # ---- pasta shear taper: value/slope match at n18, double root at ncc ----
    def solve_point(nn, qseed0):
        qj,e_,P_,u_,g_,ok_,k_=jax_crust_point_kernel(jnp.asarray(nn),jnp.asarray(qseed0),jnp.asarray(True),c_snm,c_sym,n_s)
        pr=np.asarray(crust_props_point_kernel(jnp.asarray(nn),qj,jnp.asarray(True),c_snm,c_sym,n_s))
        return np.asarray(qj),pr,bool(ok_)
    h=1e-3*n18
    _,p0,_=solve_point(n18,q18); _,pp,_=solve_point(n18+h,q18); _,pm,_=solve_point(n18-h,q18)
    m1,A1,Z1,X1=p0[4],p0[5],p0[6],p0[7]
    Ap=(pp[5]-pm[5])/(2*h); Zp=(pp[6]-pm[6])/(2*h); Xp=(pp[7]-pm[7])/(2*h)
    m1p=m1*(4/(3*n18)-4*Ap/(3*A1)-4*Xp/(3*(1-X1))+2*Zp/Z1)
    d=n18-ncc; ss=m1*d/(m1p*d-2*m1)
    if np.isfinite(ss) and ss>0:
        c2=n18-ss; c1=m1/(d*d*ss); pasta_mu=c1*(nfinal-ncc)**2*(nfinal-c2)
    else:
        pasta_mu=m1*((nfinal-ncc)/d)**2
    mu=np.where(nfinal>=n18,pasta_mu,muc); mu=np.where(nfinal>=ncc,0.0,mu)
    return dict(n=nfinal,eps=eps,P=P,G1=G1,Geq=Geq,cs2=cs2,mu=mu,u=u,A=A,Z=Z,Xn=Xn,q=qfinal,
                inner=inner_flags,ndrip=ndrip,ndrip_lambda=ndrip_lambda,n18=n18,ncc=float(ncc),
                ncc_marched=ncc_marched,q18=q18,
                tangent=(ecell,eunif,pcell,punif),stationarity_max=float(np.nanmax(Simp)))


def _continue_to_vacuum(cldm: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Continue the crust table below its floor toward P=0.

    The v22 doc: the table floor is 1e-5 fm^-3 and the final interval is
    continued to vacuum; it carries negligible mass and exists only so the
    stellar radius converges.  A local power law anchored on the lowest rows
    supplies the continuation; the composition there is frozen at the floor.
    """
    n=np.asarray(cldm['n']); P=np.asarray(cldm['P']); eps=np.asarray(cldm['eps'])
    n0,P0,e0=float(n[0]),float(P[0]),float(eps[0])
    g0=float(np.clip((np.log(P[2])-np.log(P[0]))/(np.log(n[2])-np.log(n[0])),1.05,3.0))
    n_low=np.geomspace(n0*1.0e-16,n0,96,endpoint=False)
    K=P0/n0**g0
    P_low=K*n_low**g0
    eper0=e0/n0
    eper_low=eper0+K/(g0-1.0)*(n_low**(g0-1.0)-n0**(g0-1.0))
    eps_low=n_low*eper_low
    out={}
    out['n']=np.r_[n_low,n]; out['P']=np.r_[P_low,P]; out['eps']=np.r_[eps_low,eps]
    out['G1']=np.r_[np.full_like(n_low,float(cldm['G1'][0])),np.asarray(cldm['G1'])]
    out['Geq']=np.r_[np.full_like(n_low,g0),np.asarray(cldm['Geq'])]
    out['mu']=np.r_[float(cldm['mu'][0])*(n_low/n0)**(4.0/3.0),np.asarray(cldm['mu'])]
    out['u']=np.r_[np.zeros_like(n_low),np.asarray(cldm['u'])]
    out['cs2']=out['Geq']*out['P']/np.maximum(out['eps']+out['P'],1e-300)
    return out


def _enforce_monotone_pressure(crust: Dict[str, np.ndarray], depth_cap: float = 1.5e-1) -> Dict[str, np.ndarray]:
    """Bridge non-monotone stretches of the crust pressure; reject deep dips.

    Two things produce a locally decreasing tabulated P.  Seam/solver noise
    (parts in 1e4) at branch changes, and, more importantly, the extended
    spherical cell losing mechanical stability inside the pasta band, where
    the v22 doc carries spherical values and bridges rather than solves.  Both
    are handled the same way: rows sitting below the running pressure maximum
    are dropped, which leaves a flat bridge in P across the dip (the doc
    explicitly allows dP/dn = 0 stretches, never dP/dn < 0), and the maximum
    relative depth of what was bridged is recorded as P_bridge_depth_max.  A
    dip deeper than depth_cap is no longer bookkeeping and rejects the draw by
    name (the stability guard).
    """
    P=np.asarray(crust['P'],float); nrow=len(P)
    keep=np.ones(nrow,bool)
    run=P[0]; depth=0.0
    for k in range(1,nrow):
        if P[k]<=run:
            depth=max(depth,(run-P[k])/max(run,1e-300))
            keep[k]=False
        else:
            run=P[k]
    if depth>depth_cap:
        raise ValueError(f"crust pressure dips by {depth:.3e} below its running maximum (stability guard)")
    out={}
    for key,val in crust.items():
        arr=np.asarray(val)
        out[key]=arr[keep] if (arr.ndim>=1 and arr.shape[0]==nrow) else val
    out['P_bridge_depth_max']=float(depth)
    return out


def parity_checks(cldm: Dict[str, np.ndarray], crust: Dict[str, np.ndarray]) -> Dict[str, float]:
    """The v22-doc validation numbers, computed on the production crust."""
    checks={}
    checks['ndrip_fm3']=float(cldm['ndrip'])
    checks['ndrip_lambda_fm3']=float(cldm.get('ndrip_lambda',np.nan))
    checks['ndrip_in_band']=bool(2.0e-4<=cldm['ndrip']<=3.5e-4)
    checks['bps_drip_reference_fm3']=BPS_DRIP_FM3
    checks['P_bridge_depth_max']=float(crust.get('P_bridge_depth_max',0.0))
    g1=float(np.interp(1.0e-4,np.asarray(crust['n']),np.asarray(crust['G1'])))
    checks['G1_at_1e-4_fm3']=g1
    checks['G1_plateau_ok']=bool(abs(g1-1.334)<0.01)
    checks['ncc_extrapolated_fm3']=float(cldm['ncc'])
    checks['ncc_marched_fm3']=float(cldm['ncc_marched'])
    checks['ncc_order_ok']=bool((not np.isfinite(cldm['ncc_marched'])) or (cldm['ncc']<cldm['ncc_marched']))
    return checks


def build_eos(theta10=DEFAULT_THETA, family:int=1, inner=None, nt1:float=0.50, nmax:float=3.2):
    """Construct one parity EOS to a requested high-density ceiling.

    nmax is a numerical table ceiling, not a model prior.  forward_model()
    extends it automatically until Mmax is resolved.
    """
    theta=np.asarray(theta10,dtype=float)
    if theta.size!=10: raise ValueError("theta10 must have 10 entries")
    if inner is None: inner=sample_inner_source(family,nt1,np.random.default_rng(1234))
    inner=np.asarray(inner,dtype=float); validate_inner_params(family,inner,nt1)
    if nt1<=0: raise ValueError("nt1 must be positive")

    # Cheap guard first (v22 doc guard 1): a non-positive symmetry energy in
    # the dynamically relevant band makes the closure multivalued and the crust
    # unbuildable, so reject before the expensive cell solve.  Below 1e-3 fm^-3
    # the damped expansion sends every term to zero and the sign of the
    # residual is numerically meaningless; the drip/tangent rejections cover
    # any pathology that band could hide.
    cs0,cv0,ns0=_theta_coeffs(jnp.asarray(theta))
    nchk=np.geomspace(1e-3,nt1,200)
    sym0=np.asarray(jax.vmap(lambda nn:_imp_channel(nn,cv0,ns0)[0])(jnp.asarray(nchk)))
    if np.nanmin(sym0)<=0:
        raise ValueError("symmetry energy non-positive in tabulated nucleonic range (named rejection)")

    cldm=_solve_cldm_sequence(theta)
    crust=_continue_to_vacuum(cldm)
    crust=_enforce_monotone_pressure(crust)
    if not np.all(np.diff(crust['P'])>0):
        raise ValueError("assembled crust pressure is not strictly increasing")
    parity=parity_checks(cldm,crust)

    cs,cv,ns=_theta_coeffs(jnp.asarray(theta))
    if not (cldm['ncc']<nt1): raise ValueError("nt1 must exceed tangent crust-core transition")
    ncore=np.linspace(cldm['ncc'],nt1,N_OUTER_CORE)
    oc=np.asarray(core_kernel(jnp.asarray(ncore),cs,cv,ns))
    # core_kernel columns: eps,P,G1,Geq,cs2,mu
    outer_core=np.vstack([ncore,oc[:,0],oc[:,1],oc[:,2],oc[:,3],oc[:,4],oc[:,5]])
    e1,P1,G11,Geq1,cs21,_=oc[-1]; mu1=(e1+P1)/nt1
    inner_table,nac,packed=_make_inner_parity(family,inner,nt1,nmax,mu1,P1,cs21)

    if not np.all(np.isfinite(inner_table)) or np.nanmin(inner_table[5])<0:
        raise ValueError("invalid inner-core table")

    return dict(
        theta=theta,family=int(family),inner=inner,inner_packed=packed,nt1=float(nt1),nmax=float(nmax),
        bps=_bps_reference(),cldm=cldm,crust=crust,outer_core=outer_core,inner_core=inner_table,
        transitions=np.array([cldm['ndrip'],cldm['n18'],cldm['ncc'],nt1]),n_ac=float(nac),
        parity=parity,
    )


# ============================================================
# Pressure-domain material interpolation and TOV
# ============================================================
def _strict_branch(n,eps,P,G1,Geq,mu):
    n=np.asarray(n,float); eps=np.asarray(eps,float); P=np.asarray(P,float)
    G1=np.asarray(G1,float); Geq=np.asarray(Geq,float); mu=np.asarray(mu,float)
    finite=np.isfinite(n)&np.isfinite(eps)&np.isfinite(P)&np.isfinite(G1)&np.isfinite(Geq)&np.isfinite(mu)
    n,eps,P,G1,Geq,mu=[x[finite] for x in (n,eps,P,G1,Geq,mu)]
    order=np.argsort(P); n,eps,P,G1,Geq,mu=[x[order] for x in (n,eps,P,G1,Geq,mu)]
    keep=np.r_[True,np.diff(P)>max(1e-16,1e-12*np.nanmax(P))]
    return tuple(jnp.asarray(x) for x in (P[keep]*MEVFM3_TO_GEOM,n[keep],eps[keep]*MEVFM3_TO_GEOM,G1[keep],Geq[keep],mu[keep]*MEVFM3_TO_GEOM))


def build_tov_tables(eos):
    cr=eos['crust']; oc=eos['outer_core']; ic=eos['inner_core']; fam=eos['family']; nt1=eos['nt1']
    bcr=_strict_branch(cr['n'],cr['eps'],cr['P'],cr['G1'],cr['Geq'],cr['mu'])
    boc=_strict_branch(oc[0],oc[1],oc[2],oc[3],oc[4],oc[6])
    if fam==1:
        npt=nt1+eos['inner'][0]
        k=np.searchsorted(ic[0],npt,side='left')
        bic=_strict_branch(ic[0][k:],ic[1][k:],ic[2][k:],ic[3][k:],ic[4][k:],ic[6][k:])
        # Low/high energy density at the same phase pressure.
        eps_lo=float(oc[1,-1])*MEVFM3_TO_GEOM
        eps_hi=float(np.interp(npt,ic[0],ic[1]))*MEVFM3_TO_GEOM
    else:
        bic=_strict_branch(ic[0],ic[1],ic[2],ic[3],ic[4],ic[6]); eps_lo=eps_hi=np.nan
    pcc=float(oc[2,0])*MEVFM3_TO_GEOM; pt1=float(oc[2,-1])*MEVFM3_TO_GEOM
    ps=float(bcr[0][0]); pmax=float(bic[0][-1])
    return dict(crust=bcr,outer=boc,inner=bic,pcc=pcc,pt1=pt1,ps=ps,pmax=pmax,family=fam,eps_phase_low=eps_lo,eps_phase_high=eps_hi)


def _interp_log_jax(x,xp,fp):
    x=jnp.clip(x,xp[0],xp[-1]); i=jnp.clip(jnp.searchsorted(jax.lax.stop_gradient(xp),jax.lax.stop_gradient(x),side='right')-1,0,xp.shape[0]-2)
    lx=jnp.log(jnp.maximum(x,TOV_TINY)); l0=jnp.log(jnp.maximum(xp[i],TOV_TINY)); l1=jnp.log(jnp.maximum(xp[i+1],TOV_TINY))
    t=(lx-l0)/jnp.maximum(l1-l0,TOV_TINY)
    return jnp.exp(jnp.log(jnp.maximum(fp[i],TOV_TINY))+t*(jnp.log(jnp.maximum(fp[i+1],TOV_TINY))-jnp.log(jnp.maximum(fp[i],TOV_TINY))))


def _interp_linear_jax(x,xp,fp):
    x=jnp.clip(x,xp[0],xp[-1]); i=jnp.clip(jnp.searchsorted(jax.lax.stop_gradient(xp),jax.lax.stop_gradient(x),side='right')-1,0,xp.shape[0]-2)
    t=(x-xp[i])/jnp.maximum(xp[i+1]-xp[i],TOV_TINY); return fp[i]+t*(fp[i+1]-fp[i])


def _region_material(P,branch):
    p,n,e,g1,ge,mu=branch
    return jnp.array([_interp_log_jax(P,p,n),_interp_log_jax(P,p,e),_interp_linear_jax(P,p,g1),_interp_linear_jax(P,p,ge),_interp_log_jax(P,p,jnp.maximum(mu,TOV_TINY))])


def _material(P,t,force_inner=False):
    P=jnp.clip(P,t['ps'],t['pmax'])
    if force_inner:
        v=_region_material(jnp.maximum(P,t['pt1']),t['inner'])
    else:
        v=jax.lax.cond(P<=t['pcc'],lambda _:_region_material(P,t['crust']),lambda _:jax.lax.cond(P<=t['pt1'],lambda __:_region_material(P,t['outer']),lambda __:_region_material(P,t['inner']),None),None)
    n,e,g1,ge,mu=v
    g1=jnp.maximum(g1,1e-10); ge=jnp.maximum(ge,1e-10)
    cad2=jnp.clip(g1*P/jnp.maximum(e+P,TOV_TINY),1e-12,.999999999)
    ceq2=jnp.clip(ge*P/jnp.maximum(e+P,TOV_TINY),1e-12,.999999999)
    return n,e,g1,ge,mu,cad2,ceq2


def _tov_rhs(r,x,t,force_inner=False):
    m,P,nu=x; e=_material(jnp.maximum(P,t['ps']),t,force_inner)[1]
    den=jnp.maximum(r*(r-2*m),1e-40); common=(m+4*jnp.pi*r**3*P)/den
    return jnp.array([4*jnp.pi*r**2*e,-(e+P)*common,common])


def _tov_rk4_raw(r,x,h,t,force_inner=False):
    k1=_tov_rhs(r,x,t,force_inner); k2=_tov_rhs(r+h/2,x+h*k1/2,t,force_inner); k3=_tov_rhs(r+h/2,x+h*k2/2,t,force_inner); k4=_tov_rhs(r+h,x+h*k3,t,force_inner)
    return x+h*(k1+2*k2+2*k3+k4)/6


def _tov_rk4(r,x,h,t):
    trial=_tov_rk4_raw(r,x,h,t,False); first=(t['family']==1); pt=t['pt1']; cross=first&(x[1]>pt)&(trial[1]<pt)
    def split(_):
        f=jnp.clip((x[1]-pt)/jnp.maximum(x[1]-trial[1],TOV_TINY),1e-8,1-1e-8); h1=h*f
        xm=_tov_rk4_raw(r,x,h1,t,True).at[1].set(pt)
        return _tov_rk4_raw(r+h1,xm,h-h1,t,False)
    return jax.lax.cond(cross,split,lambda _:trial,None)


def _tov_summary(Pc,t):
    ps=t['ps']; r0=jnp.array(1.0,dtype=jnp.float64); ec=_material(Pc,t)[1]; x0=jnp.array([4*jnp.pi*r0**3*ec/3,Pc,0.0])
    def step(c,_):
        r,x,done,ra,xa,rb,xb,ok=c
        def go(_):
            xn=_tov_rk4(r,x,TOV_DR,t); rn=r+TOV_DR
            good=(rn>2*xn[0])&jnp.all(jnp.isfinite(xn))&(xn[1]<=x[1]*(1+1e-12)); cross=(x[1]>ps)&(xn[1]<=ps)
            return rn,xn,cross|(~good),jnp.where(cross,r,ra),jnp.where(cross,x,xa),jnp.where(cross,rn,rb),jnp.where(cross,xn,xb),ok&good
        return jax.lax.cond(done,lambda _:c,go,None),None
    (r,x,done,ra,xa,rb,xb,ok),_=jax.lax.scan(step,(r0,x0,jnp.array(False),r0,x0,r0,x0,jnp.array(True)),None,length=N_TOV)
    d=xb[1]-xa[1]; f=jnp.clip((ps-xa[1])/jnp.where(jnp.abs(d)>TOV_TINY,d,-TOV_TINY),0,1)
    R=ra+f*(rb-ra); M=xa[0]+f*(xb[0]-xa[0]); good=ok&(xa[1]>ps)&(xb[1]<=ps)&(R>2*M)
    return R,M,good

_TOV_SWEEP_JIT=jax.jit(jax.vmap(_tov_summary,in_axes=(0,None)))


def resolve_mr(eos):
    t=build_tov_tables(eos)
    # Resolve the *whole* stable branch.  nt1 is an acceptance condition for
    # a particular sampled star, not the lower boundary of the M-R sequence.
    pmin=max(t['pcc']*1.001,t['ps']*10.0)
    pcs=jnp.geomspace(pmin,t['pmax']*0.995,N_MR)
    R,M,ok=_TOV_SWEEP_JIT(pcs,t); jax.block_until_ready(M)
    Rn=np.asarray(R); Mn=np.asarray(M); good=np.asarray(ok)&np.isfinite(np.asarray(M))
    if np.count_nonzero(good)<8: return dict(resolved=False,t=t,pcs=np.asarray(pcs),R=Rn,M=Mn,ok=good)
    ids=np.where(good)[0]; kg=ids[np.argmax(Mn[ids])]
    resolved=(kg<=ids[-1]-MMAX_MARGIN) and kg>ids[0]
    return dict(resolved=bool(resolved),t=t,pcs=np.asarray(pcs),R=Rn,M=Mn,ok=good,kmax=int(kg),Mmax=float(Mn[kg]*C_CGS**2/(G_CGS*MSUN_CGS)),Rmax=float(Rn[kg]/1e5))


def build_eos_and_resolve_mmax(theta10,family,inner,nt1,nmax0=3.2):
    nmax=max(float(nmax0),float(nt1)+0.6)
    for _ in range(5):
        eos=build_eos(theta10,family,inner,nt1,nmax)
        mr=resolve_mr(eos)
        if mr['resolved']: return eos,mr
        nmax=min(MMAX_MAX_N,nmax*MMAX_EXTEND_FACTOR)
        if nmax>=MMAX_MAX_N-1e-10: break
    raise ValueError("maximum-mass turnover not resolved before numerical emergency ceiling")


def invert_mass(mass_msun,eos,mr,require_inner_core=False):
    """Invert M(Pc)=target on the resolved stable branch.

    The root is bracketed directly in central pressure.  The coarse M-R grid
    is used only to locate the maximum-mass end of the stable branch, never as
    a hard lower mass boundary.
    """
    target=float(mass_msun)
    t=mr['t']
    kmax=int(mr['kmax'])
    pc_hi=float(mr['pcs'][kmax])
    Rhi,Mhi,okhi=_tov_summary(jnp.asarray(pc_hi),t)
    m_hi=float(Mhi*C_CGS**2/(G_CGS*MSUN_CGS))
    if (not bool(okhi)) or target > m_hi + MASS_TOL_MSUN:
        raise ValueError(f"target mass {target:.9f} Msun exceeds resolved stable maximum {m_hi:.9f} Msun")

    # Batched bracket search: one vmapped sweep finds a valid model below the
    # target, replacing the old sequential growth loop.
    probe=jnp.geomspace(max(float(t['pcc'])*1e-4,float(t['ps'])*10.0),pc_hi*(1-1e-10),24)
    Rp,Mp,okp=_TOV_SWEEP_JIT(probe,t)
    mp=np.asarray(Mp)*C_CGS**2/(G_CGS*MSUN_CGS); okn=np.asarray(okp)&np.isfinite(mp)
    below=np.where(okn&(mp<=target+MASS_TOL_MSUN))[0]
    if below.size==0:
        raise ValueError(f"could not construct a lower-pressure stable bracket below {target:.9f} Msun; lowest direct model was {np.nanmin(np.where(okn,mp,np.nan))}")
    lo=float(np.asarray(probe)[below[-1]]); hi=pc_hi
    # Batched bisection: each round evaluates a 16-point subdivision of the
    # bracket in one vmapped sweep; four rounds give a 16^-4 ~ 1.5e-5 bracket
    # reduction, then a short scalar polish reaches the mass tolerance.
    for _ in range(4):
        grid=jnp.linspace(lo,hi,16)
        Rg,Mg,okg=_TOV_SWEEP_JIT(grid,t)
        mg=np.asarray(Mg)*C_CGS**2/(G_CGS*MSUN_CGS); okgn=np.asarray(okg)&np.isfinite(mg)
        gl=np.where(okgn&(mg<target))[0]
        lo=float(np.asarray(grid)[gl[-1]]) if gl.size else lo
        gh=np.where((~okgn)|(mg>=target))[0]
        gh=gh[gh> (gl[-1] if gl.size else -1)]
        hi=float(np.asarray(grid)[gh[0]]) if gh.size else hi
    R=M=ok=None
    for _ in range(18):
        mid=0.5*(lo+hi)
        R,M,ok=_tov_summary(jnp.asarray(mid),t)
        if not bool(ok):
            hi=mid
            continue
        mm=float(M*C_CGS**2/(G_CGS*MSUN_CGS))
        if mm < target:
            lo=mid
        else:
            hi=mid
    Pc=0.5*(lo+hi)
    R,M,ok=_tov_summary(jnp.asarray(Pc),t)
    if not bool(ok):
        raise ValueError("target-mass TOV inversion ended on an invalid stellar model")
    solved=float(M*C_CGS**2/(G_CGS*MSUN_CGS))
    if abs(solved-target) > max(MASS_TOL_MSUN,2e-6):
        raise ValueError(f"target-mass inversion residual too large: target={target:.9f}, solved={solved:.9f} Msun")

    inner_core_entered = Pc >= float(t['pt1'])*(1-1e-10)
    if require_inner_core and not inner_core_entered:
        Rt,Mt,okt=_tov_summary(jnp.asarray(float(t['pt1'])*(1+1e-9)),t)
        mt=float(Mt*C_CGS**2/(G_CGS*MSUN_CGS)) if bool(okt) else np.nan
        raise ValueError(
            f"target star {target:.6f} Msun is stable but does not reach nt1; "
            f"nt1 is first reached at about {mt:.6f} Msun for this EOS"
        )
    return Pc,float(R),float(M),bool(inner_core_entered)


def _build_star_profile(eos,t,Pc):
    ps=t['ps']; r0=jnp.asarray(1.0); mc=_material(jnp.asarray(Pc),t); nc,ec=mc[0],mc[1]; m0=4*jnp.pi*r0**3*ec/3; state0=jnp.array([m0,Pc,0.0])
    def step(carry,_):
        r,state,done=carry; cand=_tov_rk4(r,state,TOV_DR,t); rn=r+TOV_DR
        finite=jnp.all(jnp.isfinite(cand)); metric=rn>2*cand[0]; monotone=cand[1]<=state[1]*(1+1e-12); crossed=(~done)&finite&metric&monotone&(cand[1]<=ps); failed=(~done)&((~finite)|(~metric)|(~monotone)); active=~done
        return (jnp.where(active,rn,r),jnp.where(active,cand,state),done|crossed|failed),(jnp.where(active,rn,r),jnp.where(active,cand,state),active&finite&metric&monotone,crossed)
    _,out=jax.lax.scan(step,(r0,state0,jnp.asarray(False)),None,length=N_TOV); rsteps,states,valid_steps,crossed=out
    if not bool(jnp.any(crossed)): raise ValueError("TOV surface not found")
    isurf=int(jnp.argmax(crossed.astype(jnp.int32))); ip=max(isurf-1,0)
    rb=float(rsteps[ip] if isurf>0 else r0); sb=np.asarray(states[ip] if isurf>0 else state0); ra=float(rsteps[isurf]); sa=np.asarray(states[isurf])
    fs=np.clip((ps-sb[1])/(sa[1]-sb[1]),0,1); rstar=rb+fs*(ra-rb); mstar=sb[0]+fs*(sa[0]-sb[0]); nustar0=sb[2]+fs*(sa[2]-sb[2]); nustar=.5*np.log(max(1-2*mstar/rstar,1e-300)); nushift=nustar-nustar0
    rr=np.r_[1.0,np.asarray(rsteps[:isurf+1])]; mm=np.r_[float(m0),np.asarray(states[:isurf+1,0])]; pp=np.r_[Pc,np.asarray(states[:isurf+1,1])]; nu=np.r_[0.0,np.asarray(states[:isurf+1,2])]+nushift
    rr[-1]=rstar; mm[-1]=mstar; pp[-1]=ps; nu[-1]=nustar
    # Material branch values.  P==pt1 belongs to low-density side by default.
    vals=np.asarray(jax.vmap(lambda p:jnp.asarray(_material(p,t)))(jnp.asarray(pp)))
    nprof,eps,g1,ge,mu,cad,ceq=vals.T
    lam=-.5*np.log(np.maximum(1-2*mm/np.maximum(rr,1e-300),1e-300))
    def radius_at_pressure(pt):
        ids=np.where((pp[:-1]>=pt)&(pp[1:]<pt))[0]
        if not len(ids): return np.nan
        i=ids[0]; f=np.clip((pt-pp[i])/(pp[i+1]-pp[i]),0,1); return rr[i]+f*(rr[i+1]-rr[i])
    rcc=radius_at_pressure(t['pcc']); rt1=radius_at_pressure(t['pt1'])
    if not np.isfinite(rcc): raise ValueError("crust-core radius not found")
    return dict(eos=eos,t=t,Pc=float(Pc),rstar=float(rstar),mstar=float(mstar),rcc=float(rcc),rt1=float(rt1),r=rr,m=mm,P=pp,eps=eps,n=nprof,mu=mu,G1=g1,Geq=ge,cad2=cad,ceq2=ceq,nu=nu,lam=lam,valid=True)


# ============================================================
# Relativistic-Cowling elastic oscillations
# ============================================================
def _mode_solver(star):
    rstar=jnp.asarray(star['rstar']); mstar=jnp.asarray(star['mstar']); rcc=jnp.asarray(star['rcc']); rt1=jnp.asarray(star['rt1'])
    rtab=jnp.asarray(star['r']); mtab=jnp.asarray(star['m']); ptab=jnp.asarray(star['P']); etab=jnp.asarray(star['eps']); nutab=jnp.asarray(star['nu']); lambdatab=jnp.asarray(star['lam']); gtab=jnp.asarray(star['G1']); ceqtab=jnp.asarray(star['ceq2']); mutab=jnp.asarray(star['mu'])
    fam=int(star['eos']['family'])
    # Static split guard: a family-1 star whose centre never reaches nt1 has no
    # interior phase boundary, and a NaN rt1 must never poison the radial grids.
    fam1_split=(fam==1) and bool(np.isfinite(float(star['rt1'])))
    rsafe=jnp.maximum(rtab,MODE_TINY); psafe=jnp.maximum(ptab,MODE_TINY); hsafe=jnp.maximum(etab+ptab,MODE_TINY); den=jnp.maximum(rsafe*(rsafe-2*mtab),MODE_TINY)
    mp=4*jnp.pi*rsafe**2*etab; nunum=mtab+4*jnp.pi*rsafe**3*ptab; nup=nunum/den; dpdr=-hsafe*nup; dedr=dpdr/jnp.maximum(ceqtab,MODE_TINY)
    nunum_p=mp+12*jnp.pi*rsafe**2*ptab+4*jnp.pi*rsafe**3*dpdr; den_p=2*rsafe-2*mtab-2*rsafe*mp
    u1=1+rsafe*(nunum_p/jnp.maximum(nunum,MODE_TINY)-den_p/den); u2=(rsafe*mp-mtab)/jnp.maximum(rsafe-2*mtab,MODE_TINY)
    v1=(1+etab/psafe)*rsafe*nup; v2=(etab/psafe)*rsafe*nup; ar=dedr/hsafe-dpdr/(jnp.maximum(gtab,MODE_TINY)*psafe)
    a1=mutab/psafe; a2=gtab-2*a1/3; a3=gtab+4*a1/3; e2l=jnp.exp(2*lambdatab)
    bg=jnp.stack((etab,ptab,mutab,gtab,ceqtab,mtab,nutab,lambdatab,nup,dpdr,u1,u2,v1,v2,ar,a1,a2,a3,e2l),axis=1); search_r=jax.lax.stop_gradient(rtab)
    def background(r):
        r=jnp.clip(r,rtab[0],rstar); i=jnp.clip(jnp.searchsorted(search_r,jax.lax.stop_gradient(r),side='right')-1,0,rtab.shape[0]-2); q=(r-rtab[i])/jnp.maximum(rtab[i+1]-rtab[i],MODE_TINY); row=bg[i]+q*(bg[i+1]-bg[i]); return tuple(row)
    ell=jnp.asarray(2.0); lf=ell*(ell+1)
    centre_join=jnp.minimum(jnp.asarray(1e4),.25*rcc)
    scan_core=jnp.concatenate((jnp.geomspace(1.0,centre_join,N_SCAN_CORE_LOG),jnp.linspace(centre_join,rcc,N_SCAN_CORE_RAD)[1:])); scan_crust=jnp.linspace(rcc,rstar,N_SCAN_CRUST)
    eig_core=jnp.concatenate((jnp.geomspace(1.0,centre_join,N_EIGEN_CORE_LOG),jnp.linspace(centre_join,rcc,N_EIGEN_CORE_RAD)[1:])); eig_crust=jnp.linspace(rcc,rstar,N_EIGEN_CRUST)
    def qcoef(r,w,nu0,nup0): return (w/C_CGS)**2*r*jnp.exp(-2*nu0)/jnp.maximum(nup0,MODE_TINY)
    def core_rhs(r,y,w):
        y1,y2=y; _,_,_,ga,_,_,nu0,_,nup0,_,U1,U2,V1,_,Ar,_,_,_,E2=background(r); q=jnp.maximum(qcoef(r,w,nu0,nup0),MODE_TINY)
        return jnp.array((-(3-V1/ga+U2)*y1-(V1/ga-lf/q)*y2,(E2*q+r*Ar)*y1-(U1+r*Ar)*y2))/jnp.maximum(r,MODE_TINY)
    def crust_rhs(r,z,w):
        z1,z2,z3,z4=z; _,_,_,ga,_,_,nu0,_,nup0,_,U1,U2,V1,V2,_,aa1,aa2,aa3,E2=background(r); aa1s=jnp.maximum(aa1,MODE_TINY); aa3s=jnp.maximum(aa3,MODE_TINY); q=qcoef(r,w,nu0,nup0)
        dz1=-(1+2*aa2/aa3s+U2)*z1+z2/aa3s+aa2*lf*z3/aa3s
        dz2=(((-3-U2+U1-E2*q)*V1)+4*aa1*(3*aa2+2*aa1)/aa3s)*z1+(V2-4*aa1/aa3s)*z2+(V1-2*aa1*(1+2*aa2/aa3s))*lf*z3+E2*lf*z4
        dz3=-E2*z1+E2*z4/aa1s
        dz4=(V1-6*ga*aa1/aa3s)*z1-aa2*z2/aa3s-(q*V1+2*aa1-2*aa1*(aa2+aa3)*lf/aa3s)*z3-(3+U2-V2)*z4
        return jnp.array((dz1,dz2,dz3,dz4))/jnp.maximum(r,MODE_TINY)
    def rk4(state,r,h,rhs,w):
        k1=h*rhs(r,state,w); k2=h*rhs(r+h/2,state+k1/2,w); k3=h*rhs(r+h/2,state+k2/2,w); k4=h*rhs(r+h,state+k3,w); c=state+(k1+2*k2+2*k3+k4)/6; s=jnp.maximum(jnp.max(jnp.abs(c)),1e-100); return c/s,jnp.log(s)
    def integrate_final(initial,grid,rhs,w):
        def st(c,pair): state,ls=c; sn,dl=rk4(state,pair[0],pair[1],rhs,w); return (sn,ls+dl),None
        return jax.lax.scan(st,(initial,jnp.asarray(0.0)),jnp.column_stack((grid[:-1],jnp.diff(grid))))[0]
    def integrate_history(initial,grid,rhs,w):
        def st(c,pair): state,ls=c; sn,dl=rk4(state,pair[0],pair[1],rhs,w); lsn=ls+dl; return (sn,lsn),(sn,lsn)
        (f,fl),(states,logs)=jax.lax.scan(st,(initial,jnp.asarray(0.0)),jnp.column_stack((grid[:-1],jnp.diff(grid)))); return f,fl,jnp.vstack((initial,states)),jnp.concatenate((jnp.zeros(1),logs))
    def fopt_jump(y):
        # outward integration: high-density inner side -> low-density outer side.
        def apply(_):
            # V1 on either side follows (1+eps/P) r nu'.  P, r, nu' are continuous,
            # hence the ratio is exactly (eps_hi+P)/(eps_lo+P).
            P=jnp.asarray(star['t']['pt1']); ehi=jnp.asarray(star['t']['eps_phase_high']); elo=jnp.asarray(star['t']['eps_phase_low']); ratio=(ehi+P)/jnp.maximum(elo+P,MODE_TINY)
            y2=y[1]; return jnp.array([y2+ratio*(y[0]-y2),y2])
        return jax.lax.cond((fam==1)&jnp.isfinite(rt1),apply,lambda _:y,None)
    def integrate_core_final(y0,grid,w):
        # fixed-shape split using rt1; points on each wrong side are replaced by the interface.
        if not fam1_split: return integrate_final(y0,grid,core_rhs,w)
        gi=jnp.where(grid<=rt1,grid,rt1); go=jnp.where(grid>=rt1,grid,rt1)
        yi,lsi=integrate_final(y0,gi,core_rhs,w); yj=fopt_jump(yi); yo,lso=integrate_final(yj,go,core_rhs,w); return yo,lsi+lso
    def surface_residual(z):
        # Delta P=0 at a solid-vacuum surface, derived from the Yoshida-Lee z variables,
        # together with zero tangential traction z4=0.
        _,_,_,_,_,_,_,_,_,_,_,_,_,_,_,aa1,_,_,_=background(rstar)
        normal=z[1]+4*aa1*z[0]-2*aa1*lf*z[2]
        return jnp.array([normal,z[3]])
    def matching_on_grids(f,cg,xg):
        w=2*jnp.pi*f; r0=cg[0]; _,_,_,_,_,_,nu0,_,nup0,_,_,_,_,_,_,_,_,_,_=background(r0); q0=qcoef(r0,w,nu0,nup0); y0=jnp.array([1.0,q0/ell]); ycc,_=integrate_core_final(y0,cg,w)
        Vcc=background(rcc)[12]; base=jnp.array([ycc[0],Vcc*(ycc[0]-ycc[1]),0.0,0.0]); diff=jnp.array([0.0,0.0,1.0,0.0]); zb,_=integrate_final(base,xg,crust_rhs,w); zd,_=integrate_final(diff,xg,crust_rhs,w)
        c0=surface_residual(zb); c1=surface_residual(zd); n0=jnp.maximum(jnp.linalg.norm(c0),MODE_TINY); n1=jnp.maximum(jnp.linalg.norm(c1),MODE_TINY); M=jnp.column_stack((c0/n0,c1/n1)); sv=jnp.linalg.svd(M,compute_uv=False); return jnp.linalg.det(M),sv[-1]/jnp.maximum(sv[0],MODE_TINY)
    coarse=lambda f:matching_on_grids(f,scan_core,scan_crust); high=lambda f:matching_on_grids(f,eig_core,eig_crust)
    fscan=jnp.geomspace(FREQ_MIN_HZ,FREQ_MAX_HZ,N_FREQ); detscan,sscan=jax.vmap(coarse)(fscan); sign=jnp.isfinite(detscan[:-1])&jnp.isfinite(detscan[1:])&(detscan[:-1]*detscan[1:]<=0); lm=(sscan[1:-1]<sscan[:-2])&(sscan[1:-1]<=sscan[2:])&(sscan[1:-1]<.35); local=jnp.concatenate((jnp.array([False]),lm,jnp.array([False]))); ids=jnp.nonzero(sign|local[:-1],size=N_CANDIDATES,fill_value=-1)[0]
    def crefine(idx):
        valid=idx>=0; i=jnp.clip(idx,0,N_FREQ-2); lo0,hi0=fscan[i],fscan[i+1]; dlo=detscan[i]; is_sign=sign[i]
        def bi(_,st): lo,hi,vl=st; mid=(lo+hi)/2; vm=coarse(mid)[0]; left=vl*vm<=0; return jnp.where(left,lo,mid),jnp.where(left,mid,hi),jnp.where(left,vl,vm)
        lo,hi,_=jax.lax.fori_loop(0,30,bi,(lo0,hi0,dlo)); rs=(lo+hi)/2; a=fscan[jnp.maximum(i-1,0)]; b=fscan[jnp.minimum(i+2,N_FREQ-1)]; gr=(jnp.sqrt(5.)-1)/2; c=b-gr*(b-a); d=a+gr*(b-a); fc,fd=coarse(c)[1],coarse(d)[1]
        def go(_,st):
            a,b,c,d,fc,fd=st; left=fc<fd; an=jnp.where(left,a,c); bn=jnp.where(left,d,b); cn=jnp.where(left,b-gr*(b-a),d); dn=jnp.where(left,c,a+gr*(b-a)); return an,bn,cn,dn,coarse(cn)[1],coarse(dn)[1]
        a,b,c,d,fc,fd=jax.lax.fori_loop(0,24,go,(a,b,c,d,fc,fd)); rm=jnp.where(fc<fd,c,d); root=jnp.where(is_sign,rs,rm); de,ra=coarse(root); acc=valid&jnp.isfinite(root)&(ra<COARSE_RATIO_MAX); return jnp.where(acc,root,jnp.nan),ra,de
    rawf,rawr,rawd=jax.vmap(crefine)(ids); order=jnp.argsort(jnp.where(jnp.isfinite(rawf),rawf,jnp.inf)); sf,sr,sd=rawf[order],rawr[order],rawd[order]
    def dd(prev,item): f,r,d=item; keep=jnp.isfinite(f)&((~jnp.isfinite(prev))|(f-prev>.12)); return jnp.where(keep,f,prev),(jnp.where(keep,f,jnp.nan),jnp.where(keep,r,jnp.inf),jnp.where(keep,d,jnp.nan),keep)
    _,ded=jax.lax.scan(dd,jnp.asarray(jnp.nan),(sf,sr,sd)); df,dr,ddet,dk=ded; vi=jnp.nonzero(dk,size=N_COARSE_MODES,fill_value=-1)[0]; cf=jnp.where(vi>=0,df[jnp.clip(vi,0,df.shape[0]-1)],jnp.nan)
    finite=jnp.isfinite(cf); prev=jnp.r_[jnp.nan,cf[:-1]]; nxt=jnp.r_[cf[1:],jnp.nan]; lb=jnp.where(jnp.isfinite(prev),jnp.sqrt(jnp.maximum(prev*cf,MODE_TINY)),jnp.maximum(FREQ_MIN_HZ,.82*cf)); rb=jnp.where(jnp.isfinite(nxt),jnp.sqrt(jnp.maximum(cf*nxt,MODE_TINY)),jnp.minimum(FREQ_MAX_HZ,1.18*cf)); lb=jnp.clip(lb,FREQ_MIN_HZ,FREQ_MAX_HZ); rb=jnp.clip(rb,FREQ_MIN_HZ,FREQ_MAX_HZ)
    def href(args):
        f0,lo0,hi0,valid=args; fs=jnp.where(valid,f0,100.); lo0=jnp.where(valid,lo0,90.); hi0=jnp.where(valid,hi0,110.); dlo,rlo=high(lo0); dhi,rhi=high(hi0); br=jnp.isfinite(dlo)&jnp.isfinite(dhi)&(dlo*dhi<=0)
        def sb(_):
            def bi(_,st): lo,hi,vl=st; mid=(lo+hi)/2; vm=high(mid)[0]; left=vl*vm<=0; return jnp.where(left,lo,mid),jnp.where(left,mid,hi),jnp.where(left,vl,vm)
            lo,hi,_=jax.lax.fori_loop(0,24,bi,(lo0,hi0,dlo)); return (lo+hi)/2
        def mb(_):
            gr=(jnp.sqrt(5.)-1)/2; a,b=lo0,hi0; c,d=b-gr*(b-a),a+gr*(b-a); fc,fd=high(c)[1],high(d)[1]
            def go(_,st): a,b,c,d,fc,fd=st; left=fc<fd; an=jnp.where(left,a,c); bn=jnp.where(left,d,b); cn=jnp.where(left,b-gr*(b-a),d); dn=jnp.where(left,c,a+gr*(b-a)); return an,bn,cn,dn,high(cn)[1],high(dn)[1]
            a,b,c,d,fc,fd=jax.lax.fori_loop(0,24,go,(a,b,c,d,fc,fd)); return jnp.where(fc<fd,c,d)
        root=jax.lax.cond(br,sb,mb,None); de,ra=high(root); acc=valid&jnp.isfinite(root)&jnp.isfinite(ra)&(ra<FINAL_RATIO_MAX); return jnp.where(acc,root,jnp.nan),jnp.where(acc,ra,jnp.inf),de
    # mini-batched candidate refinement: same physics, lower peak memory than a full vmap.
    hf0,hr0,hd0=jax.lax.map(href,(cf,lb,rb,finite),batch_size=CANDIDATE_BATCH); ho=jnp.argsort(jnp.where(jnp.isfinite(hf0),hf0,jnp.inf)); hf,hr,hd=hf0[ho],hr0[ho],hd0[ho]
    def fdd(c,item): pf,pr=c; f,r,d=item; keep=jnp.isfinite(f)&((~jnp.isfinite(pf))|(f-pf>DEDUP_HZ)); return (jnp.where(keep,f,pf),jnp.where(keep,r,pr)),(jnp.where(keep,f,jnp.nan),jnp.where(keep,r,jnp.inf),jnp.where(keep,d,jnp.nan),keep)
    _,fded=jax.lax.scan(fdd,(jnp.asarray(jnp.nan),jnp.asarray(jnp.inf)),(hf,hr,hd)); ff,fr,fdet,fk=fded; fi=jnp.nonzero(fk,size=N_MODES,fill_value=-1)[0]; freq=jnp.where(fi>=0,ff[jnp.clip(fi,0,ff.shape[0]-1)],jnp.nan); ratios=jnp.where(fi>=0,fr[jnp.clip(fi,0,fr.shape[0]-1)],jnp.inf); valid=jnp.isfinite(freq)&(ratios<FINAL_RATIO_MAX)
    def node_count(values):
        # Local-envelope normalisation: a node counts when the function changes sign against its OWN neighbourhood's
        # amplitude, so a core oscillation a million times weaker than the crust still counts (v3.3 fix).
        w=25; kern=jnp.ones(2*w+1)/(2*w+1); env=jnp.sqrt(jnp.maximum(jnp.convolve(values*values,kern,mode='same'),MODE_TINY))
        gsc=jnp.maximum(jnp.max(jnp.abs(values)),MODE_TINY); xn=jnp.where(jnp.abs(values)>1e-12*gsc,values/env,0.0)
        signs=jnp.sign(jnp.where(jnp.abs(xn)>1e-3,xn,0.0));
        def st(last,s): active=s!=0; ch=active&(last!=0)&(s!=last); return jnp.where(active,s,last),ch.astype(jnp.int32)
        _,ch=jax.lax.scan(st,jnp.asarray(0.0),signs); return jnp.sum(ch)
    def trap(y,x): return jnp.sum(.5*(y[:-1]+y[1:])*jnp.diff(x))
    def reconstruct(args):
        f,vm=args; fs=jnp.where(vm,f,100.); w=2*jnp.pi*fs; r0=eig_core[0]; _,_,_,_,_,_,nu0,_,nup0,_,_,_,_,_,_,_,_,_,_=background(r0); q0=qcoef(r0,w,nu0,nup0); y0=jnp.array([1.,q0/ell])
        # History path currently uses continuous radial background; the explicit rapid jump is applied to the state.
        # Split histories at rt1 for F1 so the eigenfunction itself respects the phase conversion condition.
        if fam1_split:
            gi=jnp.where(eig_core<=rt1,eig_core,rt1); go=jnp.where(eig_core>=rt1,eig_core,rt1); yi,lsi,yhi,li=integrate_history(y0,gi,core_rhs,w); yj=fopt_jump(yi); yo,lso,yho,lo=integrate_history(yj,go,core_rhs,w); use_inner=eig_core<=rt1; yh=jnp.where(use_inner[:,None],yhi,yho); ylogs=jnp.where(use_inner,li,lo+lsi); ycc=yo; yscale=lsi+lso
        else: ycc,yscale,yh,ylogs=integrate_history(y0,eig_core,core_rhs,w)
        Vcc=background(rcc)[12]; base=jnp.array([ycc[0],Vcc*(ycc[0]-ycc[1]),0.,0.]); diff=jnp.array([0.,0.,1.,0.]); zb,sb,hb,hbl=integrate_history(base,eig_crust,crust_rhs,w); zd,sd,hd,hdl=integrate_history(diff,eig_crust,crust_rhs,w); c0=surface_residual(zb); c1=surface_residual(zd); n0=jnp.maximum(jnp.linalg.norm(c0),MODE_TINY); n1=jnp.maximum(jnp.linalg.norm(c1),MODE_TINY); M=jnp.column_stack((c0/n0,c1/n1)); _,sv,vh=jnp.linalg.svd(M,full_matrices=False); null=vh[-1]; ratio=sv[-1]/jnp.maximum(sv[0],MODE_TINY)
        l0,l1=jnp.log(n0),jnp.log(n1); lc=(ylogs-yscale)-sb-l0; lb0=hbl-sb-l0; ld=hdl-sd-l1; ref=jnp.maximum(jnp.maximum(jnp.max(lc),jnp.max(lb0)),jnp.max(ld)); core=null[0]*yh*jnp.exp(jnp.clip(lc-ref,-745,0))[:,None]; baseh=null[0]*hb*jnp.exp(jnp.clip(lb0-ref,-745,0))[:,None]; diffh=null[1]*hd*jnp.exp(jnp.clip(ld-ref,-745,0))[:,None]; crust=baseh+diffh
        def qat(r): _,_,_,_,_,_,nu0,_,nup0,_,_,_,_,_,_,_,_,_,_=background(r); return jnp.maximum(qcoef(r,w,nu0,nup0),MODE_TINY)
        qc=jax.vmap(qat)(eig_core); cr=core[:,0]; ct=core[:,1]/qc; xr=crust[:,0]; xt=crust[:,2]; rad=jnp.r_[cr,xr[1:]]; tan=jnp.r_[ct,xt[1:]]; radius=jnp.r_[eig_core,eig_crust[1:]]; norm=jnp.sqrt(jnp.maximum(jnp.max(rad*rad+tan*tan),MODE_TINY)); rad/=norm; tan/=norm; core/=norm; crust/=norm; amp=rad*rad+tan*tan; total=jnp.maximum(trap(amp,radius),MODE_TINY); win=jnp.abs(radius-rcc)<=jnp.maximum(.04*rstar,5e4); crustreg=radius>=rcc; iface=trap(jnp.where(win,amp,0),radius)/total; cfrac=trap(jnp.where(crustreg,amp,0),radius)/total; surf=amp[-1]/jnp.maximum(jnp.max(amp),MODE_TINY); cn=node_count(cr); crn=node_count(xr); ctn=node_count(xt); cder=core_rhs(rcc,core[-1],w); xder=crust_rhs(rcc,crust[0],w); cusp=jnp.abs(xder[0]-cder[0])/jnp.maximum(jnp.abs(xder[0])+jnp.abs(cder[0]),MODE_TINY); _,_,_,_,_,_,nuc,_,nupc,_,_,_,_,_,_,_,_,_,_=background(rcc); qcc=jnp.maximum(qcoef(rcc,w,nuc,nupc),MODE_TINY); fj=core[-1,1]/qcc; sj=crust[0,2]; tj=jnp.abs(sj-fj)/jnp.maximum(jnp.abs(sj)+jnp.abs(fj),MODE_TINY); bres=jnp.linalg.norm(M@null)/jnp.maximum(jnp.linalg.norm(M)*jnp.linalg.norm(null),MODE_TINY); met=jnp.array([iface,cfrac,surf,cn,crn,ctn,ratio,jnp.abs(jnp.linalg.det(M)),cusp,tj,bres]); return radius/rstar,jnp.where(vm,rad,0),jnp.where(vm,tan,0),met
    er,ex,et,metrics=jax.lax.map(reconstruct,(freq,valid),batch_size=RECON_BATCH); valid=valid&(metrics[:,6]<FINAL_RATIO_MAX)&(metrics[:,10]<FINAL_RATIO_MAX)
    return freq,valid,metrics,fscan,detscan,sscan,er,ex,et


# Deliberately not jitted globally: star dictionaries contain host metadata.
# The heavy maps/scans inside are JAX operations and compile on first execution.

def _host_node_count(values, rel_floor=1e-3, win=25):
    """Count sign-changing radial nodes against a LOCAL amplitude envelope (v3.3): a global floor erased the
    weak core oscillation of g-modes whose crust amplitude dominates."""
    x=np.asarray(values,float)
    if not np.any(np.isfinite(x)):
        return 0
    x=np.where(np.isfinite(x),x,0.0); scale=np.max(np.abs(x))
    if not np.isfinite(scale) or scale<=0:
        return 0
    env=np.sqrt(np.convolve(x*x,np.ones(2*win+1)/(2*win+1),mode='same')); env=np.maximum(env,1e-300)
    xn=np.where(np.abs(x)>1e-12*scale,x/env,0.0)
    s=np.sign(np.where(np.abs(xn)>rel_floor,xn,0.0))
    last=0.0; n=0
    for si in s:
        if si==0: continue
        if last!=0 and si!=last: n+=1
        last=si
    return int(n)


_trapz=getattr(np,'trapezoid',None) or getattr(np,'trapz')   # numpy >=2.4 removed trapz: family-1 draws died here (212 lost)

def _radial_fraction(radius, radial, tangential, centre, halfwidth):
    radius=np.asarray(radius,float); radial=np.asarray(radial,float); tangential=np.asarray(tangential,float)
    amp=radial*radial+tangential*tangential
    if radius.size<2 or not np.any(np.isfinite(amp)):
        return 0.0
    total=_trapz(amp,radius)
    if not np.isfinite(total) or total<=0:
        return 0.0
    mask=np.abs(radius-centre)<=halfwidth
    return float(_trapz(np.where(mask,amp,0.0),radius)/total)


def classify_full_spectrum(freq,valid,metrics,eigen_radius,eigen_radial,eigen_tangential,star,eos):
    """Source-faithful family/rung assignment for every accepted spheroidal root.

    Important distinction: the cited literature gives physical mode definitions,
    not a published numerical threshold classifier.  This function therefore uses
    topology and relative morphology to automate those definitions and records the
    operational step used for every label.  Classification never controls whether
    a numerically valid root survives.

    Source rules used here:
      * Neill et al. (2026): crust-core i has zero core radial nodes, a radial
        cusp at Rcc, and a transverse-displacement discontinuity at Rcc.  In a
        mixed g1/i or i/s1 pair, their convention assigns i to the lower-frequency
        member.
      * Yoshida & Lee (2002), following McDermott et al. (1988): p_n are acoustic
        overtones, g_n are gravity overtones in their fluid cavity, f is the single
        node-free fundamental branch between g1 and p1, and s_n are spheroidal
        shear-dominated overtones strongly confined to the solid crust.
      * A fluid-fluid density discontinuity is a discontinuity g-mode, not an
        elastic solid-fluid i-mode.
    """
    f=np.asarray(freq,float); v=np.asarray(valid,bool); m=np.asarray(metrics,float)
    er=np.asarray(eigen_radius,float); ex=np.asarray(eigen_radial,float); et=np.asarray(eigen_tangential,float)
    labels=np.zeros(len(f),np.int32); labels[v]=7
    confidence=np.zeros(len(f),float)
    basis=np.full(len(f),'invalid',dtype=object)
    source_exact=np.zeros(len(f),bool)
    global_nodes=np.zeros(len(f),np.int32)
    fopt_fraction=np.zeros(len(f),float)
    i_score=np.zeros(len(f),float)
    mixed_i_pair=[]

    ids=np.where(v)[0]
    if not len(ids):
        return labels,None,confidence,dict(basis=basis,source_exact=source_exact,global_nodes=global_nodes,fopt_fraction=fopt_fraction,i_score=i_score,mixed_i_pair=mixed_i_pair)

    for j in ids:
        global_nodes[j]=_host_node_count(ex[j])

    # F1 has a fluid-fluid phase boundary at rt1.  Measure localisation there so
    # a discontinuity g-mode is not confused with the elastic crust-core i-mode.
    rt1=float(star.get('rt1',np.nan))/float(star['rstar']) if np.isfinite(star.get('rt1',np.nan)) else np.nan
    if int(eos.get('family',0))==1 and np.isfinite(rt1) and 0<rt1<1:
        for j in ids:
            fopt_fraction[j]=_radial_fraction(er[j],ex[j],et[j],rt1,max(0.015,4.0/max(er[j].size,1)))

    # ---- crust-core i mode ----
    # No numerical cusp/jump cut is published by Neill et al.  We therefore use
    # their three qualitative criteria and a threshold-free relative ranking among
    # zero-core-node roots.  The score is only an operational tie-breaker.
    # A root QUALIFIES as i when it satisfies all three published criteria:
    # zero core radial nodes, a radial cusp at Rcc, and a transverse jump at
    # Rcc.  In a stratified star SEVERAL zero-node roots can qualify (the
    # ladder-top member and the bare interface root are the entangled pair the
    # 2026 paper warns about), and its convention is verbatim: "we treat the
    # lower-frequency mode as the i-mode".  So the selection is the LOWEST
    # frequency qualifier, not a morphology score; the score is kept only as
    # a stored diagnostic.
    icand=np.where(v&(np.rint(m[:,3]).astype(int)==0))[0]
    i_idx=None
    if len(icand):
        cu=np.asarray(m[icand,8],float); tj=np.asarray(m[icand,9],float)
        i_score[icand]=np.sqrt(np.maximum(cu,0)*np.maximum(tj,0))
        qual=icand[(cu>=0.3)&(tj>=0.3)]
        if len(qual):
            qual=qual[np.argsort(f[qual])]
            i_idx=int(qual[0])
            if len(qual)>1:
                mixed_i_pair=[int(qual[0]),int(qual[1])]
            labels[i_idx]=2; confidence[i_idx]=1.0
            basis[i_idx]='Neill2026: zero core radial nodes + Rcc radial cusp + Rcc transverse jump; lower-frequency member taken when several qualify (source mixing rule)'
            source_exact[i_idx]=True

    # ---- shear family ----
    # After removing i, s_n has no core radial nodes and develops elastic nodes in
    # the solid crust.  Frequency orders overtones only after the family is known.
    sids=np.where(v&(labels==7)&(np.rint(m[:,3]).astype(int)==0)&((np.rint(m[:,4]).astype(int)+np.rint(m[:,5]).astype(int))>0))[0]
    for j in sids:
        labels[j]=3; confidence[j]=float(np.clip(m[j,1],0,1))
        basis[j]='Yoshida-Lee/McDermott: spheroidal shear branch, zero core radial nodes with elastic crust nodes; overtone ordered within branch'
        source_exact[j]=True

    # ---- Family-1 discontinuity gravity mode ----
    # A sharp fluid-fluid density jump supports a discontinuity g mode.  There is
    # no crustal transverse-traction discontinuity to make it an elastic i mode.
    if int(eos.get('family',0))==1 and np.isfinite(rt1):
        dcand=np.where(v&(labels==7)&(fopt_fraction>0))[0]
        if len(dcand):
            # Select only when the phase-boundary localisation beats that root's
            # crust-core localisation.  This is a geometric location test, not a
            # frequency window.
            viable=[int(j) for j in dcand if fopt_fraction[j] > float(m[j,0])]
            if viable:
                j=max(viable,key=lambda q:fopt_fraction[q])
                labels[j]=6; confidence[j]=float(np.clip(fopt_fraction[j],0,1))
                basis[j]='McDermott/Finn taxonomy: gravity mode localised at a fluid-fluid density discontinuity (F1 phase boundary)'
                source_exact[j]=True

    # ---- f branch ----
    # The f mode is the single global node-free branch.  Interface/shear roots have
    # already been removed.  If more than one node-free candidate remains, choose
    # the least crust-confined one and flag the choice as operational.
    fcand=np.where(v&(labels==7)&(global_nodes==0)&(np.rint(m[:,3]).astype(int)==0))[0]
    f_idx=None
    if len(fcand):
        f_idx=int(fcand[np.argmin(m[fcand,1])])
        labels[f_idx]=4; confidence[f_idx]=float(np.clip(1-m[f_idx,1],0,1))
        basis[f_idx]='Yoshida-Lee/McDermott: single global node-free fundamental branch; least crust-confined node-free remainder used if more than one candidate'
        source_exact[f_idx]=(len(fcand)==1)

    # ---- gravity and pressure overtones ----
    # In the standard spheroidal fluid ordering the f branch separates g1 below
    # from p1 above.  Overtone topology is counted from the radial eigenfunction.
    if f_idx is not None:
        for j in np.where(v&(labels==7))[0]:
            if f[j] < f[f_idx] and int(round(m[j,3]))>0:
                labels[j]=1; confidence[j]=1.0
                basis[j]='Yoshida-Lee/McDermott gravity branch below f; rung = core radial-node order'
                source_exact[j]=True
            elif f[j] > f[f_idx] and global_nodes[j]>0:
                labels[j]=5; confidence[j]=1.0
                basis[j]='Yoshida-Lee/McDermott acoustic branch above f; rung = global radial overtone topology'
                source_exact[j]=True

    # ---- gravity-ladder fallback ----
    # A dense stratified star can fill the whole retained-root budget with the
    # g ladder before the f/p band is reached, so f_idx is absent.  The gravity
    # branch is still unambiguous: buoyancy overtones carry core radial nodes
    # and lie BELOW every zero-core-node root (i, s, f all sit above the
    # ladder).  Label those directly by their core radial order.
    zero_node=np.where(v&(np.rint(m[:,3]).astype(int)==0))[0]
    f_floor=float(np.min(f[zero_node])) if len(zero_node) else np.inf
    for j in np.where(v&(labels==7))[0]:
        if int(round(m[j,3]))>0 and f[j]<f_floor:
            labels[j]=1; confidence[j]=1.0
            basis[j]='Yoshida-Lee/McDermott gravity branch below the zero-core-node band; rung = core radial-node order'
            source_exact[j]=True

    for j in np.where(v&(labels==7))[0]:
        basis[j]='retained physical root; source taxonomy does not support an unambiguous automated family assignment from this single-star diagnostic set'
        confidence[j]=0.0

    extra=dict(
        basis=basis,source_exact=source_exact,global_nodes=global_nodes,
        fopt_fraction=fopt_fraction,i_score=i_score,mixed_i_pair=mixed_i_pair,
        f_index=f_idx,
    )
    return labels,i_idx,confidence,extra


def spectrum_rows(freq,valid,labels,metrics,confidence=None,classification=None):
    f=np.asarray(freq,float); v=np.asarray(valid,bool); lab=np.asarray(labels,int); m=np.asarray(metrics,float)
    conf=np.zeros(len(f)) if confidence is None else np.asarray(confidence,float)
    c={} if classification is None else classification
    basis=np.asarray(c.get('basis',np.full(len(f),'',dtype=object)),dtype=object)
    source_exact=np.asarray(c.get('source_exact',np.zeros(len(f),bool)),bool)
    global_nodes=np.asarray(c.get('global_nodes',np.zeros(len(f),int)),int)
    fopt_fraction=np.asarray(c.get('fopt_fraction',np.zeros(len(f),float)),float)
    i_score=np.asarray(c.get('i_score',np.zeros(len(f),float)),float)
    rows=[]
    sids=sorted(np.where(v&(lab==3))[0],key=lambda i:f[i]); srank={idx:k+1 for k,idx in enumerate(sids)}
    gids=sorted(np.where(v&(lab==6))[0],key=lambda i:f[i]); drank={idx:k+1 for k,idx in enumerate(gids)}
    # For p modes, use the radial overtone count where unique.  Frequency rank is
    # retained as a fallback if a numerical node collision occurs.
    pids=sorted(np.where(v&(lab==5))[0],key=lambda i:f[i]); prank={idx:k+1 for k,idx in enumerate(pids)}
    # g rung = core radial order when that is injective; otherwise rank from the ladder top (highest frequency = g1)
    gl=sorted(np.where(v&(lab==1))[0],key=lambda i:-f[i]); gcn=[int(round(m[i,3])) for i in gl]
    if len(set(gcn))==len(gcn): grung={i:max(c,1) for i,c in zip(gl,gcn)}
    else: grung={i:k+max(gcn[0],1) for k,i in enumerate(gl)}
    for i in np.where(v)[0]:
        kind=MODE_LABELS[int(lab[i])]
        cn=int(round(m[i,3])); crn=int(round(m[i,4])); ctn=int(round(m[i,5]))
        if kind=='g': name=f'g{grung[i]}'
        elif kind=='s': name=f's{srank[i]}'
        elif kind=='p': name=f'p{global_nodes[i] if global_nodes[i]>0 else prank[i]}'
        elif kind=='g_disc': name=f'g_disc{drank[i]}'
        elif kind=='i': name='i'
        elif kind=='f': name='f'
        else: name=f'unclassified_{i}'
        rows.append(dict(
            index=int(i),mode=name,f_hz=float(f[i]),label=kind,
            classification_confidence=float(conf[i]),classification_basis=str(basis[i]),
            source_definition_satisfied=bool(source_exact[i]),global_radial_nodes=int(global_nodes[i]),
            fopt_localisation_fraction=float(fopt_fraction[i]),i_morphology_score=float(i_score[i]),
            interface_fraction=float(m[i,0]),crust_fraction=float(m[i,1]),surface_fraction=float(m[i,2]),
            core_nodes=cn,crust_radial_nodes=crn,crust_tangential_nodes=ctn,
            cusp=float(m[i,8]),transverse_jump=float(m[i,9]),
            singular_ratio=float(m[i,6]),boundary_residual=float(m[i,10]),
        ))
    return rows


def build_fixed_mode_slots(rows):
    """Derived ML target.  It never controls which physical roots are kept."""
    freq=np.full(len(SLOT_NAMES),np.nan,float)
    mask=np.zeros(len(SLOT_NAMES),np.uint8)
    source_index=np.full(len(SLOT_NAMES),-1,np.int32)
    slot={name:i for i,name in enumerate(SLOT_NAMES)}
    collisions=[]
    for r in rows:
        name=r['mode']
        if name not in slot:
            continue
        j=slot[name]
        if mask[j]:
            collisions.append((name,int(source_index[j]),r['index']))
            # Keep the root with the smaller high-resolution residual.
            old=next(x for x in rows if x['index']==int(source_index[j]))
            if r['boundary_residual'] >= old['boundary_residual']:
                continue
        freq[j]=r['f_hz']; mask[j]=1; source_index[j]=r['index']
    return dict(names=np.asarray(SLOT_NAMES,dtype='U12'),frequencies_hz=freq,mask=mask,source_index=source_index,collisions=collisions)


def _positive_xy(x, y):
    """Finite positive values for log-scale diagnostic plots."""
    x=np.asarray(x,dtype=float); y=np.asarray(y,dtype=float)
    m=np.isfinite(x)&np.isfinite(y)&(x>0)&(y>0)
    return x[m],y[m]


def plot_bps_cldm_join(eos):
    """Production CLDM crust with the BPS 1971 rows overlaid as a diagnostic."""
    bps=eos['bps']; cldm=eos['cldm']; cr=eos['crust']
    fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    ax=axs[0]
    xa,ya=_positive_xy(cr['n'],cr['P']); xb,yb=_positive_xy(bps['n'],bps['P'])
    ax.plot(xa,ya,label='production CLDM crust',lw=2.0)
    ax.plot(xb,yb,'o',ms=3,alpha=.6,label='BPS 1971 (diagnostic)')
    for val,name in ((cldm['ndrip'],'drip'),(cldm['n18'],'1/8'),(cldm['ncc'],'cc')):
        if np.isfinite(val) and val>0: ax.axvline(val,ls='--',lw=.9,label=name)
    if np.isfinite(cldm.get('ncc_marched',np.nan)):
        ax.axvline(cldm['ncc_marched'],ls=':',lw=.9,label='cc marched (diagnostic)')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel(r'$n\;[\mathrm{fm}^{-3}]$'); ax.set_ylabel(r'$P\;[\mathrm{MeV\,fm}^{-3}]$')
    ax.set_title('Crust pressure: production vs 1971 diagnostic'); ax.legend(fontsize=8)

    ax=axs[1]
    ea=np.asarray(cr['eps'])/np.maximum(np.asarray(cr['n']),1e-300)
    eb=np.asarray(bps['eps'])/np.maximum(np.asarray(bps['n']),1e-300)
    ax.plot(cr['n'],ea,label='production CLDM crust',lw=2.0)
    ax.plot(bps['n'],eb,'o',ms=3,alpha=.6,label='BPS 1971 (diagnostic)')
    for val in (cldm['ndrip'],cldm['n18'],cldm['ncc']):
        if np.isfinite(val) and val>0: ax.axvline(val,ls='--',lw=.9)
    ax.set_xscale('log'); ax.set_xlabel(r'$n\;[\mathrm{fm}^{-3}]$'); ax.set_ylabel(r'$\varepsilon/n\;[\mathrm{MeV}]$')
    ax.set_title('Energy per baryon'); ax.legend(fontsize=8)
    fig.suptitle('Crust diagnostics (one CLDM cell, surface to $n_{cc}$)'); fig.tight_layout(); return fig


def plot_cldm_microphysics(eos):
    """Plot the CLDM composition and elastic quantities used by the mode solver."""
    c=eos['cldm']; n=np.asarray(c['n'],float)
    fig,axs=plt.subplots(2,2,figsize=(12,8))
    axs[0,0].plot(n,c['u']); axs[0,0].set_ylabel('cluster volume fraction $u$')
    axs[0,1].plot(n,c['Z'],label='Z'); axs[0,1].plot(n,c['A'],label='A'); axs[0,1].set_ylabel('cluster composition'); axs[0,1].legend()
    xp,yp=_positive_xy(n,c['mu']); axs[1,0].plot(xp,yp); axs[1,0].set_yscale('log'); axs[1,0].set_ylabel(r'$\mu_{\rm sh}\;[\mathrm{MeV\,fm}^{-3}]$')
    axs[1,1].plot(n,c['G1'],label=r'$\Gamma_1$'); axs[1,1].plot(n,c['Geq'],label=r'$\Gamma_{\rm eq}$'); axs[1,1].set_ylabel('adiabatic index'); axs[1,1].legend()
    for ax in axs.flat:
        ax.set_xscale('log'); ax.set_xlabel(r'$n\;[\mathrm{fm}^{-3}]$')
        for val in (c['ndrip'],c['n18'],c['ncc']):
            if np.isfinite(val) and val>0: ax.axvline(val,ls='--',lw=.8)
    fig.suptitle('CLDM composition, stiffness and elasticity'); fig.tight_layout(); return fig


def _eos_regions(eos):
    cr=eos['crust']; oc=np.asarray(eos['outer_core']); ic=np.asarray(eos['inner_core'])
    return [
        ('crust',np.asarray(cr['n']),np.asarray(cr['eps']),np.asarray(cr['P']),np.asarray(cr['G1']),np.asarray(cr['Geq']),np.asarray(cr['mu'])),
        ('outer core',oc[0],oc[1],oc[2],oc[3],oc[4],oc[6]),
        ('inner core',ic[0],ic[1],ic[2],ic[3],ic[4],ic[6]),
    ]


def plot_eos_overview(eos):
    """Full EOS thermodynamics from crust through sampled inner core."""
    fig,axs=plt.subplots(1,3,figsize=(15,4.5))
    for name,n,e,p,g1,ge,mu in _eos_regions(eos):
        x,y=_positive_xy(e,p); axs[0].plot(x,y,label=name)
        cs2=np.asarray(ge)*np.asarray(p)/np.maximum(np.asarray(e)+np.asarray(p),1e-300)
        axs[1].plot(n,cs2,label=name)
        axs[2].plot(n,np.asarray(g1)-np.asarray(ge),label=name)
    axs[0].set_xscale('log'); axs[0].set_yscale('log'); axs[0].set_xlabel(r'$\varepsilon\;[\mathrm{MeV\,fm}^{-3}]$'); axs[0].set_ylabel(r'$P\;[\mathrm{MeV\,fm}^{-3}]$'); axs[0].set_title('EOS')
    axs[1].set_xscale('log'); axs[1].set_xlabel(r'$n\;[\mathrm{fm}^{-3}]$'); axs[1].set_ylabel(r'$c_{\rm eq}^2/c^2$'); axs[1].axhline(1.0,ls='--',lw=.8); axs[1].set_title('Equilibrium sound speed')
    axs[2].set_xscale('log'); axs[2].set_xlabel(r'$n\;[\mathrm{fm}^{-3}]$'); axs[2].set_ylabel(r'$\Gamma_1-\Gamma_{\rm eq}$'); axs[2].axhline(0.0,ls='--',lw=.8); axs[2].set_title('Compositional stratification')
    for ax in axs: ax.legend(fontsize=8)
    fig.suptitle(f'Full EOS overview — F{eos["family"]}, $n_{{t1}}={eos["nt1"]:.3f}$ fm$^{{-3}}$'); fig.tight_layout(); return fig


def plot_mass_radius(mr,star=None):
    """Resolved stable M-R branch, maximum mass, and optional target star."""
    R=np.asarray(mr['R'],float)/1e5; M=np.asarray(mr['M'],float)*C_CGS**2/(G_CGS*MSUN_CGS); ok=np.asarray(mr['ok'],bool)
    fig,ax=plt.subplots(figsize=(6.5,5.0))
    if np.any(ok): ax.plot(R[ok],M[ok],lw=2,label='resolved branch')
    if mr.get('resolved',False):
        k=int(mr['kmax']); ax.scatter([R[k]],[M[k]],s=55,label=f'Mmax={M[k]:.3f} Msun',zorder=5)
    if star is not None:
        ax.scatter([star['rstar']/1e5],[star['mstar']*C_CGS**2/(G_CGS*MSUN_CGS)],s=65,marker='*',label='target star',zorder=6)
    ax.set_xlabel('R [km]'); ax.set_ylabel(r'$M/M_\odot$'); ax.set_title('Mass–radius sequence'); ax.legend(); fig.tight_layout(); return fig


def plot_star_profiles(star):
    """Radial background profiles for the exact target star used by the mode solve."""
    r=np.asarray(star['r'],float)/1e5; R=star['rstar']/1e5; rcc=star['rcc']/1e5; rt1=star['rt1']/1e5 if np.isfinite(star['rt1']) else np.nan
    fig,axs=plt.subplots(2,2,figsize=(12,8))
    x,y=_positive_xy(r,star['P']); axs[0,0].plot(x,y); axs[0,0].set_yscale('log'); axs[0,0].set_ylabel(r'$P$ [geom.]')
    x,y=_positive_xy(r,star['n']); axs[0,1].plot(x,y); axs[0,1].set_yscale('log'); axs[0,1].set_ylabel(r'$n\;[\mathrm{fm}^{-3}]$')
    axs[1,0].plot(r,star['G1'],label=r'$\Gamma_1$'); axs[1,0].plot(r,star['Geq'],label=r'$\Gamma_{\rm eq}$'); axs[1,0].set_ylabel('adiabatic index'); axs[1,0].legend()
    mu=np.asarray(star['mu'],float); pos=mu>0
    if np.any(pos): axs[1,1].plot(r[pos],mu[pos]); axs[1,1].set_yscale('log')
    axs[1,1].set_ylabel(r'$\mu_{\rm sh}$ [geom.]')
    for ax in axs.flat:
        ax.set_xlabel('r [km]'); ax.axvline(rcc,ls='--',lw=.9,label='$R_{cc}$')
        if np.isfinite(rt1) and 0<rt1<R: ax.axvline(rt1,ls=':',lw=.9,label='$R_{t1}$')
        ax.set_xlim(0,R*1.01)
    axs[0,0].legend(fontsize=8)
    fig.suptitle('Target-star background used by the mode equations'); fig.tight_layout(); return fig

def plot_mode_spectrum(rows,title='mode spectrum'):
    order=['g','g_disc','i','s','f','p','unclassified']; ypos={k:i for i,k in enumerate(order)}; fig,ax=plt.subplots(figsize=(10,5.2))
    for r in rows:
        y=ypos[r['label']]; ax.scatter(r['f_hz'],y,s=55); ax.annotate(f"{r['mode']}  {r['f_hz']:.1f}",(r['f_hz'],y),xytext=(4,5),textcoords='offset points',fontsize=8)
    ax.axvspan(100,600,alpha=.07,label='typical i range, not a gate'); ax.set_xscale('log'); ax.set_xlim(FREQ_MIN_HZ,FREQ_MAX_HZ); ax.set_yticks(list(ypos.values()),list(ypos.keys())); ax.set_xlabel('frequency [Hz]'); ax.set_title(title); ax.grid(True,axis='x',which='both',alpha=.2); ax.legend(fontsize=8); fig.tight_layout(); return fig


def plot_scan(spec):
    fig,ax=plt.subplots(figsize=(9,4)); ax.plot(spec['frequency_scan_hz'],spec['singular_ratio_scan']); ax.set_xscale('log'); ax.set_yscale('log'); ax.axhline(FINAL_RATIO_MAX,ls='--'); ax.set_xlabel('frequency [Hz]'); ax.set_ylabel('smallest/largest singular value'); fig.tight_layout(); return fig


def _show_figure(fig):
    """Force a figure to render immediately in notebooks and interactive Python."""
    try:
        from IPython.display import display
        display(fig)
        plt.close(fig)
    except Exception:
        plt.show()



def plot_all_mode_shapes(spec, star, modes_per_figure=6):
    rows=list(spec.get('rows',[]))
    figs=[]
    for start in range(0,len(rows),modes_per_figure):
        subset=rows[start:start+modes_per_figure]
        if not subset: continue
        n=len(subset); ncols=2; nrows=max(1,math.ceil(n/ncols))
        fig,axs=plt.subplots(nrows,ncols,figsize=(12,3.4*nrows),squeeze=False)
        rcc=float(star['rcc'])/float(star['rstar']) if np.isfinite(star['rcc']) else np.nan
        rt1=float(star['rt1'])/float(star['rstar']) if np.isfinite(star['rt1']) else np.nan
        for ax,row in zip(axs.ravel(),subset):
            idx=row['index']; rr=np.asarray(spec['eigen_radius_frac'][idx]); xr=np.asarray(spec['eigen_radial'][idx]); xt=np.asarray(spec['eigen_tangential'][idx])
            ax.plot(rr,xr,label='radial'); ax.plot(rr,xt,label='tangential')
            if np.isfinite(rcc): ax.axvline(rcc,ls='--',alpha=.35,label='crust-core')
            if np.isfinite(rt1): ax.axvline(rt1,ls=':',alpha=.35,label='nt1')
            ax.set_title(f"{row['mode']}   {row['f_hz']:.2f} Hz   [{row['label']}]",fontsize=9)
            ax.set_xlabel('r / R'); ax.set_ylabel('normalised displacement'); ax.grid(True,alpha=.15)
        for ax in axs.ravel()[len(subset):]: ax.axis('off')
        axs[0,0].legend(fontsize=7,loc='best'); fig.tight_layout(); figs.append(fig)
    return figs


def forward_model(theta10=DEFAULT_THETA,mass_msun:float=1.4,family:int=1,inner=None,nt1:float=0.50,plots:bool=True,solve_modes:bool=True,nmax0:float=3.2,require_inner_core:bool=False):
    """Run one complete Neill/Newton-2026 parity star.

    With plots=True, diagnostics are emitted as soon as the corresponding
    stage has finished.  The expensive mode solve therefore does not hide the
    EOS/TOV figures until the very end.
    """
    t0=time.perf_counter()
    if inner is None:
        inner=sample_inner_source(family,nt1,np.random.default_rng(7))
    inner=np.asarray(inner,float)

    print('\n================ FORWARD MODEL ================')
    print(f'SOLVER VERSION: {SOLVER_VERSION}')
    print(f'family=F{family}  target mass={mass_msun:.3f} Msun  nt1={nt1:.4f} fm^-3')

    # Resolve enough high-density EOS to contain the true stable-branch Mmax.
    nmax=max(float(nmax0),float(nt1)+0.6)
    eos=mr=None
    for attempt in range(5):
        te=time.perf_counter()
        print(f'\n[1/4 EOS] building through n={nmax:.3f} fm^-3 ...', flush=True)
        eos=build_eos(theta10,family,inner,nt1,nmax)
        print(f'      n_drip={eos["cldm"]["ndrip"]:.6g}  n_1/8={eos["cldm"]["n18"]:.6g}  n_cc={eos["cldm"]["ncc"]:.6g}  n_cc(marched diag)={eos["cldm"]["ncc_marched"]:.6g}')
        pc=eos['parity']
        print(f'      v22 parity: drip in 2.0-3.5e-4 band={pc["ndrip_in_band"]}  G1(1e-4)={pc["G1_at_1e-4_fm3"]:.4f} (BPS 1.334, ok={pc["G1_plateau_ok"]})  ncc order ok={pc["ncc_order_ok"]}')
        print(f'      EOS build {time.perf_counter()-te:.2f} s', flush=True)

        tm=time.perf_counter()
        print('[2/4 TOV] resolving stable mass-radius branch ...', flush=True)
        mr=resolve_mr(eos)
        if mr.get('resolved',False):
            print(f'      Mmax={mr["Mmax"]:.4f} Msun at R={mr["Rmax"]:.3f} km  (2 Msun guard: {mr["Mmax"]>=2.0})')
            print(f'      M-R solve {time.perf_counter()-tm:.2f} s', flush=True)
            break
        print('      maximum-mass turnover not yet interior; extending high-density table', flush=True)
        nmax=min(MMAX_MAX_N,nmax*MMAX_EXTEND_FACTOR)
        if nmax>=MMAX_MAX_N-1e-10:
            raise ValueError('maximum-mass turnover not resolved before numerical emergency ceiling')
    if eos is None or mr is None or not mr.get('resolved',False):
        raise ValueError('maximum-mass turnover could not be resolved')

    figures=[]
    if plots:
        print('\n[DIAGRAMS] EOS and crust diagnostics', flush=True)
        for maker in (plot_bps_cldm_join, plot_cldm_microphysics, plot_eos_overview):
            fig=maker(eos); figures.append(fig); _show_figure(fig)
        fig=plot_mass_radius(mr,None); figures.append(fig); _show_figure(fig)

    tt=time.perf_counter()
    print(f'\n[3/4 STAR] inverting stable branch for {mass_msun:.3f} Msun ...', flush=True)
    Pc,R,M,inner_core_entered=invert_mass(mass_msun,eos,mr,require_inner_core=require_inner_core)
    star=_build_star_profile(eos,mr['t'],Pc)
    summary=dict(
        M_msun=M*C_CGS**2/(G_CGS*MSUN_CGS),
        R_km=R/1e5,
        Rcc_km=star['rcc']/1e5,
        Rt1_km=star['rt1']/1e5 if np.isfinite(star['rt1']) else np.nan,
        Pc_mevfm3=Pc/MEVFM3_TO_GEOM,
        Mmax_msun=mr['Mmax'],
        inner_core_entered=bool(inner_core_entered),
        parity=eos['parity'],
    )
    print(f'      M={summary["M_msun"]:.6f} Msun  R={summary["R_km"]:.4f} km')
    print(f'      Rcc={summary["Rcc_km"]:.4f} km  Rt1={summary["Rt1_km"]:.4f} km')
    print(f'      entered sampled inner core: {summary["inner_core_entered"]}')
    print(f'      target-star solve {time.perf_counter()-tt:.2f} s', flush=True)

    out=dict(eos=eos,mr=mr,star=star,summary=summary)
    if plots:
        print('\n[DIAGRAMS] target-star structure', flush=True)
        fig=plot_mass_radius(mr,star); figures.append(fig); _show_figure(fig)
        fig=plot_star_profiles(star); figures.append(fig); _show_figure(fig)

    if solve_modes:
        ts=time.perf_counter()
        print('\n[4/4 MODES] scanning, high-resolution refining, reconstructing ...', flush=True)
        mf,mv,met,fs,ds,ss,er,ex,et=_mode_solver(star)
        labels,iidx,class_conf,class_meta=classify_full_spectrum(mf,mv,met,er,ex,et,star,eos)
        rows=spectrum_rows(mf,mv,labels,met,class_conf,class_meta)
        slots=build_fixed_mode_slots(rows)
        spec=dict(
            frequencies_hz=np.asarray(mf),valid=np.asarray(mv),labels=labels,
            metrics=np.asarray(met),frequency_scan_hz=np.asarray(fs),
            determinant_scan=np.asarray(ds),singular_ratio_scan=np.asarray(ss),
            eigen_radius_frac=np.asarray(er),eigen_radial=np.asarray(ex),
            eigen_tangential=np.asarray(et),rows=rows,all_modes=rows,fixed_slots=slots,classification_confidence=class_conf,classification_metadata=class_meta,i_index=iidx,
            i_frequency_hz=(float(np.asarray(mf)[iidx]) if iidx is not None else np.nan),
        )
        out['spectrum']=spec
        print(f'      ALL residual-passing high-res roots={int(np.sum(spec["valid"]))}')
        print('      families:', {k:sum(r['label']==k for r in rows) for k in sorted(set(r['label'] for r in rows))})
        print(f'      fixed ML slots filled={int(np.sum(slots["mask"]))}/{len(slots["mask"])}; extra roots retained={sum(r["mode"] not in SLOT_NAMES for r in rows)}')
        print(f'      i-mode={spec["i_frequency_hz"]:.4f} Hz' if np.isfinite(spec['i_frequency_hz']) else '      i-mode not uniquely identified')
        print(f'      mode solve {time.perf_counter()-ts:.2f} s', flush=True)
        if plots:
            print('\n[DIAGRAMS] eigenvalue scan and solved modes', flush=True)
            fig=plot_scan(spec); figures.append(fig); _show_figure(fig)
            fig=plot_mode_spectrum(rows,title=f'M={summary["M_msun"]:.3f} Msun, R={summary["R_km"]:.2f} km, F{family}')
            figures.append(fig); _show_figure(fig)
            for fig in plot_all_mode_shapes(spec,star,modes_per_figure=6):
                figures.append(fig); _show_figure(fig)

    out['figures']=figures
    out['wall_seconds']=time.perf_counter()-t0
    print(f'\nDONE: total wall time {out["wall_seconds"]:.2f} s')
    return out


# ============================================================
# Fast unit/smoke checks that do not run the full high-resolution spectrum
# ============================================================
def self_test(verbose=True):
    tests={}
    tests['bps_reference_shape']=bool(len(_BPS_GAMMA_EQ)==43 and abs(BPS_DRIP_FM3-2.572e-4)<1e-7)
    # FOPT rapid map invariant, written algebraically.
    y=np.array([1.2,.7]); vh,vl=4.0,2.0; yo=np.array([y[1]+vh/vl*(y[0]-y[1]),y[1]]); tests['fopt_traction']=bool(abs(vh*(y[0]-y[1])-vl*(yo[0]-yo[1]))<1e-14 and yo[1]==y[1])
    # F5 VALUE-only continuity on a synthetic seam (v22 doc: slope deliberately free).
    p=np.array([.5,.8,.3,.2,0.,.6]); c1=float(_solve_f5_value(jnp.asarray(.35),jnp.asarray(.45),p)); pack=jnp.array([*p[:5],c1,p[5],0.])
    f0=float(_inner_raw_parity(5,jnp.asarray(.35),jnp.asarray(1.),jnp.asarray(1.),pack,jnp.asarray(.45),jnp.asarray(.35)))
    tests['f5_value_seam']=bool(abs(f0-.45)<1e-10)
    # Full-spectrum source-label regression using the previously validated 23-mode topology.
    rf=np.array([30.4165,33.8526,37.9771,43.1182,50.1043,60.3670,76.5587,104.6814,157.0835,223.1241,461.5971,628.0419,840.6660,1079.6318,1315.3532,1560.7129,1753.7722,1930.4633,2147.4495,2308.5438,2372.5677,2606.3037,2847.8056])
    nv=len(rf); rv=np.ones(nv,bool); rm=np.zeros((nv,11)); rm[:,6]=1e-6; rm[:,10]=1e-6; rm[:9,3]=np.arange(9,0,-1); rm[:9,1]=.1; rm[9,[0,1,8,9]]=[.8,.45,.9,.9]
    sidx=list(range(10,19))+list(range(20,23))
    for kk,jj in enumerate(sidx,1): rm[jj,1]=.9; rm[jj,4]=kk
    rm[19,1]=.08; rr=np.linspace(0,1,400); rer=np.tile(rr,(nv,1)); rex=np.zeros_like(rer); ret=np.zeros_like(rer)
    for jj in range(9): rex[jj]=np.sin((int(rm[jj,3])+1)*np.pi*(rr+.001))
    rex[9]=np.exp(-((rr-.92)/.04)**2); ret[9]=.6*np.exp(-((rr-.92)/.03)**2)
    for jj in sidx: rex[jj]=.05*np.sin(np.pi*rr); ret[jj]=np.sin((int(rm[jj,4])+1)*np.pi*np.clip((rr-.9)/.1,0,1))*(rr>.9)
    rex[19]=rr
    rl,ri,rc,rx=classify_full_spectrum(rf,rv,rm,rer,rex,ret,{'rstar':1.0,'rt1':np.nan},{'family':2}); rrows=spectrum_rows(rf,rv,rl,rm,rc,rx)
    expected=[f'g{k}' for k in range(9,0,-1)]+['i']+[f's{k}' for k in range(1,10)]+['f']+[f's{k}' for k in range(10,13)]
    tests['full_spectrum_labels']=([r['mode'] for r in rrows]==expected and ri==9)
    if verbose:
        for k,v in tests.items(): print(f"{k:24s}: {'PASS' if v else 'FAIL'}")
    if not all(tests.values()): raise AssertionError(tests)
    return tests


# ============================================================
# Batch production for the NN-placement experiment (v22 doc, second half)
# ============================================================
# Every sample stores ALL intermediate stages, so a surrogate can be trained
# to replace the pipeline at any cut:
#   theta -> EOS tables -> TOV background/star profile -> mode spectrum.
# One .npz per accepted sample; a CSV manifest records every draw, accepted
# or rejected-by-name, so the prior bookkeeping survives (v22 doc: a
# rejection the inference is not told about silently reweights the prior).

RESOLUTION_TIERS = {
    # locked V3 production gate
    'full':     dict(N_SCAN_CORE_LOG=161, N_SCAN_CORE_RAD=700, N_SCAN_CRUST=360,
                     N_EIGEN_CORE_LOG=3001, N_EIGEN_CORE_RAD=11032, N_EIGEN_CRUST=900,
                     N_FREQ=512, N_CANDIDATES=96, N_COARSE_MODES=56, N_MODES=48,
                     FINAL_RATIO_MAX=3.0e-5),
    # NN-training throughput tier; validate a subset against 'full' before trusting
    'training': dict(N_SCAN_CORE_LOG=121, N_SCAN_CORE_RAD=420, N_SCAN_CRUST=220,
                     N_EIGEN_CORE_LOG=1201, N_EIGEN_CORE_RAD=4400, N_EIGEN_CRUST=420,
                     N_FREQ=320, N_CANDIDATES=64, N_COARSE_MODES=48, N_MODES=40,
                     FINAL_RATIO_MAX=1.0e-4),
    # training grids with a wide root budget: the 40-root cap filled with g-rungs on dense ladders and never reached i/s/f
    'training_wide': dict(N_SCAN_CORE_LOG=121, N_SCAN_CORE_RAD=420, N_SCAN_CRUST=220,
                     N_EIGEN_CORE_LOG=1201, N_EIGEN_CORE_RAD=4400, N_EIGEN_CRUST=420,
                     N_FREQ=1024, N_CANDIDATES=192, N_COARSE_MODES=176, N_MODES=160,
                     FINAL_RATIO_MAX=1.0e-4),
    # smoke/debug only
    'smoke':    dict(N_SCAN_CORE_LOG=81, N_SCAN_CORE_RAD=240, N_SCAN_CRUST=140,
                     N_EIGEN_CORE_LOG=301, N_EIGEN_CORE_RAD=1100, N_EIGEN_CRUST=240,
                     N_FREQ=192, N_CANDIDATES=48, N_COARSE_MODES=40, N_MODES=32,
                     FINAL_RATIO_MAX=5.0e-4),
}

def set_resolution(tier: str) -> None:
    """Set the mode-solver grids once per process (changing them re-triggers
    XLA compilation, so pick the tier before the first star)."""
    for k, v in RESOLUTION_TIERS[tier].items():
        globals()[k] = v

# Operational sampling ensemble for the ten empirical parameters.
# Lower orders: uniform laboratory bands.  The fourth-order pair follows the
# stable normal distributions the 2026 supplement quotes (Z0 = 1306 +/- 214,
# Zsym = -2317 +/- 379 MeV), clipped at 3 sigma; an independent broad uniform
# box there builds unphysical matter and rejects nearly every draw.
# theta order: E0,K0,Q0,Z0,J,L,Ksym,Qsym,Zsym,ns.
THETA_RANGES = dict(
    E0=(-16.5, -15.5), K0=(190.0, 270.0), Q0=(-600.0, 600.0),
    J=(28.0, 36.0), L=(30.0, 80.0), Ksym=(-250.0, 50.0), Qsym=(-1000.0, 1000.0),
    ns=(0.15, 0.17),
)
THETA_NORMALS = dict(Z0=(1306.0, 214.0), Zsym=(-2317.0, 379.0))
THETA_ORDER = ('E0','K0','Q0','Z0','J','L','Ksym','Qsym','Zsym','ns')


def _draw_theta(rng: np.random.Generator) -> np.ndarray:
    vals = {}
    for k, (lo, hi) in THETA_RANGES.items():
        vals[k] = rng.uniform(lo, hi)
    for k, (mu, sd) in THETA_NORMALS.items():
        vals[k] = float(np.clip(rng.normal(mu, sd), mu - 3 * sd, mu + 3 * sd))
    return np.array([vals[k] for k in THETA_ORDER])


def sample_draw(rng: np.random.Generator):
    theta = _draw_theta(rng)
    family = int(rng.integers(1, 6))
    nt1 = float(rng.uniform(0.24, 0.48 if family == 5 else 0.80))
    inner = sample_inner_source(family, nt1, rng)
    mass = float(rng.uniform(1.0, 3.0))
    return theta, family, inner, nt1, mass


def _ds(a, k):
    a = np.asarray(a)
    step = max(1, a.shape[-1] // k) if a.ndim > 1 else max(1, len(a) // k)
    return a[..., ::step] if a.ndim > 1 else a[::step]


def _partial_payload(theta, family, inner, nt1, mass, seed, idx, eos, mr, status):
    """What a rejected draw still knows once its EOS exists: the tables, transitions and the coarse M-R sequence.
    Kept so that the EOS seat, the gatekeeper's Mmax head and the failure taxonomy can learn from failures too."""
    oc = np.asarray(eos['outer_core']); ic = np.asarray(eos['inner_core']); cr = eos['crust']; c = eos['cldm']
    return dict(theta=theta, family=family, inner=np.asarray(inner, float), nt1=nt1, mass_target=mass, seed=seed, index=idx,
                solver_version=SOLVER_VERSION, status=status,
                crust_n=np.asarray(cr['n']), crust_eps=np.asarray(cr['eps']), crust_P=np.asarray(cr['P']),
                crust_G1=np.asarray(cr['G1']), crust_Geq=np.asarray(cr['Geq']), crust_mu=np.asarray(cr['mu']),
                outer_core=oc, inner_core=ic, transitions=np.asarray(eos['transitions']), n_ac=eos['n_ac'],
                cldm_A=np.asarray(c['A']), cldm_Z=np.asarray(c['Z']), cldm_Xn=np.asarray(c['Xn']), cldm_u=np.asarray(c['u']),
                ncc_marched=c['ncc_marched'],
                parity=np.asarray([eos['parity'][k] for k in ('ndrip_fm3','G1_at_1e-4_fm3','ncc_extrapolated_fm3','ncc_marched_fm3')], float),
                mr_pcs=np.asarray(mr.get('pcs', [])), mr_R_km=np.asarray(mr.get('R', []), float) / 1e5, 
                mr_M_msun=np.asarray(mr.get('M', []), float) * C_CGS**2 / (G_CGS * MSUN_CGS), mr_ok=np.asarray(mr.get('ok', []), bool),
                Mmax=float(mr.get('Mmax', np.nan)), Rmax_km=float(mr.get('Rmax', np.nan)))


def run_one_sample(idx: int, seed: int, require_inner_core: bool = False):
    """One draw through the whole pipeline.  Returns (status, payload_or_None, row)."""
    rng = np.random.default_rng(seed + idx)
    theta, family, inner, nt1, mass = sample_draw(rng)
    row = dict(index=idx, status='', family=family, nt1=nt1, mass_target=mass,
               M=np.nan, R_km=np.nan, Mmax=np.nan, ndrip=np.nan, ncc=np.nan,
               i_hz=np.nan, n_roots=0, wall_s=np.nan,
               theta='|'.join(f'{x:.4g}' for x in theta))
    t0 = time.perf_counter(); eos = None; mr = None
    try:
        te = time.perf_counter()
        eos, mr = build_eos_and_resolve_mmax(theta, family, inner, nt1)
        t_eos = time.perf_counter() - te
        row['ndrip'] = eos['cldm']['ndrip']; row['ncc'] = eos['cldm']['ncc']; row['Mmax'] = mr['Mmax']
        if mr['Mmax'] < 2.0:
            row['status'] = 'reject:mmax_lt_2'; row['wall_s'] = time.perf_counter() - t0
            return 'reject', _partial_payload(theta, family, inner, nt1, mass, seed, idx, eos, mr, row['status']), row
        if mass > mr['Mmax']:
            row['status'] = 'reject:mass_gt_mmax'; row['wall_s'] = time.perf_counter() - t0
            return 'reject', _partial_payload(theta, family, inner, nt1, mass, seed, idx, eos, mr, row['status']), row
        tt = time.perf_counter()
        Pc, R, M, entered = invert_mass(mass, eos, mr, require_inner_core=require_inner_core)
        star = _build_star_profile(eos, mr['t'], Pc)
        t_star = time.perf_counter() - tt
        tm = time.perf_counter()
        mf, mv, met, fs, ds_, ss, er, ex, et = _mode_solver(star)
        labels, iidx, conf, meta = classify_full_spectrum(mf, mv, met, er, ex, et, star, eos)
        rows = spectrum_rows(mf, mv, labels, met, conf, meta)
        slots = build_fixed_mode_slots(rows)
        t_modes = time.perf_counter() - tm
        Msun = M * C_CGS**2 / (G_CGS * MSUN_CGS)
        row.update(status='ok', M=Msun, R_km=R / 1e5, n_roots=int(np.sum(np.asarray(mv))),
                   i_hz=(float(np.asarray(mf)[iidx]) if iidx is not None else np.nan),
                   wall_s=time.perf_counter() - t0)
        oc = np.asarray(eos['outer_core']); ic = np.asarray(eos['inner_core']); cr = eos['crust']; c = eos['cldm']
        payload = dict(
            # stage 0: inputs
            theta=theta, family=family, inner=np.asarray(inner, float), nt1=nt1,
            mass_target=mass, seed=seed, index=idx, solver_version=SOLVER_VERSION,
            # stage 1: EOS tables and transitions
            crust_n=np.asarray(cr['n']), crust_eps=np.asarray(cr['eps']), crust_P=np.asarray(cr['P']),
            crust_G1=np.asarray(cr['G1']), crust_Geq=np.asarray(cr['Geq']), crust_mu=np.asarray(cr['mu']),
            outer_core=oc, inner_core=ic,
            transitions=np.asarray(eos['transitions']), n_ac=eos['n_ac'],
            cldm_A=np.asarray(c['A']), cldm_Z=np.asarray(c['Z']), cldm_Xn=np.asarray(c['Xn']), cldm_u=np.asarray(c['u']),
            ncc_marched=c['ncc_marched'],
            parity=np.asarray([eos['parity'][k] for k in ('ndrip_fm3','G1_at_1e-4_fm3','ncc_extrapolated_fm3','ncc_marched_fm3')], float),
            # stage 2: TOV / star background (downsampled to <=800 rows)
            Mmax=mr['Mmax'], Rmax_km=mr['Rmax'], Pc=Pc, M_msun=Msun, R_km=R / 1e5,
            Rcc_km=star['rcc'] / 1e5, Rt1_km=(star['rt1'] / 1e5 if np.isfinite(star['rt1']) else np.nan),
            inner_core_entered=bool(entered),
            star_r=_ds(star['r'], 800), star_m=_ds(star['m'], 800), star_P=_ds(star['P'], 800),
            star_eps=_ds(star['eps'], 800), star_n=_ds(star['n'], 800), star_mu=_ds(star['mu'], 800),
            star_G1=_ds(star['G1'], 800), star_Geq=_ds(star['Geq'], 800),
            star_nu=_ds(star['nu'], 800), star_lam=_ds(star['lam'], 800),
            # stage 3: spectrum (target) plus eigenfunction morphology (<=512 cols)
            mode_freqs=np.asarray(mf), mode_valid=np.asarray(mv), mode_labels=np.asarray(labels),
            mode_metrics=np.asarray(met),
            mode_names=np.asarray([r['mode'] for r in rows], dtype='U16'),
            mode_row_freqs=np.asarray([r['f_hz'] for r in rows]),
            slot_names=slots['names'], slot_freqs=slots['frequencies_hz'], slot_mask=slots['mask'],
            i_index=(-1 if iidx is None else int(iidx)), i_hz=row['i_hz'],
            scan_f=np.asarray(fs), scan_sr=np.asarray(ss),
            eig_r=_ds(np.asarray(er), 512), eig_xr=_ds(np.asarray(ex), 512), eig_xt=_ds(np.asarray(et), 512),
            timings=np.asarray([t_eos, t_star, t_modes]),
        )
        return 'ok', payload, row
    except Exception as exc:
        row['status'] = ('reject:' + str(exc))[:160]
        row['wall_s'] = time.perf_counter() - t0
        partial = None
        if eos is not None and mr is not None:
            try: partial = _partial_payload(theta, family, inner, nt1, mass, seed, idx, eos, mr, row['status'])
            except Exception: partial = None
        return 'reject', partial, row


def run_batch(n_samples: int, out_dir: str, seed: int = 0, tier: str = 'training',
              require_inner_core: bool = False, start: int = 0, indices=None):
    """Produce n_samples draws with full intermediate data.  Resume-safe:
    existing sample files are skipped, and the manifest is append-only."""
    import os, csv
    set_resolution(tier)
    os.makedirs(out_dir, exist_ok=True)
    manifest = os.path.join(out_dir, 'manifest.csv')
    fields = ['index','status','family','nt1','mass_target','M','R_km','Mmax','ndrip','ncc','i_hz','n_roots','wall_s','theta']
    newfile = not os.path.exists(manifest)
    print(f'[batch] tier={tier} seed={seed} out={out_dir} n={n_samples} start={start}', flush=True)
    n_ok = n_rej = 0
    with open(manifest, 'a', newline='') as mf_:
        w = csv.DictWriter(mf_, fieldnames=fields)
        if newfile: w.writeheader()
        for idx in (list(indices) if indices is not None else range(start, start + n_samples)):
            fn = os.path.join(out_dir, f'sample_{idx:06d}.npz')
            if os.path.exists(fn):
                print(f'[batch] {idx}: exists, skipping', flush=True); continue
            status, payload, row = run_one_sample(idx, seed, require_inner_core)
            if status == 'ok':
                np.savez_compressed(fn, **payload); n_ok += 1
            elif payload is not None:   # rejected AFTER the EOS was built: keep the EOS tables and the M-R sequence
                np.savez_compressed(os.path.join(out_dir, f'reject_{idx:06d}.npz'), **payload)
            else:
                n_rej += 1
            w.writerow({k: row.get(k) for k in fields}); mf_.flush()
            print(f'[batch] {idx}: {row["status"]}  M={row["M"]:.3f} R={row["R_km"]:.2f} '
                  f'i={row["i_hz"]:.1f}Hz roots={row["n_roots"]} ({row["wall_s"]:.1f}s)'
                  if status=='ok' else f'[batch] {idx}: {row["status"]} ({row["wall_s"]:.1f}s)', flush=True)
    print(f'[batch] done: ok={n_ok} rejected={n_rej}', flush=True)
    return n_ok, n_rej


if __name__ == '__main__':
    import argparse, sys
    ap = argparse.ArgumentParser(description='Neill/Newton 2026 parity forward model (v22-doc)')
    ap.add_argument('--batch', type=int, default=0, help='number of samples to produce (0 = run the reference demo star)')
    ap.add_argument('--out', type=str, default='fm3_dataset', help='output directory for batch samples')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--start', type=int, default=0, help='first sample index (for sharding across processes/GPUs)')
    ap.add_argument('--tier', type=str, default='training', choices=list(RESOLUTION_TIERS))
    ap.add_argument('--require-inner-core', action='store_true', help='apply the 2026 G5 guard (reject stars that never reach nt1)')
    ap.add_argument('--jaxcache', type=str, default='', help='persistent XLA compilation cache directory')
    ap.add_argument('--indices', type=str, default='', help='file of sample indices to (re)solve, one per line; overrides --batch/--start')
    ap.add_argument('--no-plots', action='store_true')
    args = ap.parse_args()
    if args.jaxcache:
        enable_persistent_jax_cache(args.jaxcache)
    self_test()
    print(f'\n{SOLVER_VERSION}')
    if args.indices:
        idxs=[int(x) for x in open(args.indices).read().split()]
        run_batch(len(idxs), args.out, seed=args.seed, tier=args.tier,
                  require_inner_core=args.require_inner_core, start=args.start, indices=idxs)
        sys.exit(0)
    if args.batch > 0:
        run_batch(args.batch, args.out, seed=args.seed, tier=args.tier,
                  require_inner_core=args.require_inner_core, start=args.start)
        sys.exit(0)
    print('Safety checks passed. Starting the reference 1.4 Msun full-spectrum star now.')
    theta_demo=np.array([-16.,230.,-300.,500.,32.,60.,-100.,300.,-500.,0.16])
    inner_demo=np.array([0.10,0.30,2.0])
    demo=forward_model(
        theta10=theta_demo,
        mass_msun=1.4,
        family=1,
        inner=inner_demo,
        nt1=0.50,
        plots=not args.no_plots,
        solve_modes=True,
        require_inner_core=False,
    )
    print('\nSUMMARY:',demo['summary'])
    if 'spectrum' in demo:
        print('i-mode =',demo['spectrum']['i_frequency_hz'],'Hz')

