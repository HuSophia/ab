"""Global fine balancing with a fixed raw coarse allocation; analytic mixtures."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.optimize import minimize
from scipy.special import logsumexp, xlogy
from scipy.stats import t as student_t

from eval_hierarchy import analytic, plans, score_groups
from evaluate import integrated_decay


def feasible_round(p, a, b):
    """Nonnegative rounding onto two prescribed marginals of equal mass."""
    p = np.maximum(0., p).copy()
    row = p.sum(1)
    p *= np.minimum(1, np.divide(a,row,out=np.ones_like(a),where=row>0))[:,None]
    col = p.sum(0)
    p *= np.minimum(1, np.divide(b,col,out=np.ones_like(b),where=col>0))
    da, db = np.maximum(0,a-p.sum(1)), np.maximum(0,b-p.sum(0))
    if db.sum() > 1e-30:
        p += np.outer(da, db/db.sum())
    return p


def block_masses(p,I,J):
    return np.array([[p[np.ix_(i,j)].sum() for j in J] for i in I])


def objective(p, logk):
    return float(np.sum(xlogy(p,p)-p*logk))


def solve_block(logk, a, b, I, J, q, incumbent):
    """Reduced-dual solve with fixed block masses and exact nested rounding.

    For each positive block the potential w_AB is eliminated analytically.
    The resulting convex objective involves only fine row/column potentials.
    Restoring all primal constraints supplies an upper bound while the dual
    stays a lower bound, even if the iteration ends before convergence.
    """
    N,M=logk.shape
    np.testing.assert_allclose(q.sum(1),[a[i].sum() for i in I],atol=5e-13)
    np.testing.assert_allclose(q.sum(0),[b[j].sum() for j in J],atol=5e-13)
    np.testing.assert_allclose(block_masses(incumbent,I,J),q,atol=5e-13)
    np.testing.assert_allclose(incumbent.sum(1),a,atol=5e-13)
    np.testing.assert_allclose(incumbent.sum(0),b,atol=5e-13)
    free_u=np.sort(np.concatenate([i[1:] for i in I]))
    free_v=np.sort(np.concatenate([j[1:] for j in J]))
    count=len(free_u)
    blocks=[(alpha,beta,i,j,np.ix_(i,j),logk[np.ix_(i,j)],q[alpha,beta])
            for alpha,i in enumerate(I) for beta,j in enumerate(J) if q[alpha,beta]>0]

    def evaluate(z, eps, want_plan=False):
        u,v=np.zeros(N),np.zeros(M)
        u[free_u]=z[:count]
        v[free_v]=z[count:]
        ru,rv=np.zeros(N),np.zeros(M)
        dual=a@u+b@v
        p=np.zeros_like(logk) if want_plan else None
        for _,_,i,j,ix,k,mass in blocks:
            ell=(k+u[i,None]+v[j][None,:])/eps
            norm=logsumexp(ell)
            zloc=mass*np.exp(ell-norm)
            dual += eps*mass*(np.log(mass)-norm)
            ru[i]+=zloc.sum(1)
            rv[j]+=zloc.sum(0)
            if want_plan:
                p[ix]=zloc
        # Maximize dual, with primal objective normalized to unit mass.
        if want_plan:
            return p,float(dual),max(np.abs(ru-a).sum(),np.abs(rv-b).sum())
        return -dual,np.r_[ru[free_u]-a[free_u],rv[free_v]-b[free_v]]

    def make_feasible(p):
        # Allocate its fine row mass across target blocks and its fine column
        # mass across source blocks. Round these auxiliary transport matrices
        # to match both the fine marginal and the fixed coarse masses.
        row_alloc=np.stack([p[:,j].sum(1) for j in J],axis=1)
        col_alloc=np.stack([p[i,:].sum(0) for i in I],axis=0)
        for alpha,i in enumerate(I):
            row_alloc[i]=feasible_round(row_alloc[i],a[i],q[alpha])
        for beta,j in enumerate(J):
            col_alloc[:,j]=feasible_round(col_alloc[:,j],q[:,beta],b[j])
        fixed=np.zeros_like(p)
        for alpha,beta,i,j,ix,_,_ in blocks:
            fixed[ix]=feasible_round(p[ix],row_alloc[i,beta],col_alloc[alpha,j])
        return fixed

    def error(p):
        return float(max(np.abs(p.sum(1)-a).sum(),np.abs(p.sum(0)-b).sum(),
                         np.abs(block_masses(p,I,J)-q).sum()))

    start=perf_counter()
    z=np.zeros(count+len(free_v))
    nfe=0
    # Start at softer temperatures only for strongly concentrated kernels.
    max_eps=max(1.,min(64.,np.ptp(logk)/50))
    temperatures=np.unique(np.geomspace(1.,max_eps,5))[::-1]
    if z.size:
        for eps in temperatures:
            sol=minimize(evaluate,z,args=(eps,),jac=True,method="L-BFGS-B",
                         options={"gtol":1e-12 if eps==1 else 1e-8,
                                  "ftol":1e-15 if eps==1 else 1e-11,
                                  "maxiter":4000 if eps==1 else 250,
                                  "maxcor":25,"maxls":50})
            z=sol.x
            nfe+=int(sol.nfev)
    raw,lo,res=evaluate(z,1.,True)
    cand=make_feasible(raw)
    if error(cand)>5e-11 or (cand<0).any():
        raise RuntimeError(f"Nested block rounding could not restore feasibility: {error(cand):.3g}")
    before=objective(incumbent,logk)
    cand_val=objective(cand,logk)
    chosen=cand if cand_val<before else incumbent.copy()
    hi=min(before,cand_val)
    gap=hi-lo
    if gap < -1e-8:
        raise RuntimeError(f"Negative primal-dual gap: {gap:.3g}")
    return chosen,dict(
        seconds=perf_counter()-start,nfe=nfe,raw_marginal_l1=float(res),
        feasible_l1=error(chosen),rounding_l1=float(np.abs(cand-raw).sum()),
        primal=hi,dual=lo,gap=max(0.,gap),candidate_primal=cand_val,
        candidate_plan_l1_to_incumbent=float(np.abs(cand-incumbent).sum()),
        incumbent_primal=before,used_incumbent=bool(cand_val>=before),
        certified_gap_below_1e_5=bool(max(0.,gap)<1e-5),
    )


def run(input_path,out):
    with np.load(input_path,allow_pickle=False) as f:
        keys="source target mu rates sigma forcing waves test_times test_paths".split()
        x,y,mu,r,s,force,w,t,truth=[np.asarray(f[k],float) for k in keys]
    assert all(np.isfinite(z).all() for z in (x,y,mu,r,s,force,w,t,truth))
    assert len(x)>=4 and len(y)>=4 and len(truth)>=5 and len(t)>=4 and min(s)>0 and t[0]==0.
    assert np.all(np.diff(t)>0) and np.all(r>=0.) and (np.square(w).sum(1)>0).all()
    active=np.abs(force)>1e-8*np.max(np.abs(force))
    assert active.any()
    scale=np.maximum(np.std(np.concatenate([x,y]),axis=0),1e-4)
    scale=np.maximum(scale,.05*np.median(scale))
    original,info=plans(x,y,mu,r,s,t[-1],scale)
    old=original["raw_hierarchy"]; flat=original["flat"]
    _,il=kmeans2((x-mu)/scale,4,minit="points",seed=0)
    _,jl=kmeans2((y-mu)/scale,4,minit="points",seed=1)
    I=[np.flatnonzero(il==i) for i in np.unique(il)]
    J=[np.flatnonzero(jl==j) for j in np.unique(jl)]
    a,b=np.full(len(x),1/len(x)),np.full(len(y),1/len(y))
    g=np.exp(-r*t[-1])
    qT=s**2*integrated_decay(t[-1],2*r)
    logk=((x-x.mean(0))*(g/qT))@(y-y.mean(0)).T
    raw_m,flat_m=block_masses(old,I,J),block_masses(flat,I,J)
    repaired,report=solve_block(logk,a,b,I,J,raw_m,old)
    oracle_exact,oracle=solve_block(logk,a,b,I,J,flat_m,flat)
    # Retain the actual solve's discrepancy even when the feasible incumbent
    # is tighter and returned. The logs identify which plan was used.
    feasibility=np.abs(block_masses(repaired,I,J)-raw_m).sum()
    flat_base=objective(flat,logk)
    old_obj=objective(old,logk)
    coarse_kl=float(np.sum(xlogy(raw_m,raw_m)-xlogy(raw_m,flat_m)))
    if not np.isfinite(coarse_kl):
        coarse_kl=None  # Numerical zeros in projected flat, not a finite certificate.
    comp={"old_proportional":old,"global_relaxed":repaired,"flat":flat}
    report["coarse_kl_estimate"]=coarse_kl
    report["flat_objective"]=flat_base
    report["original_info"]=info
    report["projected_flat_oracle"]=oracle
    report["projected_flat_returned_l1"]=float(np.abs(oracle_exact-flat).sum())
    report["original_block_mass_l1_residual"]=float(feasibility)
    report["raw_hierarchy_objective_minus_flat"]=old_obj-flat_base
    report["relaxed_objective_minus_flat"]=report["primal"]-flat_base
    report["relaxed_objective_minus_old"]=report["primal"]-old_obj
    predictions={label:analytic(x,y,mu,r,s,t,p) for label,p in comp.items()}
    for label in predictions:
        np.testing.assert_allclose(predictions[label][0],predictions["flat"][0],atol=1e-12)
        np.testing.assert_allclose(predictions[label][1][[0,-1]],predictions["flat"][1][[0,-1]],atol=1e-8,rtol=1e-10)
    scores={label:score_groups(truth,scale,w,active,p) for label,p in predictions.items()}
    summary=[]
    G=len(truth)
    for label,comparator in (("global_relaxed","old_proportional"),("old_proportional","flat"),("global_relaxed","flat")):
        for key in scores["flat"][0]:
            u=np.array([row[key] for row in scores[label]])
            v=np.array([row[key] for row in scores[comparator]])
            delta=u-v
            hw=student_t.ppf(.975,G-1)*delta.std(ddof=1)/np.sqrt(G)
            summary.append(dict(arm=label,comparator=comparator,metric=key,
                                arm_score=u.mean(),comparator_score=v.mean(),
                                delta=delta.mean(),low=delta.mean()-hw,high=delta.mean()+hw))
    out.mkdir(parents=True,exist_ok=True)
    with (out/"summary.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=summary[0])
        writer.writeheader()
        writer.writerows(summary)
    (out/"metadata.json").write_text(json.dumps(report,indent=2)+"\n")
    lines=[
        "# APEBench fixed-coarse global-balancing diagnostic","",
        "This mechanism check reuses the hierarchy run's split, holdout groups and train-mean-centered OU reference. "
        "The original proportional lift and globally balanced variant share the same raw coarse block allocation "
        "and fine endpoint marginals. Only the proportional per-block fine-marginal restrictions were removed. "
        "All scores use analytic mixtures, not neural rollouts.","",
        f"Endpoint populations: {len(x)} and {len(y)}. Held-out groups: {G}, {truth.shape[1]} trajectories each. "
        "Adjacent interior joint cross-covariance errors use training-only scale normalization. "
        "Paired 95% t intervals vary held-out groups while holding the train split fixed. Lower is better; "
        "zero terminal contrasts are an endpoint-matching check.","",
        "| Arm | Comparator | Metric | Arm | Comparator | Paired difference | 95% lower | 95% upper |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    lines+=["| {arm} | {comparator} | {metric} | {arm_score:.6g} | {comparator_score:.6g} | "
            "{delta:+.6g} | {low:+.6g} | {high:+.6g} |".format(**row) for row in summary]
    lines+=["",f"Fine objective minus the flat solution: proportional {old_obj-flat_base:.6g}, "
            f"global balancing {report['relaxed_objective_minus_flat']:.6g}. "
            f"Global balancing minus proportional: {report['relaxed_objective_minus_old']:+.6g}. "
            f"Raw-projection coarse KL estimate: {coarse_kl if coarse_kl is not None else 'not finite numerically'}.","",
            f"Fixed-coarse primal/dual gap: {report['gap']:.3g}; maximum constraint-family L1 residual: "
            f"{report['feasible_l1']:.3g}. Original and globally-balanced raw block allocations agree to "
            f"{feasibility:.3g} in L1. The relaxed feasible objective is guaranteed no larger than the original "
            "proportional lift; equivalence holds only if its additional constraints are already met by the optimum.","",
            f"Projected-flat allocation oracle: dual gap {oracle['gap']:.3g}, rounded candidate L1 distance to "
            f"the flat plan {oracle['candidate_plan_l1_to_incumbent']:.3g}. "
            f"Oracle used incumbent flat plan: {oracle['used_incumbent']}. This is a correctness check, not a deployable baseline.","",
            f"One-run CPU timings: flat construction {info['flat_seconds']:.4g} s; old dense hierarchy construction "
            f"{info['hierarchy_seconds']:.4g} s; additional fixed-block global solve {report['seconds']:.4g} s. "
            "No speedup claim is made.",""]
    (out/"report.md").write_text("\n".join(lines))
    print("Block relaxation diagnostic complete.")
    return report,summary


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    arg=p.parse_args()
    run(arg.input,arg.out)
