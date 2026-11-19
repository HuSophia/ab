"""Fixed-reference, analytic flat versus block-lift bridge diagnostic."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.special import logsumexp, xlogy
from scipy.stats import t as student_t

from evaluate import flat_plan, integrated_decay


def coarse_sinkhorn(logk, a, b):
    """Stable small-matrix Sinkhorn with feasible post-processing."""
    u, v = np.zeros(len(a)), np.zeros(len(b))
    start = max(1.0, np.ptp(logk) / 10)
    old = start
    for temperature in np.geomspace(start, 1.0, 8):
        u *= old / temperature
        v *= old / temperature
        kernel = logk / temperature
        for j in range(10000 if temperature == 1 else 1000):
            u = np.log(a) - logsumexp(kernel + v[None, :], axis=1)
            v = np.log(b) - logsumexp(kernel + u[:, None], axis=0)
            if j % 25 == 0:
                pi = np.exp(kernel + u[:, None] + v)
                tol = 1e-10 if temperature == 1 else 1e-5
                if np.sum(np.abs(pi.sum(1) - a)) < tol:
                    break
        old = temperature
    pi = np.exp(logk + u[:, None] + v)
    raw_res = max(np.abs(pi.sum(1) - a).sum(), np.abs(pi.sum(0) - b).sum())
    dual = a @ u + b @ v - pi.sum()
    pi *= np.minimum(1, a / pi.sum(1))[:, None]
    pi *= np.minimum(1, b / pi.sum(0))
    da, db = a - pi.sum(1), b - pi.sum(0)
    if da.sum() > 1e-15:
        pi += np.outer(np.maximum(0, da), np.maximum(0, db)) / da.sum()
    np.testing.assert_allclose(pi.sum(1), a, atol=1e-12)
    np.testing.assert_allclose(pi.sum(0), b, atol=1e-12)
    primal = np.sum(xlogy(pi, pi) - pi * (logk + 1))
    return pi, float(raw_res), max(0., float(primal - dual))


def plans(x, y, mu, rates, sigma, T, scale):
    """Same dense raw-macro lift as the learned experiment, plus coarse oracle."""
    g = np.exp(-rates*T)
    q = sigma**2 * integrated_decay(T, 2*rates)
    start = perf_counter()
    flat, info = flat_plan(x,y,g,q)
    flat_time = perf_counter()-start
    start = perf_counter()
    _, i_label = kmeans2((x-mu)/scale,4,minit="points",seed=0)
    _, j_label = kmeans2((y-mu)/scale,4,minit="points",seed=1)
    I = [np.flatnonzero(i_label == i) for i in np.unique(i_label)]
    J = [np.flatnonzero(j_label == j) for j in np.unique(j_label)]
    if len(I) < 2 or len(J) < 2:
        raise RuntimeError("Fewer than two nonempty clusters on a side")
    # Constants marginalized by the flat fine solve must not be dropped
    # before this raw coarse log-mean-exp operation.
    K = -.5*np.square(((x-mu)*g)[:,None,:]-(y-mu)[None,:,:]) @ (1/q)
    raw_coarse = np.empty((len(I),len(J)))
    projected = np.empty_like(raw_coarse)
    for ai,i in enumerate(I):
        for bj,j in enumerate(J):
            ij=np.ix_(i,j)
            raw_coarse[ai,bj] = logsumexp(K[ij])-np.log(len(i)*len(j))
            projected[ai,bj] = flat[ij].sum()
    macro, macro_res, macro_gap = coarse_sinkhorn(raw_coarse,
            np.array([len(i) for i in I])/len(x),
            np.array([len(j) for j in J])/len(y))
    raw, conditional_only = np.zeros_like(K), np.zeros_like(K)
    local_raw_max = local_gap_max = 0.
    for ai,i in enumerate(I):
        for bj,j in enumerate(J):
            local, diag = flat_plan(x[i],y[j],g,q)
            local_raw_max = max(local_raw_max,diag["raw_marginal_l1"])
            local_gap_max = max(local_gap_max,diag["duality_gap"])
            ij=np.ix_(i,j)
            raw[ij] = macro[ai,bj]*local
            conditional_only[ij] = projected[ai,bj]*local
    raw_time = perf_counter()-start
    xc,yc=x-x.mean(0),y-y.mean(0)
    canonical=(xc*(g/q))@yc.T
    obj=lambda p: float(np.sum(xlogy(p,p)-p*canonical))
    controls={"flat":flat,"raw_hierarchy":raw,"projected_flat_macro_control":conditional_only}
    meta={
      "flat_seconds":flat_time,"hierarchy_seconds":raw_time,
      "flat_solver":info,"macro_raw_marginal_l1":macro_res,"macro_duality_gap":macro_gap,
      "worst_local_raw_marginal_l1":local_raw_max,"worst_local_duality_gap":local_gap_max,
      "cluster_counts": [len(I),len(J)],
      "macro_l1_distance":float(np.abs(macro-projected).sum()),"plans":{}
    }
    for label,p in controls.items():
        res=max(np.abs(p.sum(1)-1/len(x)).sum(),np.abs(p.sum(0)-1/len(y)).sum())
        assert res < 1e-10
        meta["plans"][label]={"feasible_l1":float(res),"objective_minus_flat":obj(p)-obj(flat),
                               "plan_l1_vs_flat":float(np.abs(p-flat).sum())}
    for label, p, target in (("raw",raw,macro),("projected",conditional_only,projected)):
        reconstructed=np.array([[p[np.ix_(i,j)].sum() for j in J] for i in I])
        np.testing.assert_allclose(reconstructed,target,atol=1e-12)
    return controls,meta


def analytic(x,y,mu,r,s,t,p):
    """Marginal moments and adjacent cross-covariances of the exact mixture."""
    T=t[-1]; u=t[:,None]
    Q=s**2*integrated_decay(u,2*r)
    Qu=s**2*integrated_decay(T-u,2*r)
    b=Q*np.exp(-r*(T-u))/Q[-1]
    a=np.exp(-r*u)-b*np.exp(-r*T)
    m=mu+a*(x.mean(0)-mu)+b*(y.mean(0)-mu)
    xc=x-x.mean(0); yc=y-y.mean(0)
    cxx=xc.T@xc/len(x); cyy=yc.T@yc/len(y)
    cxy=xc.T@p@yc
    def pair(i,j):
        assert i<=j
        lin=(a[i,:,None]*cxx*a[j,None,:]+b[i,:,None]*cyy*b[j,None,:]
             +a[i,:,None]*cxy*b[j,None,:]+b[i,:,None]*cxy.T*a[j,None,:])
        return lin+np.diag(Q[i]*np.exp(-r*(t[j]-t[i]))*Qu[j]/Q[-1])
    covariance=np.stack([pair(i,i) for i in range(len(t))])
    pair_ids=list(zip(range(1,len(t)-2),range(2,len(t)-1)))
    lag=np.stack([pair(i,j) for i,j in pair_ids])
    return m,covariance,lag,pair_ids


def score_groups(truth, scale, waves, active, prediction):
    means,covs,lag,ij=prediction
    pred_second=means**2+np.diagonal(covs,axis1=1,axis2=2)
    k2=np.square(waves).sum(1)
    shells=np.unique(k2)
    inv_scale=1/np.outer(scale,scale)
    band=np.any(np.all(waves[:,None,:]==waves[active][None,:,:],axis=-1),axis=1)
    assert band.any()
    weights=np.stack([(k2==k)/k2 for k in shells],axis=1)
    scores=[]
    for data in truth:
        c=data-data.mean(0)
        actual_lag=np.stack([c[:,i,:].T@c[:,j,:]/len(c) for i,j in ij])
        err=(lag-actual_lag)*inv_scale
        actual_shell=np.mean(data**2,axis=0)@weights
        pred_shell=pred_second@weights
        relative=np.sum(np.abs(pred_shell-actual_shell),axis=1)/actual_shell.sum(1)
        scores.append({
          "interior_cross_cov_rmse":float(np.sqrt(np.mean(err**2))),
          "interior_forced_band_cross_cov_rmse":float(np.sqrt(np.mean(err[:,band,:][:,:,band]**2))),
          "interior_retained_shell_energy_rel_l1":float(relative[1:-1].mean()),
          "terminal_retained_shell_energy_rel_l1":float(relative[-1])
        })
    return scores


def run(input_path,out):
    with np.load(input_path,allow_pickle=False) as f:
        keys="source target mu rates sigma forcing waves test_times test_paths".split()
        x,y,mu,r,s,force,w,t,truth=[np.asarray(f[k],float) for k in keys]
    assert all(np.isfinite(z).all() for z in (x,y,mu,r,s,force,w,t,truth))
    assert len(x)>=4 and len(y)>=4 and len(truth)>=5 and len(t)>=4 and min(s)>0 and t[0]==0
    assert np.all(np.diff(t)>0) and np.all(r>=0) and (np.square(w).sum(1)>0).all()
    active=np.abs(force)>1e-8*np.max(np.abs(force))
    assert active.any()
    scale=np.maximum(np.std(np.concatenate([x,y]),axis=0),1e-4)
    scale=np.maximum(scale,.05*np.median(scale))
    p,meta=plans(x,y,mu,r,s,t[-1],scale)
    start=perf_counter()
    pred={label:analytic(x,y,mu,r,s,t,pi) for label,pi in p.items()}
    for label in pred:
        np.testing.assert_allclose(pred[label][0],pred["flat"][0],atol=1e-12)
        np.testing.assert_allclose(pred[label][1][[0,-1]],pred["flat"][1][[0,-1]],atol=1e-8,rtol=1e-10)
    scores={label:score_groups(truth,scale,w,active,v) for label,v in pred.items()}
    meta["moment_and_score_seconds"]=perf_counter()-start
    meta["test_groups"],meta["trajectories_per_group"]=truth.shape[:2]
    meta["train_counts"]=[len(x),len(y)]
    meta["score_pairs"]=[(float(t[i]),float(t[j])) for i,j in pred["flat"][3]]
    meta["identity"]="Same two estimated endpoint marginals and reference in every arm. Exact interior means therefore coincide."
    summary=[]
    G=len(truth)
    for label in ("raw_hierarchy","projected_flat_macro_control"):
        for key in scores["flat"][0]:
            a=np.array([v[key] for v in scores[label]])
            b=np.array([v[key] for v in scores["flat"]])
            assert np.isfinite(a).all() and np.isfinite(b).all()
            d=a-b
            half=student_t.ppf(.975,G-1)*d.std(ddof=1)/np.sqrt(G)
            summary.append(dict(arm=label,metric=key,arm_score=a.mean(),flat_score=b.mean(),
                                delta=d.mean(),low=d.mean()-half,high=d.mean()+half))
    out.mkdir(parents=True,exist_ok=True)
    with (out/"summary.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=summary[0])
        writer.writeheader()
        writer.writerows(summary)
    (out/"metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
    lines=["# Fixed-reference APEBench hierarchy diagnostic","",
        "Three analytic conditional-mixture lifts share the same train-mean-centered OU reference and train endpoint marginals. "
        "The primary comparison is the original raw-macro block lift against the flat empirical bridge. "
        "The projected-flat macro arm retains the block allocation of the flat solution but uses the same local constraints as the original hierarchy. "
        "It requires a flat solve and is a mechanism control, not an efficient competitor. No neural network was trained.","",
        f"Train counts: {len(x)} and {len(y)}. Held-out samples: {G} disjoint groups of {truth.shape[1]} trajectories on one fixed train split. "
        f"Primary adjacent interior time pairs: {meta['score_pairs']}. Nonempty source/target cluster counts: {meta['cluster_counts']}.","",
        "The primary metric is standardized cross-covariance RMSE: divide a covariance entry by the product of training-only coordinate scales, "
        "then take the RMSE over all coordinates and adjacent interior time pairs in each group. The forced-band subset and spectrum errors are "
        "companion diagnostics. Spectrum errors are relative L1 errors across retained kinetic-energy shells. Smaller is better. "
        "Paired 95% t intervals use the held-out groups, conditional on the one training split. Terminal moments are matched by construction, "
        "not fitted to the test set. All intervals are unadjusted.","",
        "| Arm | Metric | Arm score | Flat score | Delta (arm minus flat) | 95% lower | 95% upper |",
        "|---|---|---:|---:|---:|---:|---:|"]
    lines += ["| {arm} | {metric} | {arm_score:.6g} | {flat_score:.6g} | {delta:+.6g} | {low:+.6g} | {high:+.6g} |".format(**row)
              for row in summary]
    lines += ["",
        f"Construction time: flat {meta['flat_seconds']:.4g} s, dense hierarchy {meta['hierarchy_seconds']:.4g} s. "
        "These are one-run CPU timings, not a scaling benchmark. "
        f"Flat feasible dual gap: {meta['flat_solver']['duality_gap']:.3g}; "
        f"coarse raw-macro feasible dual gap: {meta['macro_duality_gap']:.3g}; "
        f"largest local feasible dual gap: {meta['worst_local_duality_gap']:.3g}. "
        f"Raw macro vs. projected-flat L1 distance: {meta['macro_l1_distance']:.4g}.","",
        "| Arm | Fine objective minus flat | Endpoint marginal L1 residual | Plan L1 from flat |",
        "|---|---:|---:|---:|"]
    lines += [f"| {a} | {d['objective_minus_flat']:.6g} | {d['feasible_l1']:.3g} | {d['plan_l1_vs_flat']:.6g} |"
              for a,d in meta["plans"].items()]
    lines += ["","An endpoint plan can depart from the reference KL optimum while reducing a finite-sample held-out scientific discrepancy. "
              "That would be metric-specific evidence, not evidence of solving the same SB optimization better.",""]
    (out/"report.md").write_text("\n".join(lines))
    print("Hierarchy diagnostic complete.")
    return meta,summary


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    run(args.input,args.out)
