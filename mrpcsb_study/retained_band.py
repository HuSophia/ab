"""Recompute low-band diagnostics from a completed run without any retraining."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student


def variance(t, r, s):
    if np.all(r == 0):
        return s**2*t
    out = s**2*t*np.ones_like(r)
    mask = r > 0
    out[mask] = s[mask]**2*(-np.expm1(-2*r[mask]*t))/(2*r[mask])
    return out


def evaluate(bundle, model_dir, dest):
    keys = 'source target mu rates sigma T test_paths test_times waves test_ids train_ids num_points'.split()
    with np.load(bundle, allow_pickle=False) as f:
        dat = {k:f[k] for k in keys}
    x,y,mu,r,s,T,truth,ts,w,test_ids,train_ids,n = (dat[k] for k in keys)
    T,n = float(T),int(n)
    assert truth.shape==(len(test_ids),len(ts),len(mu))
    assert not np.intersect1d(test_ids, train_ids).size
    assert w.shape==(len(mu),2) and np.all(w[1::2]==w[::2])
    k2=np.sum(w*w,axis=1)
    assert np.all(k2>0)
    shells=np.unique(k2)
    masks=(k2[None,:]==shells[:,None])
    weights=dict(kinetic_energy=1/(2*n*n*k2), enstrophy=np.full(len(k2),1/(2*n*n)))
    ss={}
    def sample_stats(paths):
        assert paths.ndim==3 and paths.shape[0]==len(ts) and np.all(np.isfinite(paths))
        return np.mean((paths+mu)**2,axis=1)
    ref_truth=sample_stats(truth.transpose(1,0,2))
    rows=[]
    for ref in ('brownian','physical'):
        r0 = r if ref=='physical' else np.zeros_like(r)
        vt = variance(T,r0,s)
        a,b,q=[],[],[]
        for ti in ts:
            v = variance(ti,r0,s)
            bi=v*np.exp(-r0*(T-ti))/vt
            a.append(np.exp(-r0*ti)-bi*np.exp(-r0*T))
            b.append(bi)
            q.append(v*variance(T-ti,r0,s)/vt)
        a,b,q = (np.asarray(z) for z in (a,b,q))
        for plan in ('flat','hierarchical'):
            prefix=f'{ref}_{plan}'
            with np.load(model_dir/f'coupling_{prefix}.npz',allow_pickle=False) as f:
                pi=f['coupling']
            assert pi.shape==(len(x),len(y)) and (pi>=0).all()
            np.testing.assert_allclose(pi.sum(1),1/len(x),atol=1e-8)
            np.testing.assert_allclose(pi.sum(0),1/len(y),atol=1e-8)
            ss[ref,plan]=np.einsum('ij,id,jd->d',pi,x,y)
            # The original reference mixture uses empirical training starts, not test starts.
            ex=[] 
            for ai,bi,qi in zip(a,b,q):
                exact_mean=ai*x[:,None,:]+bi*y[None,:,:]+mu
                ex.append(np.einsum('ij,ijd->d',pi,exact_mean**2)+qi)
            ex=np.asarray(ex)
            np.testing.assert_allclose(ex[0],np.mean((x+mu)**2,axis=0),atol=1e-8)
            np.testing.assert_allclose(ex[-1],np.mean((y+mu)**2,axis=0),atol=1e-8)
            if plan=='flat':
                flat_exact=ex
            else:
                rhs=2*a*b*(ss[ref,plan]-ss[ref,'flat'])
                np.testing.assert_allclose(ex-flat_exact,rhs,rtol=1e-7,atol=1e-7)
            for seed in range(5):
                with np.load(model_dir/f'{prefix}_seed{seed}_heldout_paths.npz',allow_pickle=False) as f:
                    np.testing.assert_array_equal(f['test_ids'],test_ids)
                    np.testing.assert_allclose(f['times'],ts)
                    val=sample_stats(f['marginals'])
                with np.load(model_dir/f'{prefix}_seed{seed}.npz',allow_pickle=False) as f:
                    ii=[np.argmin(abs(t-f['times'])) for t in ts]
                    np.testing.assert_allclose(f['times'][ii],ts)
                    trained=sample_stats(f['paths'][ii])
                for quantity,wt in weights.items():
                    arrays = [(field*wt)@masks.T for field in (ref_truth,val,trained,ex)]
                    for i,t in enumerate(ts):
                        for j,k in enumerate(shells):
                            rows.append(dict(reference=ref,coupling=plan,seed=seed,t=t,
                                  region=('start' if i==0 else 'endpoint' if i==len(ts)-1 else 'interior'),
                                  quantity=quantity,shell_k2=k,
                                  heldout_truth=arrays[0][i,j],heldout_rollout=arrays[1][i,j],
                                  training_start_rollout=arrays[2][i,j],training_start_mixture=arrays[3][i,j]))
    dest.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(rows)
    df.to_csv(dest/'retained_band_shells.csv',index=False)
    sub=df[df.region!='start'].copy()
    sub['abs_error']=abs(sub.heldout_rollout-sub.heldout_truth)
    group=['reference','coupling','seed','region','quantity']
    by_t=sub.groupby(group+['t'])[['abs_error','heldout_truth']].sum()
    by_t['relative_shell_l1']=by_t.abs_error/by_t.heldout_truth
    out=by_t.groupby(group).relative_shell_l1.mean().unstack('coupling')
    out['hierarchy_minus_flat']=out.hierarchical-out.flat
    out.to_csv(dest/'retained_band_seed_scores.csv')
    keys=['reference','region','quantity']
    paired=out.groupby(keys).hierarchy_minus_flat.agg(['mean','std','count'])
    half=student.ppf(.975,paired['count']-1)*paired['std']/np.sqrt(paired['count'])
    paired['ci_low']=paired['mean']-half
    paired['ci_high']=paired['mean']+half
    paired.to_csv(dest/'retained_band_paired_contrasts.csv')
    print('One saved held-out draw per seed, fixed population, retained modes only.')
    print(paired[['mean','ci_low','ci_high']].to_string())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--models',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    evaluate(a.input,a.models,a.out)
