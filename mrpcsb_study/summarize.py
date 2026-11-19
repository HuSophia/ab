"""Validate and summarize the APEBench four-arm study without pooling folds or draws as seeds."""
import argparse
import csv
import json
import math
import statistics as stats
from collections import Counter
from pathlib import Path

from scipy.stats import t

REFERENCES = ('brownian', 'physical')
COUPLINGS = ('flat', 'hierarchical')
REGIONS = ('interior', 'endpoint')
TIMES = (0.2, 0.4, 0.6, 0.8, 1.0)


def parameters(rows, field):
    vec = [row[field] for row in rows]
    assert all(math.isfinite(x) and x >= 0 for x in vec), field
    return stats.mean(vec)


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        output = csv.DictWriter(f, fieldnames=rows[0].keys())
        output.writeheader()
        output.writerows(rows)


def interval(xs):
    m = stats.mean(xs)
    half = t.ppf(.975, len(xs)-1)*stats.stdev(xs)/math.sqrt(len(xs))
    return {'mean': m, 'ci_low': m-half, 'ci_high': m+half}


def study(path, seeds):
    train = json.loads((path/'results.json').read_text())
    rows = json.loads((path/'heldout_metrics.json').read_text())
    arms = [(s, r, c) for s in range(seeds) for r in REFERENCES for c in COUPLINGS]
    key = lambda z: (z['seed'], z['reference'], z['coupling'])
    assert Counter(map(key, train)) == Counter(arms), 'Missing or repeated fit'
    expected = Counter((a,d,v) for a in arms for d in range(3) for v in TIMES)
    assert Counter((key(z),z['draw'],round(z['t'],4)) for z in rows) == expected, 'Incomplete held-out evaluation'
    assert all(z['region']==('endpoint' if round(z['t'],4)==1 else 'interior') for z in rows)
    assert max(z['max_marginal_residual'] for z in train) < 1e-8, 'Endpoint coupling not feasible'
    times = ['coupling_seconds','training_seconds','sampling_seconds','heldout_eval_seconds','total_seconds']
    fit = {key(z):z for z in train}
    tab, effects, seed_deltas = [], [], []
    for seed,ref,coupling in arms:
        arm = seed,ref,coupling
        for region in REGIONS:
            z = [v for v in rows if key(v)==arm and v['region']==region]
            tab.append(dict(seed=seed,reference=ref,coupling=coupling,region=region,
                            sw2_squared=parameters(z,'SW2_squared'),
                            resampling_floor=parameters(z,'empirical_resampling_variability'),
                            **{field:parameters([fit[arm]],field) for field in times}))
    by = {(x['seed'],x['reference'],x['coupling'],x['region']):x for x in tab}
    for region in REGIONS:
        definitions = {
            'hierarchy_given_Brownian': {('brownian','hierarchical'):1,('brownian','flat'):-1},
            'hierarchy_given_OU': {('physical','hierarchical'):1,('physical','flat'):-1},
            'OU_given_flat': {('physical','flat'):1,('brownian','flat'):-1},
            'OU_given_hierarchy': {('physical','hierarchical'):1,('brownian','hierarchical'):-1},
            'interaction': {('physical','hierarchical'):1,('physical','flat'):-1,
                            ('brownian','hierarchical'):-1,('brownian','flat'):1},
        }
        for label,coeff in definitions.items():
            vals = [sum(w*by[(s,*arm,region)]['sw2_squared'] for arm,w in coeff.items()) for s in range(seeds)]
            effects.append(dict(region=region,contrast=label,**interval(vals)))
            seed_deltas.extend(dict(region=region, contrast=label, seed=s, delta_sw2_squared=v) for s,v in enumerate(vals))
    write_csv(path/'seed_scores.csv', tab)
    write_csv(path/'paired_contrasts.csv', effects)
    write_csv(path/'paired_seed_differences.csv',seed_deltas)
    desc = [
        '# APEBench Kolmogorov: five-seed split-conditional study',
        '',
        'Generated at 128² on one fixed 128-train / 64-test trajectory split. The learned state has',
        '8 real Fourier coordinates and excludes directly forced mode 4; this is not a full-field model.',
        'Training and evaluation are unpaired snapshot interpolation, not forecasting.',
        '',
        'SW² is squared sliced Wasserstein on training-normalized coefficients. Each seed averages three',
        'Monte Carlo draws per time. The resampling floor compares two draws from the same held-out',
        'empirical population; it is not an irreducible error bound. The paired 95% t intervals measure',
        'training/sampling variability on this split, not variation across independently generated splits.',
        '',
        '| Reference | Coupling | Interior SW² | Endpoint SW² | Interior floor | Endpoint floor | Runtime (s/arm) |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    for r in REFERENCES:
        for c in COUPLINGS:
            iv,ev=([by[(s,r,c,v)] for s in range(seeds)] for v in REGIONS)
            nums=[parameters(iv,'sw2_squared'),parameters(ev,'sw2_squared'),
                  parameters(iv,'resampling_floor'),parameters(ev,'resampling_floor'),parameters(iv,'total_seconds')]
            desc.append(f'| {r} | {c} | '+' | '.join(f'{v:.6g}' for v in nums)+' |')
    desc += ['', '| Region | Contrast | Paired difference | 95% low | 95% high |',
             '|---|---|---:|---:|---:|']
    for x in effects:
        desc.append('| {region} | {contrast} | {mean:.6g} | {ci_low:.6g} | {ci_high:.6g} |'.format(**x))
    (path/'report.md').write_text('\n'.join(desc)+'\n')


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('path',type=Path)
    a.add_argument('--seeds',type=int,default=5)
    arg=a.parse_args()
    study(arg.path,arg.seeds)
