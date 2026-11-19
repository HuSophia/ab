# Official APEBench Kolmogorov five-split replication

There are five new train draws (64 trajectories each), each paired with a fresh independent test draw
(five disjoint groups of 32 trajectories). Group scores are averaged within each split before
computing any intervals. The same endpoint data, reference, coarse plan where applicable,
and held-out data are used within each split. The original published split is not included.
All predictions are analytic OU conditional mixtures in 10 real Fourier coordinates; no network was trained.
Intervals are descriptive paired 95% Student t intervals on the five split-level effects and are unadjusted.
Numerical enclosures for the recovery fraction are separate from sampling intervals.

| Train seed | Test seed | Proportional minus flat | Balanced minus flat | Objective gap recovered | Energy Δ | Cross-covariance Δ |
|---:|---:|---:|---:|---:|---:|---:|
| 51001 | 91001 | 3.64752 | 0.408026 | 88.81% | -0.013821 | +0.0048077 |
| 51002 | 91002 | 3.17835 | 0.44806 | 85.90% | -0.0036739 | +0.0018967 |
| 51003 | 91003 | 3.33529 | 0.350473 | 89.49% | +0.003893 | +0.0053925 |
| 51004 | 91004 | 3.38349 | 0.0561089 | 98.34% | -0.0072538 | +0.0029445 |
| 51005 | 91005 | 4.46583 | 0.818835 | 81.66% | -0.0026003 | +0.00033254 |

Mean fraction of objective gap recovered: **88.84%** (95% interval 81.22% to 96.47%). This statistic is not a trajectory-error reduction.

| Metric | Comparison | Mean difference | 95% low | 95% high | Improved splits |
|---|---|---:|---:|---:|---:|
| Adjacent interior cross-covariance | Global balancing minus Proportional | +0.00307477 | +0.000492562 | +0.00565698 | 0/5 |
| Interior shell kinetic energy | Global balancing minus Proportional | -0.0046912 | -0.0127624 | +0.00337997 | 4/5 |
| Adjacent interior cross-covariance | Proportional minus Flat | -0.00339154 | -0.00610014 | -0.000682944 | 5/5 |
| Interior shell kinetic energy | Proportional minus Flat | +0.00492307 | -0.00409097 | +0.0139371 | 1/5 |
| Adjacent interior cross-covariance | Global balancing minus Flat | -0.00031677 | -0.00104121 | +0.000407666 | 3/5 |
| Interior shell kinetic energy | Global balancing minus Flat | +0.000231871 | -0.00211023 | +0.00257397 | 2/5 |

Negative error differences favor the first arm. The energy metric is relative L1 over retained shells;
the covariance metric is standardized RMSE over adjacent interior time pairs. No full-field or speed claim is made.
The worst shared coarse-allocation L1 residual was 1.57e-16.
The worst flat and balanced primal–dual gaps were 6.11e-08 and 2.15e-07. See CSV for per-split numerical enclosures.
