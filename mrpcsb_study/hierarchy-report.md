# Fixed-reference APEBench hierarchy diagnostic

Three analytic conditional-mixture lifts share the same train-mean-centered OU reference and train endpoint marginals. The primary comparison is the original raw-macro block lift against the flat empirical bridge. The projected-flat macro arm retains the block allocation of the flat solution but uses the same local constraints as the original hierarchy. It requires a flat solve and is a mechanism control, not an efficient competitor. No neural network was trained.

Train counts: 64 and 64. Held-out samples: 5 disjoint groups of 32 trajectories on one fixed train split. Primary adjacent interior time pairs: [(0.4, 0.8), (0.8, 1.2000000000000002), (1.2000000000000002, 1.6)]. Nonempty source/target cluster counts: [4, 4].

The primary metric is standardized cross-covariance RMSE: divide a covariance entry by the product of training-only coordinate scales, then take the RMSE over all coordinates and adjacent interior time pairs in each group. The forced-band subset and spectrum errors are companion diagnostics. Spectrum errors are relative L1 errors across retained kinetic-energy shells. Smaller is better. Paired 95% t intervals use the held-out groups, conditional on the one training split. Terminal moments are matched by construction, not fitted to the test set. All intervals are unadjusted.

| Arm | Metric | Arm score | Flat score | Delta (arm minus flat) | 95% lower | 95% upper |
|---|---|---:|---:|---:|---:|---:|
| raw_hierarchy | interior_cross_cov_rmse | 0.218249 | 0.218585 | -0.000335721 | -0.00412348 | +0.00345204 |
| raw_hierarchy | interior_forced_band_cross_cov_rmse | 0.212124 | 0.194921 | +0.0172023 | -0.0206698 | +0.0550744 |
| raw_hierarchy | interior_retained_shell_energy_rel_l1 | 0.0818899 | 0.0635221 | +0.0183678 | +0.0107824 | +0.0259533 |
| raw_hierarchy | terminal_retained_shell_energy_rel_l1 | 0.0737464 | 0.0737464 | +0 | +0 | +0 |
| projected_flat_macro_control | interior_cross_cov_rmse | 0.219029 | 0.218585 | +0.000444373 | -0.00267126 | +0.00356001 |
| projected_flat_macro_control | interior_forced_band_cross_cov_rmse | 0.213871 | 0.194921 | +0.0189493 | -0.00904798 | +0.0469465 |
| projected_flat_macro_control | interior_retained_shell_energy_rel_l1 | 0.0819888 | 0.0635221 | +0.0184667 | +0.0117235 | +0.0252099 |
| projected_flat_macro_control | terminal_retained_shell_energy_rel_l1 | 0.0737464 | 0.0737464 | +0 | +0 | +0 |

Construction time: flat 0.8779 s, dense hierarchy 1.425 s. These are one-run CPU timings, not a scaling benchmark. Flat feasible dual gap: 3.33e-08; coarse raw-macro feasible dual gap: 8.88e-16; largest local feasible dual gap: 2.29e-09. Raw macro vs. projected-flat L1 distance: 0.3338.

| Arm | Fine objective minus flat | Endpoint marginal L1 residual | Plan L1 from flat |
|---|---:|---:|---:|
| flat | 0 | 9.89e-17 | 0 |
| raw_hierarchy | 3.58435 | 1.72e-16 | 1.26167 |
| projected_flat_macro_control | 3.37125 | 1.94e-16 | 1.23889 |

An endpoint plan can depart from the reference KL optimum while reducing a finite-sample held-out scientific discrepancy. That would be metric-specific evidence, not evidence of solving the same SB optimization better.
