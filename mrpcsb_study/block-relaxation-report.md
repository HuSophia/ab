# APEBench fixed-coarse global-balancing diagnostic

This mechanism check reuses the hierarchy run's split, holdout groups and train-mean-centered OU reference. The original proportional lift and globally balanced variant share the same raw coarse block allocation and fine endpoint marginals. Only the proportional per-block fine-marginal restrictions were removed. All scores use analytic mixtures, not neural rollouts.

Endpoint populations: 64 and 64. Held-out groups: 5, 32 trajectories each. Adjacent interior joint cross-covariance errors use training-only scale normalization. Paired 95% t intervals vary held-out groups while holding the train split fixed. Lower is better; zero terminal contrasts are an endpoint-matching check.

| Arm | Comparator | Metric | Arm | Comparator | Paired difference | 95% lower | 95% upper |
|---|---|---|---:|---:|---:|---:|---:|
| global_relaxed | old_proportional | interior_cross_cov_rmse | 0.218161 | 0.218249 | -8.82128e-05 | -0.00299241 | +0.00281599 |
| global_relaxed | old_proportional | interior_forced_band_cross_cov_rmse | 0.197708 | 0.212124 | -0.014416 | -0.0420492 | +0.0132172 |
| global_relaxed | old_proportional | interior_retained_shell_energy_rel_l1 | 0.064254 | 0.0818899 | -0.0176359 | -0.0238893 | -0.0113825 |
| global_relaxed | old_proportional | terminal_retained_shell_energy_rel_l1 | 0.0737464 | 0.0737464 | +0 | +0 | +0 |
| old_proportional | flat | interior_cross_cov_rmse | 0.218249 | 0.218585 | -0.000335721 | -0.00412348 | +0.00345204 |
| old_proportional | flat | interior_forced_band_cross_cov_rmse | 0.212124 | 0.194921 | +0.0172023 | -0.0206698 | +0.0550744 |
| old_proportional | flat | interior_retained_shell_energy_rel_l1 | 0.0818899 | 0.0635221 | +0.0183678 | +0.0107824 | +0.0259533 |
| old_proportional | flat | terminal_retained_shell_energy_rel_l1 | 0.0737464 | 0.0737464 | +0 | +0 | +0 |
| global_relaxed | flat | interior_cross_cov_rmse | 0.218161 | 0.218585 | -0.000423934 | -0.00182676 | +0.000978892 |
| global_relaxed | flat | interior_forced_band_cross_cov_rmse | 0.197708 | 0.194921 | +0.00278631 | -0.0144807 | +0.0200533 |
| global_relaxed | flat | interior_retained_shell_energy_rel_l1 | 0.064254 | 0.0635221 | +0.000731914 | -0.00142901 | +0.00289284 |
| global_relaxed | flat | terminal_retained_shell_energy_rel_l1 | 0.0737464 | 0.0737464 | +0 | +0 | +0 |

Fine objective minus the flat solution: proportional 3.58435, global balancing 0.436965. Global balancing minus proportional: -3.14739. Raw-projection coarse KL estimate: 0.1304455173881121.

Fixed-coarse primal/dual gap: 3.41e-08; maximum constraint-family L1 residual: 2.17e-16. Original and globally-balanced raw block allocations agree to 9.5e-17 in L1. The relaxed feasible objective is guaranteed no larger than the original proportional lift; equivalence holds only if its additional constraints are already met by the optimum.

Projected-flat allocation oracle: dual gap 3.33e-08, rounded candidate L1 distance to the flat plan 2.3e-07. Oracle used incumbent flat plan: True. This is a correctness check, not a deployable baseline.

One-run CPU timings: flat construction 0.8604 s; old dense hierarchy construction 1.374 s; additional fixed-block global solve 0.8799 s. No speedup claim is made.
