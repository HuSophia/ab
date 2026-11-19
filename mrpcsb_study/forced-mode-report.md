# Fixed-coupling APEBench forced-mode diagnostic

This is a control experiment on exact affine-OU bridge mixtures, not neural training or a hierarchy comparison. Endpoints, transport plan, viscosity/drag damping, and diffusion were matched across all three arms. Only the additive reference drift changed: zero in raw coefficients, explicit simulator forcing, or relaxation toward the initial training mean. The last comparator tests whether explicit forcing adds value beyond simple empirical centering.

Used 64 training starts and 64 training endpoints; 5 independent held-out groups of 32 trajectories each share one training split. Retained 10 real Fourier coordinates. Directly forced wave(s): [[0.0, 4.0]].

Final marginal feasibility correction was preceded by an L1 residual of 1.11e-08; the feasible primal/dual gap was 4.19e-08. The plan is numerical; conditional moments are evaluated analytically. Changing constant forcing shifts the interior mean but leaves the centered covariance and both endpoint moments unchanged. Maximum deterministic forced-coordinate mean shift vs zero: 21.9769 (raw coefficient units).

Lower is better. The forced-coordinate mean RMSE (the main diagnostic) divides the mean-vector squared error by the training endpoints' raw mean squared norm, then takes its square root after averaging over evaluation times within each test group. The second-moment RMSE instead divides the squared-norm expectation error by that same training norm before taking a time-RMSE. Retained-energy relative RMSE uses the per-time true retained kinetic energy to normalize. Each reported score averages group-level errors. Contrasts and unadjusted paired 95% t intervals are over the held-out groups, conditional on this training split; negative favors the named arm. Companion metrics are descriptive, not additional confirmatory tests.

| Region | Band | Metric | Arm | Comparator | Arm score | Comparator score | Paired difference | 95% low | 95% high |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| endpoint | all_retained | retained_energy_relative_rmse | forced_ou | unforced_ou | 0.0544697 | 0.0544697 | 0 | 0 | 0 |
| endpoint | all_retained | retained_energy_relative_rmse | forced_ou | train_mean_centered_ou | 0.0544697 | 0.0544697 | 0 | 0 | 0 |
| endpoint | all_retained | retained_energy_relative_rmse | train_mean_centered_ou | unforced_ou | 0.0544697 | 0.0544697 | 0 | 0 | 0 |
| endpoint | forced_coordinate | mean_rmse | forced_ou | unforced_ou | 0.0283596 | 0.0283596 | 0 | 0 | 0 |
| endpoint | forced_coordinate | mean_rmse | forced_ou | train_mean_centered_ou | 0.0283596 | 0.0283596 | 0 | 0 | 0 |
| endpoint | forced_coordinate | mean_rmse | train_mean_centered_ou | unforced_ou | 0.0283596 | 0.0283596 | 0 | 0 | 0 |
| endpoint | forced_coordinate | second_moment_rmse | forced_ou | unforced_ou | 0.060557 | 0.060557 | 0 | 0 | 0 |
| endpoint | forced_coordinate | second_moment_rmse | forced_ou | train_mean_centered_ou | 0.060557 | 0.060557 | 0 | 0 | 0 |
| endpoint | forced_coordinate | second_moment_rmse | train_mean_centered_ou | unforced_ou | 0.060557 | 0.060557 | 0 | 0 | 0 |
| endpoint | forced_mode | mean_rmse | forced_ou | unforced_ou | 0.0337118 | 0.0337118 | 0 | 0 | 0 |
| endpoint | forced_mode | mean_rmse | forced_ou | train_mean_centered_ou | 0.0337118 | 0.0337118 | 0 | 0 | 0 |
| endpoint | forced_mode | mean_rmse | train_mean_centered_ou | unforced_ou | 0.0337118 | 0.0337118 | 0 | 0 | 0 |
| endpoint | forced_mode | second_moment_rmse | forced_ou | unforced_ou | 0.0584578 | 0.0584578 | 0 | 0 | 0 |
| endpoint | forced_mode | second_moment_rmse | forced_ou | train_mean_centered_ou | 0.0584578 | 0.0584578 | 0 | 0 | 0 |
| endpoint | forced_mode | second_moment_rmse | train_mean_centered_ou | unforced_ou | 0.0584578 | 0.0584578 | 0 | 0 | 0 |
| interior | all_retained | retained_energy_relative_rmse | forced_ou | unforced_ou | 0.181358 | 0.0725467 | 0.108811 | 0.0336199 | 0.184002 |
| interior | all_retained | retained_energy_relative_rmse | forced_ou | train_mean_centered_ou | 0.181358 | 0.0484182 | 0.13294 | 0.0770153 | 0.188864 |
| interior | all_retained | retained_energy_relative_rmse | train_mean_centered_ou | unforced_ou | 0.0484182 | 0.0725467 | -0.0241285 | -0.046646 | -0.00161102 |
| interior | forced_coordinate | mean_rmse | forced_ou | unforced_ou | 0.13997 | 0.0303039 | 0.109666 | 0.066593 | 0.15274 |
| interior | forced_coordinate | mean_rmse | forced_ou | train_mean_centered_ou | 0.13997 | 0.0219588 | 0.118012 | 0.0890029 | 0.14702 |
| interior | forced_coordinate | mean_rmse | train_mean_centered_ou | unforced_ou | 0.0219588 | 0.0303039 | -0.00834511 | -0.0277226 | 0.0110323 |
| interior | forced_coordinate | second_moment_rmse | forced_ou | unforced_ou | 0.284042 | 0.0683674 | 0.215675 | 0.129595 | 0.301755 |
| interior | forced_coordinate | second_moment_rmse | forced_ou | train_mean_centered_ou | 0.284042 | 0.044689 | 0.239353 | 0.173861 | 0.304846 |
| interior | forced_coordinate | second_moment_rmse | train_mean_centered_ou | unforced_ou | 0.044689 | 0.0683674 | -0.0236783 | -0.0554754 | 0.00811876 |
| interior | forced_mode | mean_rmse | forced_ou | unforced_ou | 0.140528 | 0.0338677 | 0.10666 | 0.064657 | 0.148663 |
| interior | forced_mode | mean_rmse | forced_ou | train_mean_centered_ou | 0.140528 | 0.0264533 | 0.114074 | 0.0833673 | 0.144781 |
| interior | forced_mode | mean_rmse | train_mean_centered_ou | unforced_ou | 0.0264533 | 0.0338677 | -0.00741437 | -0.0235839 | 0.00875517 |
| interior | forced_mode | second_moment_rmse | forced_ou | unforced_ou | 0.277501 | 0.0696122 | 0.207889 | 0.120484 | 0.295293 |
| interior | forced_mode | second_moment_rmse | forced_ou | train_mean_centered_ou | 0.277501 | 0.043557 | 0.233944 | 0.165394 | 0.302495 |
| interior | forced_mode | second_moment_rmse | train_mean_centered_ou | unforced_ou | 0.043557 | 0.0696122 | -0.0260552 | -0.0556007 | 0.00349028 |
