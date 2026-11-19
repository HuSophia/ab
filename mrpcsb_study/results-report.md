# APEBench Kolmogorov: five-seed split-conditional study

Generated at 128² on one fixed 128-train / 64-test trajectory split. The learned state has
8 real Fourier coordinates and excludes directly forced mode 4; this is not a full-field model.
Training and evaluation are unpaired snapshot interpolation, not forecasting.

SW² is squared sliced Wasserstein on training-normalized coefficients. Each seed averages three
Monte Carlo draws per time. The resampling floor compares two draws from the same held-out
empirical population; it is not an irreducible error bound. The paired 95% t intervals measure
training/sampling variability on this split, not variation across independently generated splits.

| Reference | Coupling | Interior SW² | Endpoint SW² | Interior floor | Endpoint floor | Runtime (s/arm) |
|---|---|---:|---:|---:|---:|---:|
| brownian | flat | 0.0236838 | 0.0378416 | 0.0154245 | 0.0162989 | 3.31593 |
| brownian | hierarchical | 0.0242009 | 0.0404814 | 0.0154245 | 0.0162989 | 3.50186 |
| physical | flat | 0.0236786 | 0.037789 | 0.0154245 | 0.0162989 | 3.25956 |
| physical | hierarchical | 0.0238217 | 0.0382479 | 0.0154245 | 0.0162989 | 3.49433 |

| Region | Contrast | Paired difference | 95% low | 95% high |
|---|---|---:|---:|---:|
| interior | hierarchy_given_Brownian | 0.000517119 | -0.00124354 | 0.00227778 |
| interior | hierarchy_given_OU | 0.000143091 | -0.00113895 | 0.00142513 |
| interior | OU_given_flat | -5.21896e-06 | -0.000431562 | 0.000421124 |
| interior | OU_given_hierarchy | -0.000379247 | -0.00140855 | 0.000650057 |
| interior | interaction | -0.000374028 | -0.00130986 | 0.000561801 |
| endpoint | hierarchy_given_Brownian | 0.0026398 | -0.00447759 | 0.0097572 |
| endpoint | hierarchy_given_OU | 0.000458954 | -0.00486544 | 0.00578335 |
| endpoint | OU_given_flat | -5.26632e-05 | -0.0011073 | 0.00100197 |
| endpoint | OU_given_hierarchy | -0.00223351 | -0.00607101 | 0.00160399 |
| endpoint | interaction | -0.00218085 | -0.00569601 | 0.00133432 |
