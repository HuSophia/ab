# APEBench retained-band replication check

This post-hoc diagnostic reuses paths from the [completed five-seed run](https://github.com/HuSophia/ab/actions/runs/44499102892); no models were retrained or selected using these scores. Earlier in-house experiments motivated the kinetic-energy check. Both references, both regions, and both diagnostics are shown. The relative shell error is the sum of absolute errors in mean shell values divided by the sum of true means, averaged over the specified evaluation times. Lower is better.

These are the eight retained Fourier coordinates (two shells), not a full-field spectrum. Each training seed contributes one saved held-out draw, all on the same train/test split. Intervals are unadjusted paired 95% t intervals across five seeds, conditional on this split. Negative paired changes favor hierarchy.

| Reference | Region | Quantity | Flat | Hierarchical | Paired change | 95% low | 95% high |
|---|---|---|---:|---:|---:|---:|---:|
| brownian | endpoint | enstrophy | 0.0585564 | 0.093385 | 0.0348286 | -0.0234693 | 0.0931266 |
| brownian | endpoint | kinetic_energy | 0.0724504 | 0.0847264 | 0.012276 | -0.0369665 | 0.0615185 |
| brownian | interior | enstrophy | 0.0649253 | 0.061558 | -0.00336732 | -0.0362077 | 0.029473 |
| brownian | interior | kinetic_energy | 0.0690567 | 0.0625403 | -0.00651639 | -0.0350468 | 0.022014 |
| physical | endpoint | enstrophy | 0.0612792 | 0.0788898 | 0.0176106 | -0.0297526 | 0.0649737 |
| physical | endpoint | kinetic_energy | 0.075469 | 0.0744415 | -0.00102748 | -0.0398615 | 0.0378065 |
| physical | interior | enstrophy | 0.0684093 | 0.0644588 | -0.00395053 | -0.0326436 | 0.0247425 |
| physical | interior | kinetic_energy | 0.0725548 | 0.0669516 | -0.00560325 | -0.0328642 | 0.0216577 |
