"""Render the complete retained-band aggregate table; no seed or path values published."""
from pathlib import Path
import sys
import pandas as pd

root = Path(sys.argv[1])
scores = pd.read_csv(root / "retained_band_seed_scores.csv")
paired = pd.read_csv(root / "retained_band_paired_contrasts.csv")
groups = ["reference", "region", "quantity"]
means = scores.groupby(groups)[["flat", "hierarchical"]].mean().reset_index()
view = means.merge(paired, on=groups, validate="one_to_one")
assert len(view) == 8 and (view["count"] == 5).all()
intro = """# APEBench retained-band replication check

This post-hoc diagnostic reuses paths from the [completed five-seed run](https://github.com/HuSophia/ab/actions/runs/44499102892); no models were retrained or selected using these scores. Earlier in-house experiments motivated the kinetic-energy check. Both references, both regions, and both diagnostics are shown. The relative shell error is the sum of absolute errors in mean shell values divided by the sum of true means, averaged over the specified evaluation times. Lower is better.

These are the eight retained Fourier coordinates (two shells), not a full-field spectrum. Each training seed contributes one saved held-out draw, all on the same train/test split. Intervals are unadjusted paired 95% t intervals across five seeds, conditional on this split. Negative paired changes favor hierarchy.

| Reference | Region | Quantity | Flat | Hierarchical | Paired change | 95% low | 95% high |
|---|---|---|---:|---:|---:|---:|---:|
"""
lines = [
    f"| {z.reference} | {z.region} | {z.quantity} | {z.flat:.6g} | {z.hierarchical:.6g} | {z.mean:.6g} | {z.ci_low:.6g} | {z.ci_high:.6g} |"
    for z in view.itertuples(index=False)
]
(root / "retained-band-report.md").write_text(intro + "\n".join(lines) + "\n")
