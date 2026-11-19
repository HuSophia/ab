"""Frozen three-coupling replication on five fresh official APEBench data draws."""
from __future__ import annotations

import argparse
import csv
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

import numpy as np
from scipy.stats import t as student_t

from eval_block_relaxation import run as evaluate
from generate_hierarchy import wave_catalog

HERE = Path(__file__).resolve().parent
# Declared before any official rerun. The original (49123, 89391) is excluded.
SEEDS = (
    (51001, 91001), (51002, 91002), (51003, 91003),
    (51004, 91004), (51005, 91005),
)
ENDPOINT_SHUFFLE_SEED = 9821
ENERGY = "interior_retained_shell_energy_rel_l1"
COV = "interior_cross_cov_rmse"
ARMS = ("flat", "old_proportional", "global_relaxed")
PAIRS = (("global_relaxed", "old_proportional"),
         ("old_proportional", "flat"), ("global_relaxed", "flat"))
LABELS = {
    "flat": "Flat",
    "old_proportional": "Proportional",
    "global_relaxed": "Global balancing",
}
ENV = ("JAX_PLATFORMS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "XLA_FLAGS")
FILES = ("mrpcsb_replication.py", "generate_hierarchy.py", "evaluate.py",
         "eval_hierarchy.py", "eval_block_relaxation.py", "requirements-apebench.txt")


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def sha256(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def packages():
    out = {}
    for name in ("apebench", "exponax", "equinox", "numpy", "scipy", "jax", "jaxlib", "optax", "chex"):
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def generate_fixture(data, train_seed, test_seed):
    """A cheap OU-coordinate integration fixture; never a PDE benchmark."""
    data.mkdir()
    waves = wave_catalog(64)
    keep = (abs(waves).max(1)<2) | (waves==[0,4]).all(1)
    w = np.repeat(waves[keep],2,axis=0)
    r = .1 + .01 * np.square(w).sum(1)
    force = np.zeros(len(w))
    force[np.flatnonzero((w==[0,4]).all(1))[0]] = -4
    scale = np.geomspace(4, 1, len(w))
    mean = np.zeros_like(scale)
    mean[(w==[0,4]).all(1)] = [-6, 0]
    times = np.arange(6) * .4

    def paths(seed, n):
        rng = np.random.default_rng(seed)
        z = [rng.normal(size=(n,len(w))) * scale + mean]
        g = np.exp(-r * .4)
        sd = scale * np.sqrt(1-g*g)
        for _ in range(5):
            z.append(mean + g*(z[-1]-mean) + rng.normal(size=z[0].shape)*sd)
        return np.stack(z,axis=1)

    tr = paths(train_seed,64)
    te = paths(test_seed,160).reshape(5,32,6,len(w))
    mu = tr[:,0].mean(0)
    var = tr[:,0].var(0)
    var = np.maximum(var,max(np.median(var)*1e-4,1e-12))
    rng = np.random.default_rng(ENDPOINT_SHUFFLE_SEED)
    np.savez_compressed(data/"input.npz",
                        source=tr[rng.permutation(64),0],
                        target=tr[rng.permutation(64),-1],
                        test_paths=te, test_times=times, rates=r, sigma=np.sqrt(2*r*var),
                        forcing=force, waves=w, mu=mu, num_points=64)
    write_json(data/"metadata.json", {"profile":"synthetic-integration-fixture",
               "parameters":{"train_seed":train_seed,"test_seed":test_seed}})


def generate(data, pair, profile):
    if profile == "fixture":
        generate_fixture(data, *pair)
        return
    subprocess.run([sys.executable, str(HERE / "generate_hierarchy.py"),
                    "--profile", "study", "--out", str(data),
                    "--train-seed", str(pair[0]), "--test-seed", str(pair[1]),
                    "--endpoint-shuffle-seed", str(ENDPOINT_SHUFFLE_SEED)], check=True)


def validate(data, pair, profile, meta):
    """Check frozen benchmark dimensions and reject mislabeled inputs."""
    assert (meta["parameters"]["train_seed"], meta["parameters"]["test_seed"]) == pair
    if profile == "study":
        assert meta["profile"] == "study" and meta["parameters"]["num_points"] == 64
        assert meta["parameters"]["num_warmup_steps"] == 500
        assert meta["parameters"]["num_substeps"] == 20
        assert meta["endpoint_shuffle_seed"] == ENDPOINT_SHUFFLE_SEED
        assert meta["forced_wave"] == [0, 4]
    else:
        assert meta["profile"] == "synthetic-integration-fixture"
    with np.load(data,allow_pickle=False) as f:
        assert f["source"].shape == (64,10) and f["target"].shape == (64,10)
        assert f["test_paths"].shape == (5,32,6,10)
        assert np.array_equal(np.unique(np.square(f["waves"]).sum(1)),[1,2,16])
        np.testing.assert_allclose(f["test_times"], np.linspace(0,2,6))
        assert np.isclose(f["num_points"], 64)
        # Verify the reference calibration from training endpoints only.
        standard_deviation = np.std(f["source"], axis=0)
        var = standard_deviation**2
        var = np.maximum(var,max(np.median(var)*1e-4,1e-12))
        np.testing.assert_allclose(f["mu"], f["source"].mean(0), atol=1e-8)
        np.testing.assert_allclose(f["sigma"], np.sqrt(2*f["rates"]*var), atol=1e-8)
        np.testing.assert_allclose(f["rates"], .1+.01*np.square(f["waves"]).sum(1), atol=1e-12)


def collect(index, pair, report, summary):
    """Read identical metrics from the original estimator and audit its identity checks."""
    provenance = {"split": index, "train_seed": pair[0], "test_seed": pair[1]}
    metrics = {}
    for row in summary:
        for name, key in (("arm","arm_score"),("comparator","comparator_score")):
            slot = metrics.setdefault(row["metric"], {})
            if row[name] in slot:
                assert np.isclose(slot[row[name]], row[key], atol=1e-14, rtol=0)
            slot[row[name]] = float(row[key])
    flat_gap = float(report["original_info"]["flat_solver"]["duality_gap"])
    relaxed_gap = float(report["gap"])
    old_minus_f = float(report["raw_hierarchy_objective_minus_flat"])
    balanced_minus_f = float(report["relaxed_objective_minus_flat"])
    improvement = -float(report["relaxed_objective_minus_old"])
    np.testing.assert_allclose(old_minus_f-balanced_minus_f, improvement, atol=1e-11)
    assert old_minus_f >= -flat_gap-1e-9 and balanced_minus_f >= -flat_gap-1e-9
    assert improvement >= -1e-10
    assert report["original_block_mass_l1_residual"] < 5e-11
    assert report["feasible_l1"] < 5e-11
    assert max(v["feasible_l1"] for v in report["original_info"]["plans"].values()) < 1e-10
    if report["original_info"]["cluster_counts"] != [4,4]:
        raise RuntimeError("Frozen four-cluster fit produced an empty cluster; report this split rather than replacing it.")
    term = metrics["terminal_retained_shell_energy_rel_l1"]
    np.testing.assert_allclose(list(term.values()), term["flat"], atol=1e-10, rtol=1e-10)
    # For a feasible rounded value F, true optimum is in [F-g_F,F].
    # N=J_proportional-J_balanced^* and D=J_proportional-J_flat^*
    # have independent numerical enclosures, so bracket N/D conservatively.
    # These enclosures address solver error, not sampling uncertainty.
    recovered = improvement/old_minus_f if old_minus_f > 1e-10 else None
    lower = max(0., improvement)/(old_minus_f+flat_gap) if recovered is not None else None
    upper = min(1., (improvement+relaxed_gap)/old_minus_f) if recovered is not None else None
    rows = [provenance | {"arm":arm,"metric":key,"score":scores[arm]}
            for key,scores in metrics.items() for arm in ARMS]
    first = provenance | {
        "old_minus_flat": old_minus_f, "balanced_minus_flat": balanced_minus_f,
        "old_minus_balanced": improvement, "fraction_recovered": recovered,
        "numerical_lower": lower, "numerical_upper": upper,
        "flat_dual_gap":flat_gap, "balanced_dual_gap":relaxed_gap,
        "balanced_constraint_l1":float(report["feasible_l1"]),
        "shared_coarse_l1":float(report["original_block_mass_l1_residual"]),
        "flat_seconds":float(report["original_info"]["flat_seconds"]),
        "proportional_seconds":float(report["original_info"]["hierarchy_seconds"]),
        "additional_balancing_seconds":float(report["seconds"]),
    }
    return first,rows,metrics


def ci(values):
    x = np.asarray(values,float)
    assert len(x) == len(SEEDS) and np.isfinite(x).all()
    center = float(x.mean())
    hw = float(student_t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x)))
    return {"mean":center,"low":center-hw,"high":center+hw,"n":len(x)}


def summarize(out, profile, objective, all_rows, split_scores):
    write_csv(out/"objectives_by_split.csv", objective)
    write_csv(out/"scores_by_split.csv", all_rows)
    contrast_rows = []
    per_split_contrasts = []
    for a,b in PAIRS:
        for metric in split_scores[0]:
            paired = [scores[metric][a]-scores[metric][b] for scores in split_scores]
            contrast_rows.append({"metric":metric,"arm":a,"comparator":b} | ci(paired)
                                 | {"improved":sum(d<0 for d in paired),
                                    "worsened":sum(d>0 for d in paired)})
            for (tr,te), d in zip(SEEDS, paired):
                per_split_contrasts.append({"train_seed":tr,"test_seed":te,"metric":metric,
                                            "arm":a,"comparator":b,"difference":d})
    write_csv(out/"paired_summary.csv", contrast_rows)
    write_csv(out/"paired_differences_by_split.csv", per_split_contrasts)
    raw_rec = [d["fraction_recovered"] for d in objective]
    recovery = ci(raw_rec) if all(a is not None for a in raw_rec) else None
    write_json(out/"objective_summary.json", {
        "fraction_recovered":recovery,
        "absolute_objective_gap_reduction":ci([d["old_minus_balanced"] for d in objective]),
        "original_objective_gap":ci([d["old_minus_flat"] for d in objective]),
        "balanced_objective_gap":ci([d["balanced_minus_flat"] for d in objective]),
        "max_flat_dual_gap":max(d["flat_dual_gap"] for d in objective),
        "max_balanced_dual_gap":max(d["balanced_dual_gap"] for d in objective),
        "max_constraint_l1":max(d["balanced_constraint_l1"] for d in objective),
        "max_shared_coarse_l1":max(d["shared_coarse_l1"] for d in objective),
    })
    title = "Official APEBench Kolmogorov five-split replication" if profile == "study" else "Synthetic integration fixture only — NOT an APEBench result"
    lines = [f"# {title}", "",
        "There are five new train draws (64 trajectories each), each paired with a fresh independent test draw",
        "(five disjoint groups of 32 trajectories). Group scores are averaged within each split before",
        "computing any intervals. The same endpoint data, reference, coarse plan where applicable,",
        "and held-out data are used within each split. The original published split is not included.",
        "All predictions are analytic OU conditional mixtures in 10 real Fourier coordinates; no network was trained.",
        "Intervals are descriptive paired 95% Student t intervals on the five split-level effects and are unadjusted.",
        "Numerical enclosures for the recovery fraction are separate from sampling intervals.", "",
        "| Train seed | Test seed | Proportional minus flat | Balanced minus flat | Objective gap recovered | Energy Δ | Cross-covariance Δ |",
        "|---:|---:|---:|---:|---:|---:|---:|"]
    for d,s in zip(objective,split_scores):
        p = d["fraction_recovered"]
        p = f"{p:.2%}" if p is not None else "undefined"
        e = s[ENERGY]["global_relaxed"] - s[ENERGY]["old_proportional"]
        c = s[COV]["global_relaxed"] - s[COV]["old_proportional"]
        lines.append(f"| {d['train_seed']} | {d['test_seed']} | {d['old_minus_flat']:.6g} | "
                     f"{d['balanced_minus_flat']:.6g} | {p} | {e:+.5g} | {c:+.5g} |")
    if recovery:
        lines += ["",f"Mean fraction of objective gap recovered: **{recovery['mean']:.2%}** "
                  f"(95% interval {recovery['low']:.2%} to {recovery['high']:.2%}). "
                  "This statistic is not a trajectory-error reduction."]
    else:
        lines += ["","No ratio interval: at least one original objective gap was too close to zero to report a stable recovered fraction."]
    lines += ["","| Metric | Comparison | Mean difference | 95% low | 95% high | Improved splits |",
              "|---|---|---:|---:|---:|---:|"]
    for r in contrast_rows:
        if r["metric"] in (ENERGY,COV):
            m = "Interior shell kinetic energy" if r["metric"] == ENERGY else "Adjacent interior cross-covariance"
            lines.append(f"| {m} | {LABELS[r['arm']]} minus {LABELS[r['comparator']]} | {r['mean']:+.6g} | "
                         f"{r['low']:+.6g} | {r['high']:+.6g} | {r['improved']}/{r['n']} |")
    lines += ["","Negative error differences favor the first arm. The energy metric is relative L1 over retained shells;",
              "the covariance metric is standardized RMSE over adjacent interior time pairs. No full-field or speed claim is made.",
              f"The worst shared coarse-allocation L1 residual was {max(d['shared_coarse_l1'] for d in objective):.3g}.",
              f"The worst flat and balanced primal–dual gaps were {max(d['flat_dual_gap'] for d in objective):.3g} "
              f"and {max(d['balanced_dual_gap'] for d in objective):.3g}. See CSV for per-split numerical enclosures.",""]
    (out/"report.md").write_text("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True,help="new directory, not an existing run")
    p.add_argument("--profile",choices=("study","fixture"),default="study")
    p.add_argument("--discard-inputs",action="store_true",help="remove generated coordinate archives after recording hashes and metrics")
    args = p.parse_args()
    out = args.out.resolve()
    if args.profile == "study":
        expected = {"apebench":"0.1.1","exponax":"0.1.0","jax":"0.4.35","jaxlib":"0.4.35",
                    "equinox":"0.11.10","optax":"0.2.4","chex":"0.1.87"}
        found = packages()
        bad = {k:(v,found[k]) for k,v in expected.items() if found[k] != v}
        if bad:
            raise RuntimeError(f"Install the frozen study requirements first. (Expected, found): {bad}")
    assert len(set(s for pair in SEEDS for s in pair))==2*len(SEEDS)
    out.mkdir(parents=True,exist_ok=False)
    config = {"profile":args.profile, "train_test_seeds":SEEDS, "endpoint_shuffle_seed":9821,
              "versions":packages(), "python":sys.version, "platform":platform.platform(),
              "environment":{k:os.environ.get(k) for k in ENV},
              "source_hashes":{k:sha256(HERE/k) for k in FILES}, "completed_splits":[]}
    write_json(out/"manifest.json",config)
    objectives, all_rows, split_scores = [],[],[]
    for i,pair in enumerate(SEEDS,1):
        here = out / f"split_{i}"
        here.mkdir()
        data, dst = here/"data", here/"evaluation"
        print(f"{args.profile}: split {i}/{len(SEEDS)} (train={pair[0]}, test={pair[1]})",flush=True)
        since = perf_counter()
        generate(data,pair,args.profile)
        source = data/"input.npz"
        validate(source,pair,args.profile,json.loads((data/"metadata.json").read_text()))
        file_hash = sha256(source)
        report,summary = evaluate(source,dst)
        objective, rows, scores = collect(i,pair,report,summary)
        write_csv(here/"objective.csv",[objective])
        write_csv(here/"scores.csv",rows)
        objectives.append(objective)
        all_rows.extend(rows)
        split_scores.append(scores)
        if args.discard_inputs:
            source.unlink()
        config["completed_splits"].append({"index":i,"train_seed":pair[0],"test_seed":pair[1],
                                           "generated_input_sha256":file_hash,
                                           "seconds":perf_counter()-since})
        write_json(out/"manifest.json",config)
    summarize(out,args.profile,objectives,all_rows,split_scores)
    print(f"Finished {len(SEEDS)} splits. Report: {out / 'report.md'}",flush=True)


if __name__ == "__main__":
    main()
