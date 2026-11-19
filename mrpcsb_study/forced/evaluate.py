"""Exact fixed-coupling affine OU moments on independent held-out trajectory groups.

Inputs are raw, real, orthonormal FFT coefficients; forcing is additive drift
in those same coordinates. This is not a neural evaluation or a comparison
between the hierarchical and flat transport plans.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.special import logsumexp
from scipy.stats import t as student_t


ARMS = ("unforced_ou", "forced_ou", "train_mean_centered_ou")


def integrated_decay(t, rate):
    """Integral from 0 to t of exp(-rate*s) ds, including rate=0."""
    t, rate = np.broadcast_arrays(np.asarray(t, dtype=float), rate)
    return np.divide(
        -np.expm1(-rate * t), rate, out=t.copy(), where=rate != 0
    )


def flat_plan(x, y, terminal_gain, terminal_variance, tol=1e-8):
    """Uniform empirical endpoint SB coupling; solve in the log domain."""
    n, m = len(x), len(y)
    # The dropped row/column terms in the Gaussian transition kernel do
    # not change a coupling whose two marginals are fixed.
    xc, yc = x - x.mean(0), y - y.mean(0)
    logk = (xc * (terminal_gain / terminal_variance)) @ yc.T
    if not np.isfinite(logk).all():
        raise ValueError("Nonfinite endpoint kernel")
    u, v = np.zeros(n), np.zeros(m)
    loga, logb = -np.log(n), -np.log(m)
    scale = max(1.0, np.max(np.abs(logk)) / 10.0)
    temperatures = np.geomspace(scale, 1.0, num=8) if scale > 1.0 else [1.0]
    old_temperature, iterations = temperatures[0], 0
    for temperature in temperatures:
        u *= old_temperature / temperature
        v *= old_temperature / temperature
        l = logk / temperature
        target_tol = tol if temperature == 1.0 else max(tol, 1e-5)
        for _ in range(20000 if temperature == 1.0 else 1500):
            u = loga - logsumexp(l + v, axis=1)
            v = logb - logsumexp(l + u[:, None], axis=0)
            iterations += 1
            if iterations % 20 == 0:
                pi = np.exp(l + u[:, None] + v)
                res = max(
                    np.abs(pi.sum(1) - 1 / n).sum(),
                    np.abs(pi.sum(0) - 1 / m).sum(),
                )
                if res < target_tol:
                    break
        old_temperature = temperature
    raw = np.exp(logk + u[:, None] + v)
    residual = max(
        np.abs(raw.sum(1) - 1 / n).sum(),
        np.abs(raw.sum(0) - 1 / m).sum(),
    )
    # A floating-point residual with a sharp kernel is possible. Never
    # present a poor fit as exact. Round only a near-feasible plan and
    # explicitly report its original residual and feasible objective gap.
    if residual > 2e-3:
        raise RuntimeError(f"Endpoint Sinkhorn did not converge: L1 residual {residual:.3g}")
    pi = raw * np.minimum(1, (1 / n) / raw.sum(1))[:, None]
    pi *= np.minimum(1, (1 / m) / pi.sum(0))
    a, b = np.maximum(0.0, 1 / n - pi.sum(1)), np.maximum(0.0, 1 / m - pi.sum(0))
    if a.sum() > 1e-15:
        pi += np.outer(a, b) / a.sum()
    np.testing.assert_allclose(pi.sum(1), 1 / n, atol=1e-13)
    np.testing.assert_allclose(pi.sum(0), 1 / m, atol=1e-13)
    # Primal-Dual gap for sum(pi(log(pi)-logk-1)), KKT pi=exp(logk+u+v).
    nonzero = pi > 0.0
    primal = np.sum(pi[nonzero] * (np.log(pi[nonzero]) - logk[nonzero] - 1))
    dual = u.mean() + v.mean() - raw.sum()
    gap = max(0.0, float(primal - dual))
    if gap > 5e-2:
        raise RuntimeError(f"Endpoint solver feasible dual gap too large: {gap:.3g}")
    return pi, {"raw_marginal_l1": float(residual), "duality_gap": gap,
                "iterations": iterations}


def exact_moments(x, y, times, rates, sigma, forcing, pi):
    """Marginal mean, covariance diagonal, and deterministic forcing shift.

    For t in [0,T], D=diag(exp(-rates*t)), Q the integrated noise and
    B_t = Q_t D_{T-t}/Q_T. Conditioning gives A_t*x+B_t*y+shift_t and
    variance Q_t Q_{T-t}/Q_T, with A_t=D_t-B_t D_T.
    """
    t, T = times[:, None], times[-1]
    r, s = rates[None], sigma[None]
    q_t = s**2 * integrated_decay(t, 2 * r)
    q_u = s**2 * integrated_decay(T - t, 2 * r)
    q_T = q_t[-1]
    B = q_t * np.exp(-r * (T - t)) / q_T
    A = np.exp(-r * t) - B * np.exp(-r * T)
    shift = (integrated_decay(t, r) - B * integrated_decay(T, r)) * forcing
    mean = A * x.mean(0) + B * y.mean(0) + shift
    xc, yc = x - x.mean(0), y - y.mean(0)
    cross = np.einsum("ij,id,jd->d", pi, xc, yc, optimize=True)
    var = A**2 * (xc**2).mean(0) + B**2 * (yc**2).mean(0) + 2 * A * B * cross
    var += q_t * q_u / q_T
    if np.min(var) < -1e-12:
        raise ValueError(f"Negative exact variance {np.min(var)}")
    return mean, np.maximum(var, 0.0), shift


def metric_rows(paths, masks, means, variances, scale, k2, times):
    out = []
    for group, sample in enumerate(paths):
        truth_m = sample.mean(0)
        # This is the empirical-population second moment, not a sample
        # covariance. No finite-sample bias correction is used.
        truth_s = (sample**2).mean(0)
        for arm in ARMS:
            predicted_full_energy = (means[arm]**2 + variances[arm]) @ (0.5 / k2)
            truth_full_energy = truth_s @ (0.5 / k2)
            for metric, mask in masks.items():
                var_scale = scale[metric]
                m = means[arm][:, mask]
                score = ((m - truth_m[:, mask])**2).sum(1) / var_scale
                s = (m**2 + variances[arm][:, mask]).sum(1)
                shell_e = ((s - truth_s[:, mask].sum(1)) / var_scale)**2
                assert np.isfinite(score).all() and np.isfinite(shell_e).all()
                fields = {
                    "mean_rmse": score,
                    "second_moment_rmse": shell_e,
                    "retained_energy_relative_rmse": (
                        (predicted_full_energy - truth_full_energy) /
                        np.maximum(truth_full_energy, np.finfo(float).tiny)
                    )**2,
                }
                if metric != "forced_mode":
                    fields.pop("retained_energy_relative_rmse")
                for name, series in fields.items():
                    band = "all_retained" if name == "retained_energy_relative_rmse" else metric
                    for region, ind in [("interior", slice(1, -1)), ("endpoint", [-1])]:
                        out.append(dict(group=group, arm=arm, region=region,
                                        band=band, metric=name,
                                        score=float(np.sqrt(np.mean(series[ind])))))
    return out


def aggregate(rows, group_count):
    indexed = {(r["region"], r["band"], r["metric"], r["arm"], r["group"]): r["score"]
               for r in rows}
    diagnostics = sorted({(r["region"], r["band"], r["metric"]) for r in rows})
    out = []
    for key in diagnostics:
        for arm, comparator in (("forced_ou", "unforced_ou"),
                                ("forced_ou", "train_mean_centered_ou"),
                                ("train_mean_centered_ou", "unforced_ou")):
            scores = np.array([indexed[key + (arm, g)] for g in range(group_count)])
            base = np.array([indexed[key + (comparator, g)] for g in range(group_count)])
            d = scores - base
            hw = student_t.ppf(0.975, group_count - 1) * np.std(d, ddof=1) / np.sqrt(group_count)
            out.append(dict(region=key[0], band=key[1], metric=key[2],
                            arm=arm, comparator=comparator,
                            score=float(scores.mean()), comparator_score=float(base.mean()),
                            paired_difference=float(d.mean()), ci_low=float(d.mean() - hw),
                            ci_high=float(d.mean() + hw), groups=group_count))
    return out


def write_csv(filename, rows):
    with filename.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(out, meta, summary):
    lines = [
        "# Fixed-coupling APEBench forced-mode diagnostic", "",
        "This is a control experiment on exact affine-OU bridge mixtures, not neural training "
        "or a hierarchy comparison. Endpoints, transport plan, viscosity/drag damping, and "
        "diffusion were matched across all three arms. Only the additive reference drift "
        "changed: zero in raw coefficients, explicit simulator forcing, or relaxation "
        "toward the initial training mean. The last comparator tests whether explicit "
        "forcing adds value beyond simple empirical centering.", "",
        f"Used {meta['train_source_count']} training starts and {meta['train_target_count']} "
        f"training endpoints; {meta['test_group_count']} independent held-out groups of "
        f"{meta['test_trajectories_per_group']} trajectories each share one training split. "
        f"Retained {meta['dims']} real Fourier coordinates. "
        f"Directly forced wave(s): {meta['forced_waves']}.", "",
        f"Final marginal feasibility correction was preceded by an L1 residual of "
        f"{meta['flat_solver']['raw_marginal_l1']:.3g}; the feasible primal/dual gap was "
        f"{meta['flat_solver']['duality_gap']:.3g}. The plan is numerical; conditional "
        f"moments are evaluated analytically. Changing constant forcing shifts the interior "
        f"mean but leaves the centered covariance and both endpoint moments unchanged. "
        f"Maximum deterministic forced-coordinate mean shift vs zero: "
        f"{meta['maximum_forced_mean_shift']:.6g} (raw coefficient units).", "",
        "Lower is better. The forced-coordinate mean RMSE (the main diagnostic) divides "
        "the mean-vector squared error by the training endpoints' raw mean squared norm, "
        "then takes its square root after averaging over evaluation times within each test "
        "group. The second-moment RMSE instead divides the squared-norm expectation error "
        "by that same training norm before taking a time-RMSE. Retained-energy relative RMSE "
        "uses the per-time true retained kinetic energy to normalize. Each reported score "
        "averages group-level errors. Contrasts and unadjusted paired 95% t intervals are "
        "over the held-out groups, conditional on this training split; negative favors the "
        "named arm. Companion metrics are descriptive, not additional confirmatory tests.", "",
        "| Region | Band | Metric | Arm | Comparator | Arm score | Comparator score | Paired difference | 95% low | 95% high |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        lines.append("| {region} | {band} | {metric} | {arm} | {comparator} | "
                     "{score:.6g} | {comparator_score:.6g} | {paired_difference:.6g} | "
                     "{ci_low:.6g} | {ci_high:.6g} |".format(**r))
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(input_path, out):
    with np.load(input_path, allow_pickle=False) as data:
        x, y, paths, times, rates, sigma, forcing, waves = (
            np.asarray(data[k], float) for k in
            ("source", "target", "test_paths", "test_times", "rates", "sigma", "forcing", "waves")
        )
        mu = np.asarray(data["mu"], float) if "mu" in data.files else x.mean(0)
    if x.ndim != 2 or y.ndim != 2 or paths.ndim != 4 or len(times) < 3:
        raise ValueError("Expected endpoint matrices and test paths shaped [G,n,L,D]")
    D = x.shape[1]
    assert len(x) > 1 and len(y) > 1 and y.shape[1] == D
    assert paths.shape[0] >= 5 and paths.shape[1] > 1 and paths.shape[2:] == (len(times), D)
    assert rates.shape == sigma.shape == forcing.shape == mu.shape == (D,)
    assert times.ndim == 1 and times[0] == 0.0 and np.all(np.diff(times) > 0)
    assert np.all(rates >= 0) and np.all(sigma > 0)
    assert all(np.isfinite(a).all() for a in (x, y, paths, times, rates, sigma, forcing, waves, mu))
    assert waves.shape == (D, 2)
    k2 = (waves**2).sum(1)
    assert np.all(k2 > 0), "Zero/DC mode is not part of this audit"
    # Select by exported forcing support, independently of the outcomes.
    # Usually two real coordinates represent the conjugate mode, one with
    # an identically zero additive drift.
    eps = 1e-8 * np.max(np.abs(forcing))
    driven_waves = waves[np.abs(forcing) > eps]
    assert len(driven_waves), "Export did not retain the driving mode"
    active = (np.abs(forcing) > eps)
    # Support could include forcing on both representatives; wave sign does
    # not change shell identification.
    driven = np.any(np.all(waves[:, None] == driven_waves, axis=2) |
                    np.all(waves[:, None] == -driven_waves, axis=2), axis=1)
    drives_k4 = np.any(np.all(np.sort(np.abs(driven_waves), axis=1) == [0, 4], axis=1))
    if not drives_k4:
        raise ValueError(f"APEbench mode-4 drive was not found: {driven_waves}")
    masks = {"forced_coordinate": active, "forced_mode": driven}
    scales = {k: float(np.mean((np.concatenate([x[:, v], y[:, v]], axis=0)**2).sum(1)))
              for k, v in masks.items()}
    assert all(v > 0 for v in scales.values())
    start = perf_counter()
    terminal_gain = np.exp(-rates * times[-1])
    terminal_variance = sigma**2 * integrated_decay(times[-1], 2 * rates)
    pi, diagnostics = flat_plan(x, y, terminal_gain, terminal_variance)
    means, variances = {}, {}
    for name, f in zip(ARMS, (np.zeros(D), forcing, rates * mu)):
        means[name], variances[name], shift = exact_moments(x, y, times, rates, sigma, f, pi)
        np.testing.assert_allclose(shift[[0, -1]], 0, atol=1e-10)
        np.testing.assert_allclose(means[name][0], x.mean(0), atol=1e-10)
        np.testing.assert_allclose(means[name][-1], y.mean(0), atol=1e-10)
        np.testing.assert_allclose(variances[name], variances[ARMS[0]], atol=1e-10)
    rows = metric_rows(paths, masks, means, variances, scales, k2, times)
    summary = aggregate(rows, len(paths))
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "group_scores.csv", rows)
    write_csv(out / "paired_summary.csv", summary)
    meta = {
        "train_source_count": len(x), "train_target_count": len(y),
        "test_group_count": len(paths), "test_trajectories_per_group": paths.shape[1],
        "times": times.tolist(), "dims": D,
        "forced_waves": np.unique(driven_waves, axis=0).tolist(),
        "forcing": forcing[active].tolist(),
        "time_to_compute_seconds": perf_counter()-start,
        "flat_solver": diagnostics,
        "maximum_forced_mean_shift": float(np.max(np.abs(means["forced_ou"][:, active] - means["unforced_ou"][:, active]))),
        "primary": "interior forced-coordinate normalized mean RMSE; forced OU minus unforced OU",
        "normalization": "mean squared source and target raw coefficients on the training set only",
        "uncertainty": "unadjusted paired t intervals over independently generated test groups on one training split",
        "scope": "fixed plan, analytic affine-OU mixture; no retraining; not a full-field or hierarchy result",
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2)+"\n")
    write_report(out, meta, summary)
    primary = next(r for r in summary if r["region"] == "interior" and
                   r["band"] == "forced_coordinate" and r["metric"] == "mean_rmse" and
                   r["arm"] == "forced_ou" and r["comparator"] == "unforced_ou")
    print(json.dumps({"metadata": meta, "primary": primary}, indent=2))
    return meta, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.input, args.out)
