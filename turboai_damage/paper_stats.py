"""Bootstrap confidence intervals of the key test metrics (the numbers quoted in the README).

With a few dozen test cases, a difference between two mean errors is only meaningful if it exceeds
the sampling uncertainty. Each metric is a mean over test cases; it is resampled with replacement
(10 000 bootstrap samples) and the 2.5 and 97.5 percentiles give its 95 % confidence interval.

Usage::

    python -m turboai_damage.paper_stats            # writes results/paper_stats.json
"""
from __future__ import annotations

import json

import joblib
import numpy as np

from turboai_damage import RESULTS
from turboai_shape import DATA as SHAPE_DATA, RESULTS as SHAPE_RESULTS


def bootstrap_ci(values, n=10_000, seed=0):
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n, len(v)))].mean(1)
    return {"mean": float(v.mean()), "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
            "n": int(len(v))}


def main():
    d = np.load(SHAPE_DATA / "geo_v1.npz", allow_pickle=True)
    names = [str(n) for n in d["Y_names"]]
    X, Y, split = d["X"], d["Y"], d["split"]
    Mis = np.c_[d["Mis_SS"], d["Mis_PS"]]
    models = SHAPE_RESULTS / "models"
    pod = joblib.load(models / "podgp_mis.joblib")
    out = {"phase2": {}}
    for s in ("test", "ood"):
        k = split == s
        out["phase2"][s] = {"podgp_wall_mis": bootstrap_ci(np.abs(pod.predict(X[k])[0] - Mis[k]).mean(1))}
        for target, scale in (("zeta", 100.0), ("alpha2_deg", 1.0), ("mdot_per_m", 1.0)):
            gp = joblib.load(models / f"gp_{target}.joblib")
            err = np.abs(gp.predict(X[k])[0] - Y[k, names.index(target)]) * scale
            out["phase2"][s][f"gp_{target}"] = bootstrap_ci(err)
    for name in ("gnn", "gnn_aug"):
        f = SHAPE_RESULTS / name / "metrics.json"
        if f.exists():
            g = json.loads(f.read_text())
            out[name] = {s: {"wall_mis": bootstrap_ci(list(g[s]["per_case_wall_Mis_mae"].values()))}
                         for s in ("test", "ood") if s in g}
    defects = RESULTS / "defects" / "defects.json"
    if defects.exists():
        rec = json.loads(defects.read_text())["records"]
        out["defects"] = {"parametric_dmis_error": bootstrap_ci([r["dmis_signal"] for r in rec]),
                          "network_dmis_error": bootstrap_ci([r["dmis_error"] for r in rec]),
                          "parametric_dmis_error_region": bootstrap_ci([r["dmis_signal_region"] for r in rec]),
                          "network_dmis_error_region": bootstrap_ci([r["dmis_error_region"] for r in rec])}
    hybrid = RESULTS / "hybrid" / "metrics.json"
    if hybrid.exists():
        # paired comparisons: the difference between two models is resampled case by case, which is much
        # tighter than comparing two independent intervals (case-to-case difficulty cancels)
        h = json.loads(hybrid.read_text())
        out["hybrid"] = {}
        for s in ("test", "ood"):
            pc = np.array(list(h[s]["per_case"].values()))   # columns: parametric, hybrid (Mis); same for zeta [pp]
            out["hybrid"][s] = {"parametric_wall_mis": bootstrap_ci(pc[:, 0]), "hybrid_wall_mis": bootstrap_ci(pc[:, 1]),
                                "difference_wall_mis": bootstrap_ci(pc[:, 1] - pc[:, 0])}
            if pc.shape[1] >= 4:
                out["hybrid"][s] |= {"parametric_zeta_pp": bootstrap_ci(pc[:, 2]), "hybrid_zeta_pp": bootstrap_ci(pc[:, 3]),
                                     "difference_zeta_pp": bootstrap_ci(pc[:, 3] - pc[:, 2])}
        rec = h["defect_records"]
        sig = np.array([r["dmis_signal"] for r in rec])
        err = np.array([r["dmis_error"] for r in rec])
        dz_err = np.abs([r["d_zeta_net"] - r["d_zeta_cfd"] for r in rec]) * 100
        dz_par = np.abs([r["d_zeta_cfd"] for r in rec]) * 100
        out["hybrid"]["defects"] = {"parametric_dmis_error": bootstrap_ci(sig), "hybrid_dmis_error": bootstrap_ci(err),
                                    "difference_dmis": bootstrap_ci(err - sig),
                                    "parametric_dzeta_abs_error_pp": bootstrap_ci(dz_par),
                                    "hybrid_dzeta_abs_error_pp": bootstrap_ci(dz_err),
                                    "difference_dzeta_pp": bootstrap_ci(dz_err - dz_par)}
        lin = {r["id"]: r for r in h.get("linear_baseline", {}).get("defect_records", [])}
        if lin:
            cfd = {r["id"]: r["d_zeta_cfd"] for r in rec}
            ids = [r["id"] for r in rec if r["id"] in lin]
            lin_err = np.abs([lin[i]["d_zeta_net"] - cfd[i] for i in ids]) * 100
            hyb_err = np.abs([r["d_zeta_net"] - r["d_zeta_cfd"] for r in rec if r["id"] in lin]) * 100
            out["hybrid"]["defects"] |= {"linear_dzeta_abs_error_pp": bootstrap_ci(lin_err),
                                         "hybrid_minus_linear_dzeta_pp": bootstrap_ci(hyb_err - lin_err)}
        if "blind" in h and "records" in h["blind"]:
            b = h["blind"]["records"]
            out["hybrid"]["blind"] = {"hybrid_dzeta_abs_error_pp": bootstrap_ci(
                np.abs([r["d_zeta_net"] - r["d_zeta_cfd"] for r in b]) * 100),
                "parametric_dzeta_abs_error_pp": bootstrap_ci(np.abs([r["d_zeta_cfd"] for r in b]) * 100)}
    (RESULTS / "paper_stats.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
