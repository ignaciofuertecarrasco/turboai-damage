"""Damage-tolerance map of the vane: how much does a small surface defect cost, and where?

A blade in service is never the drawing: deposits, erosion, foreign-object damage and manufacturing
deviations change the surface locally by a few tenths of a millimetre. This module quantifies the
aerodynamic effect of such a defect as a function of where it sits on the blade, in three ways:

* **CFD** (ground truth): the defect campaign ``defect_v1`` (turboai.dataset.phase2_campaign
  plan-defects), a 0.4 mm protrusion or dent at 12 positions on each side, plus the intact blade.
* **Graph network**: a PhysicsNeMo MeshGraphNet trained by turboai_shape.meshgraphnet. It reads the
  mesh, so it can evaluate any defect in about a second.
* **Parametric surrogate**: a model driven by shape parameters cannot represent a local defect and
  therefore predicts no change at all. Its error equals the size of the change itself.

What is compared is the **change** caused by each defect relative to the intact blade, computed in
the same way for CFD and for the network (defect minus intact):

* the change of the blade loading, Delta Mis(x), on 120 stations per side, over the whole blade and
  inside the region the defect affects (where |Delta Mis| exceeds 20 % of its maximum);
* the change of the loss coefficient, Delta zeta. For the network it is estimated from the predicted
  field (mass-averaged total pressure over the graph nodes near the outlet); the same estimator
  applied to the CFD values on the same nodes shows how much of the difference is due to the
  estimator itself.

Usage::

    python -m turboai_damage.defect_map --model results/gnn/mgn.pt --out results/defects
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from turboai.config import GAMMA, RUNS
from turboai_damage import RESULTS
from turboai.dataset.phase1_campaign import wall_curves
from turboai.flow import isentropic_mach

G = GAMMA
XS = np.linspace(0, 1, 120)


def outlet_state(g, values, band=3e-3):
    """Loss coefficient and exit angle estimated from node values (Mach, p/P0, u/a0, v/a0) near the outlet.

    Nodes within ``band`` of the most downstream node are used; on the uniform far-field node grid
    each node represents about the same area, so the weights are the local mass flux rho * u.
    """
    pos = g["pos"].numpy()
    k = pos[:, 0] > pos[:, 0].max() - band
    M, pr, u, v = (np.asarray(values)[k, i] for i in range(4))
    rho = pr * (1 + 0.5 * (G - 1) * M ** 2)               # p / T up to a constant factor
    w = rho * u
    p0 = pr * (1 + 0.5 * (G - 1) * M ** 2) ** (G / (G - 1))
    P02, p2 = (w * p0).sum() / w.sum(), pr.mean()
    return (1 - P02) / (1 - p2), math.degrees(math.atan2((w * v).sum(), (w * u).sum()))


def graph_curves(g, values):
    """Wall isentropic Mach of a graph (suction side then pressure side) on the stations XS."""
    from turboai_shape.meshgraphnet import wall_curve
    mis = isentropic_mach(np.clip(np.asarray(values)[:, 1], 1e-3, 1.0), 1.0)
    c = wall_curve(g, mis)
    return np.r_[np.interp(XS, *c["SS"]), np.interp(XS, *c["PS"])]


def evaluate(model_path, name="defect_v1"):
    """Network vs CFD on every simulated defect. Returns a list of per-defect records."""
    from turboai_shape.meshgraphnet import _normalise, load_model, make_graph
    d = RUNS / "datasets" / name
    plan = {r["id"]: r for r in json.loads((d / "plan.json").read_text())}
    model, stats = load_model(model_path)

    def one(cid):
        res = d / "cases" / cid / "results"
        sc = json.loads((res / "scalars.json").read_text())
        g = make_graph(res / "fields.npz", plan[cid]["M2is"], plan[cid]["alpha_in_deg"])
        _normalise([g], stats)
        with torch.no_grad():
            p = (model(g["xn"], g["en"], g["G"]) * stats["y"][1] + stats["y"][0]).numpy()
        return {"zeta": sc["zeta"], "alpha2": sc["alpha2_deg"], "mis": np.concatenate(wall_curves(res, XS)),
                "mis_net": graph_curves(g, p), "zeta_net": outlet_state(g, p)[0],
                "zeta_nodes": outlet_state(g, g["y"].numpy())[0]}

    base = one("baseline")
    records = []
    for cid, r in plan.items():
        if cid == "baseline" or not (d / "cases" / cid / "results" / "scalars.json").exists():
            continue
        c = one(cid)
        d_cfd, d_net = c["mis"] - base["mis"], c["mis_net"] - base["mis_net"]
        region = np.abs(d_cfd) > 0.2 * np.abs(d_cfd).max()
        records.append({
            "id": cid, "bump": r["bumps"][0],
            "d_zeta_cfd": c["zeta"] - base["zeta"],
            "d_zeta_net": c["zeta_net"] - base["zeta_net"],
            "d_zeta_nodes": c["zeta_nodes"] - base["zeta_nodes"],
            "d_alpha2_cfd": c["alpha2"] - base["alpha2"],
            "dmis_signal": float(np.abs(d_cfd).mean()),            # = error of a parametric model
            "dmis_signal_region": float(np.abs(d_cfd[region]).mean()),
            "dmis_error": float(np.abs(d_net - d_cfd).mean()),
            "dmis_error_region": float(np.abs(d_net - d_cfd)[region].mean()),
            "dmis_cfd": d_cfd.tolist(), "dmis_net": d_net.tolist(),
        })
    return records


def summary(records):
    a = lambda k: float(np.mean([r[k] for r in records]))  # noqa: E731
    dz_c = np.array([r["d_zeta_cfd"] for r in records])
    dz_n = np.array([r["d_zeta_net"] for r in records])
    return {"n": len(records),
            "dmis_parametric_error": a("dmis_signal"), "dmis_network_error": a("dmis_error"),
            "dmis_parametric_error_region": a("dmis_signal_region"), "dmis_network_error_region": a("dmis_error_region"),
            "dzeta_mean_abs_cfd_pp": float(np.abs(dz_c).mean() * 100),
            "dzeta_network_mae_pp": float(np.abs(dz_n - dz_c).mean() * 100),
            "dzeta_correlation": float(np.corrcoef(dz_c, dz_n)[0, 1]) if len(records) > 2 else None,
            "dzeta_estimator_bias_pp": float(np.mean([r["d_zeta_nodes"] - r["d_zeta_cfd"] for r in records]) * 100)}


def figures(records, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    side_name = {"upper": "pressure side", "lower": "suction side"}

    # 1. loss change vs defect position
    fig, axs = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True)
    for row, side in enumerate(("lower", "upper")):
        for col, (sign, kind) in enumerate(((1, "bump (+0.4 mm)"), (-1, "dent (-0.4 mm)"))):
            ax = axs[row, col]
            pts = sorted([r for r in records if r["bump"]["side"] == side and np.sign(r["bump"]["amp"]) == sign],
                         key=lambda r: r["bump"]["s0"])
            s = [r["bump"]["s0"] for r in pts]
            ax.plot(s, [r["d_zeta_cfd"] * 100 for r in pts], "ko-", ms=5, lw=1, label="CFD (OpenFOAM)")
            ax.plot(s, [r["d_zeta_net"] * 100 for r in pts], "s--", c="C0", ms=5, lw=1, label="graph network")
            ax.axhline(0, c="C1", lw=1.5, ls=":", label="parametric surrogate (blind)")
            ax.set_title(f"{kind} on the {side_name[side]}", fontsize=11)
            ax.grid(alpha=0.3)
            if col == 0:
                ax.set_ylabel("Δζ [% points]")
            if row == 1:
                ax.set_xlabel("defect position s (0 = leading edge, 1 = trailing edge)")
    axs[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("Damage-tolerance map at M2is = 0.90: loss change caused by a local 0.4 mm defect")
    fig.tight_layout()
    fig.savefig(out / "defect_map.png", dpi=150)
    plt.close(fig)

    # 2. loading change of four representative defects (largest CFD effect on each side and sign)
    picks = []
    for side in ("lower", "upper"):
        for sign in (1, -1):
            cand = [r for r in records if r["bump"]["side"] == side and np.sign(r["bump"]["amp"]) == sign]
            if cand:
                picks.append(max(cand, key=lambda r: r["dmis_signal_region"]))
    n = len(XS)
    fig, axs = plt.subplots(2, 2, figsize=(13, 8))
    for ax, r in zip(axs.ravel(), picks):
        side = 0 if r["bump"]["side"] == "lower" else 1
        sl = slice(side * n, (side + 1) * n)
        ax.plot(XS, np.array(r["dmis_cfd"])[sl], "k", lw=2, label="CFD")
        ax.plot(XS, np.array(r["dmis_net"])[sl], "C0--", lw=1.8, label="graph network")
        ax.axhline(0, c="C1", ls=":", lw=1.5, label="parametric surrogate")
        ax.set_title(f"{'bump' if r['bump']['amp'] > 0 else 'dent'} on the {side_name[r['bump']['side']]} at "
                     f"s = {r['bump']['s0']:.2f}", fontsize=11)
        ax.set_xlabel("x / c_ax")
        ax.set_ylabel("ΔMis (defect - intact)")
        ax.grid(alpha=0.3)
    axs[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("Change of the blade loading caused by a defect, on the side of the defect")
    fig.tight_layout()
    fig.savefig(out / "defect_loading.png", dpi=150)
    plt.close(fig)


def cfd_map(name, out):
    """CFD-only analysis: effect of each defect on loss, exit angle and mass flow, and its split into a
    part that changes sign with the defect (linear, (D+ - D-)/2) and a part that does not
    ((D+ + D-)/2). A linear (adjoint) sensitivity can only represent the first."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = RUNS / "datasets" / name
    plan = {r["id"]: r for r in json.loads((d / "plan.json").read_text())}
    scal = lambda cid: json.loads((d / "cases" / cid / "results" / "scalars.json").read_text())  # noqa: E731
    base = scal("baseline")
    table = {}
    for cid, r in plan.items():
        if cid == "baseline" or not (d / "cases" / cid / "results" / "scalars.json").exists():
            continue
        s, b = scal(cid), r["bumps"][0]
        table[(b["side"], round(b["s0"], 2), int(np.sign(b["amp"])))] = {
            "zeta": (s["zeta"] - base["zeta"]) * 100, "alpha2": s["alpha2_deg"] - base["alpha2_deg"],
            "mdot": (s["mdot_per_m"] / base["mdot_per_m"] - 1) * 100}
    side_name = {"lower": "suction side", "upper": "pressure side"}
    quantities = (("zeta", "Δζ [% points]"), ("alpha2", "Δα₂ [deg]"), ("mdot", "Δṁ [%]"))
    fig, axs = plt.subplots(len(quantities), 2, figsize=(13, 10), sharex=True)
    decomposition = {}
    for col, side in enumerate(("lower", "upper")):
        s0s = sorted({k[1] for k in table if k[0] == side})
        for row, (q, label) in enumerate(quantities):
            ax = axs[row, col]
            for sign, style, lab in ((1, "o-", "bump +0.4 mm"), (-1, "s--", "dent -0.4 mm")):
                pts = [(s0, table[(side, s0, sign)][q]) for s0 in s0s if (side, s0, sign) in table]
                ax.plot(*zip(*pts), style, ms=5, label=lab)
            pairs = [s0 for s0 in s0s if (side, s0, 1) in table and (side, s0, -1) in table]
            lin = [(table[(side, s0, 1)][q] - table[(side, s0, -1)][q]) / 2 for s0 in pairs]
            non = [(table[(side, s0, 1)][q] + table[(side, s0, -1)][q]) / 2 for s0 in pairs]
            decomposition[f"{side}_{q}"] = {"s0": pairs, "linear": lin, "nonlinear": non}
            ax.axhline(0, c="0.5", lw=0.8)
            ax.grid(alpha=0.3)
            if col == 0:
                ax.set_ylabel(label)
            if row == 0:
                ax.set_title(side_name[side])
        axs[-1, col].set_xlabel("defect position s (0 = leading edge, 1 = trailing edge)")
    axs[0, 0].legend(frameon=False)
    fig.suptitle(f"Effect of a local 0.4 mm defect on the LS-89 at M2is = {plan['baseline']['M2is']:.2f} (CFD, "
                 f"{len(table)} defects; intact blade: ζ = {base['zeta'] * 100:.2f} %)")
    fig.tight_layout()
    fig.savefig(out / "defect_map_cfd.png", dpi=150)
    plt.close(fig)
    mechanism_figure(d, out, max(table, key=lambda k: table[k]["zeta"]))
    rows = [{"side": k[0], "s0": k[1], "sign": k[2], **v} for k, v in sorted(table.items())]
    (out / "defects_cfd.json").write_text(json.dumps({"baseline": {k: base[k] for k in ("zeta", "alpha2_deg", "mdot_per_m")},
                                                      "defects": rows, "decomposition": decomposition}, indent=1))
    return rows, decomposition


def mechanism_figure(d, out, worst):
    """Why the worst defect costs so much: loading and Mach field of the intact and damaged blades."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.path import Path as MPath
    from matplotlib.tri import Triangulation
    from turboai.post.post import order_outline
    side, s0, sign = worst
    cid = f"{side[0]}{'p' if sign > 0 else 'm'}{s0:.2f}".replace(".", "")
    fig = plt.figure(figsize=(14, 5.6))
    ax0 = fig.add_axes([0.05, 0.12, 0.36, 0.78])
    for case, colour, label in (("baseline", "0.4", "intact blade"), (cid, "C3", "0.4 mm bump")):
        c = wall_curves(d / "cases" / case / "results", XS)
        ax0.plot(XS, c[0], c=colour, lw=2, label=label)
        ax0.plot(XS, c[1], c=colour, lw=1.2, ls="--")
    ax0.axhline(1, c="0.6", lw=0.8, ls=":")
    ax0.set_xlabel("x / c_ax")
    ax0.set_ylabel("isentropic Mach number")
    ax0.legend(frameon=False)
    ax0.grid(alpha=0.3)
    ax0.set_title("Blade loading (solid: suction side)")
    fields = {k: np.load(d / "cases" / k / "results" / "fields.npz") for k in ("baseline", cid)}
    blade = order_outline(fields[cid]["blade_xy"].astype(float)) * 1e3
    k_peak = np.argmax(wall_curves(d / "cases" / cid / "results", XS)[0])
    x_peak = blade[:, 0].min() + XS[k_peak] * np.ptp(blade[:, 0])
    y_peak = blade[np.abs(blade[:, 0] - x_peak).argmin(), 1]
    for k, (case, title) in enumerate((("baseline", "intact"), (cid, "with the bump"))):
        ax = fig.add_axes([0.44 + k * 0.27, 0.12, 0.25, 0.78])
        f = fields[case]
        P = f["xy"] * 1e3
        tri = Triangulation(P[:, 0], P[:, 1])
        c = P[tri.triangles]
        bl = order_outline(f["blade_xy"].astype(float)) * 1e3
        tri.set_mask(MPath(bl).contains_points(c.mean(1)) | (np.ptp(c[:, :, 0], 1) > 3) | (np.ptp(c[:, :, 1], 1) > 3))
        im = ax.tricontourf(tri, f["Mach"], levels=np.linspace(0.5, 1.35, 35), cmap="turbo", extend="both")
        ax.tricontour(tri, f["Mach"], levels=[1.0], colors="w", linewidths=1.0)
        ax.fill(*bl.T, color="0.75")
        ax.set_aspect("equal")
        ax.set_xlim(x_peak - 6, x_peak + 6)
        ax.set_ylim(y_peak - 8, y_peak + 6)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Mach number, {title} (white: M = 1)", fontsize=10)
    fig.colorbar(im, ax=fig.axes[-2:], shrink=0.8, label="Mach number")
    fig.suptitle("Mechanism: a 0.4 mm bump placed where the flow is already sonic creates a supersonic pocket "
                 "closed by a shock")
    fig.savefig(out / "defect_mechanism.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Damage-tolerance map: CFD vs graph network.")
    ap.add_argument("--model", default=str(__import__('turboai_shape').RESULTS / "gnn" / "mgn.pt"))
    ap.add_argument("--name", default="defect_v1")
    ap.add_argument("--out", default=str(RESULTS / "defects"))
    ap.add_argument("--cfd-only", action="store_true", help="only the CFD analysis, no network")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfd_map(a.name, out)
    if a.cfd_only:
        return
    records = evaluate(a.model, a.name)
    s = summary(records)
    (out / "defects.json").write_text(json.dumps({"summary": s, "records": records}, indent=1, default=float))
    figures(records, out)
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
