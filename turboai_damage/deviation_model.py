"""Hybrid surrogate for blades with local deviations (as-manufactured, eroded or damaged blades).

A blade with a local deviation is described in two parts:

* its **nominal shape**, the closest member of the six-parameter family. The parametric surrogates of
  phase 2 (POD-GP for the loading, GP for the loss) predict its performance accurately;
* its **deviation** from that nominal shape, a normal displacement delta(s) of the surface. Its effect
  is predicted by a graph neural network that works on the blade surface.

The surface network (NVIDIA PhysicsNeMo MeshGraphNet) has one node per loading station, 120 per side.
Node inputs (8): x / c_ax, side (0 suction, 1 pressure), nominal isentropic Mach number from the
POD-GP, the deviation delta / c and its first and second derivatives along x (slope and curvature
change of the surface), M2is and incidence / 10. Edges join stations 1, 2, 4, 8 and 16 apart along
each side, and the two sides at the leading and trailing edges, so that the network sees the deviation
both locally and over the surrounding blade. Node outputs (2): the change of isentropic Mach number
caused by the deviation and a per-node contribution whose mean is the change of loss coefficient.

The network is trained on the deviation-augmented blades of data set ``geo_v2`` (family shapes, each
with one random bump or dent): the targets are the differences between CFD and the parametric
prediction of the nominal shape. Test, out-of-family and defect blades are never used in training.

Three networks are trained with different seeds (and validation splits) and averaged; their spread
estimates the model uncertainty. A linear sensitivity model fitted to the same data (the analogue of
an adjoint sensitivity map) is evaluated alongside as a baseline.

Usage::

    python -m turboai_damage.deviation_model train --out results/hybrid --seed 0   # -> member0/
    python -m turboai_damage.deviation_model train --out results/hybrid --seed 1
    python -m turboai_damage.deviation_model train --out results/hybrid --seed 2
    python -m turboai_damage.deviation_model evaluate --out results/hybrid
    python -m turboai_damage.deviation_model predict --bump lower,0.62,0.05,0.006 --M2is 0.9 --png bump.png
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch

from turboai.config import PROFILE, RUNS
from turboai_damage import RESULTS
from turboai_shape import RESULTS as SHAPE_RESULTS
from turboai.dataset.phase1_campaign import wall_curves
from turboai.geometry.parametric import PARAMS, Ls89Family

XS = np.linspace(0, 1, 120)
NAMES = list(PARAMS)
HOPS = (1, 2, 4, 8, 16)
_CACHE = {}


def family():
    if "family" not in _CACHE:
        _CACHE["family"] = Ls89Family(PROFILE)
    return _CACHE["family"]


def parametric_models():
    """The phase 2 parametric surrogates of the nominal blade: POD-GP of the loading, GP of the loss."""
    if "pod" not in _CACHE:
        models = SHAPE_RESULTS / "models"
        _CACHE["pod"] = joblib.load(models / "podgp_mis.joblib")
        _CACHE["gp_zeta"] = joblib.load(models / "gp_zeta.joblib")
    return _CACHE["pod"], _CACHE["gp_zeta"]


# geometry ------------------------------------------------------------------------------------------
def deviation(params, bumps=None, actual=None):
    """Normal deviation of a blade from its nominal family shape on the loading stations.

    ``params`` are the family parameters of the nominal shape; the actual outline is either the nominal
    shape plus ``bumps`` or given directly as ``actual`` (same point layout as Ls89Family.make).
    Returns (2, 120): suction side, pressure side; positive outwards, in chord units. Computed from
    the two outlines, so it applies to any deviation, not only to Gaussian bumps.
    """
    fam = family()
    n = len(fam.s)
    nominal = fam.make(params)
    actual = fam.make(params, bumps) if actual is None else actual
    out = []
    for side in ("lower", "upper"):                    # lower = suction side, upper = pressure side
        idx = np.r_[0, np.arange(len(nominal) - 1, n - 1, -1), n - 1] if side == "lower" else np.arange(n)
        a, b = nominal[idx], actual[idx]
        t = np.gradient(a, axis=0)
        normal = np.c_[t[:, 1], -t[:, 0]] / np.linalg.norm(t, axis=1, keepdims=True)
        delta = np.einsum("ij,ij->i", b - a, normal) / fam.chord
        if side == "upper":
            # walking from LE to TE, the right-hand normal points out of the blade on the lower side
            # and into it on the upper side: flip so that 'outwards' is positive on both sides
            delta = -delta
        x = (a[:, 0] - nominal[:, 0].min()) / np.ptp(nominal[:, 0])
        o = np.argsort(x)
        out.append(np.interp(XS, x[o], delta[o]))
    return np.array(out)


def surface_graph(mis_base, delta, M2is, alpha):
    """Graph of the blade surface: node inputs (240, 8), edge index (2, E), edge inputs (E, 3)."""
    d1 = np.gradient(delta, XS, axis=1)
    d2 = np.gradient(d1, XS, axis=1)
    n = len(XS)
    side = np.r_[np.zeros(n), np.ones(n)]
    x = np.c_[np.r_[XS, XS], side, mis_base, delta.ravel() * 100, d1.ravel() * 10, d2.ravel(),
              np.full(2 * n, M2is), np.full(2 * n, alpha / 10)]
    src, dst, feat = [], [], []
    for off in (0, n):
        for h in HOPS:
            i = np.arange(n - h)
            for a, b in ((i, i + h), (i + h, i)):
                src.append(a + off)
                dst.append(b + off)
                feat.append(np.c_[XS[b] - XS[a], np.ones(len(a)), np.full(len(a), h / HOPS[-1])])
    for i in (0, n - 1):                               # leading and trailing edges join the two sides
        src += [np.array([i]), np.array([n + i])]
        dst += [np.array([n + i]), np.array([i])]
        feat += [np.array([[0.0, 0.0, 0.0]])] * 2
    ei = np.vstack([np.concatenate(src), np.concatenate(dst)])
    return x, ei, np.vstack(feat)


# data -------------------------------------------------------------------------------------------------
def load_cases(name, splits, fit_nominal=False):
    """Inputs and CFD targets of every finished case of a data set, with the parametric baseline.

    With ``fit_nominal`` the nominal shape is not taken from the plan but fitted to the actual outline
    (Ls89Family.fit), as it would have to be for a measured blade.
    """
    d = RUNS / "datasets" / name
    rows = [r for r in json.loads((d / "plan.json").read_text()) if r["split"] in splits]
    pod, gp_zeta = parametric_models()
    cases = []
    for r in rows:
        res = d / "cases" / r["id"] / "results"
        if not (res / "scalars.json").exists():
            continue
        params = {k: r[k] for k in NAMES}
        actual = family().make(params, r.get("bumps"))
        if fit_nominal:
            params, _ = family().fit(actual)
        X = np.r_[[params[k] for k in NAMES], r["M2is"], r["alpha_in_deg"]][None]
        base = pod.predict(X)[0][0]
        cases.append({"id": r["id"], "split": r["split"], "params": params, "bumps": r.get("bumps"),
                      "M2is": r["M2is"], "alpha": r["alpha_in_deg"], "mis_base": base,
                      "zeta_base": float(gp_zeta.predict(X)[0][0]),
                      "mis_cfd": np.concatenate(wall_curves(res, XS)),
                      "zeta_cfd": json.loads((res / "scalars.json").read_text())["zeta"],
                      "delta": deviation(params, actual=actual)})
    return cases


def to_graph(c):
    from torch_geometric.data import Data
    x, ei, e = surface_graph(c["mis_base"], c["delta"], c["M2is"], c["alpha"])
    g = {"x": torch.tensor(x, dtype=torch.float32), "e": torch.tensor(e, dtype=torch.float32),
         "G": Data(edge_index=torch.tensor(ei, dtype=torch.long), num_nodes=len(x))}
    if "mis_cfd" in c:
        g["y"] = torch.tensor(c["mis_cfd"] - c["mis_base"], dtype=torch.float32)[:, None]
        g["dz"] = torch.tensor([c["zeta_cfd"] - c["zeta_base"]], dtype=torch.float32)
    return g


def new_model(hidden=64, steps=8):
    from physicsnemo.models.meshgraphnet import MeshGraphNet
    return MeshGraphNet(8, 3, 2, processor_size=steps, hidden_dim_processor=hidden, hidden_dim_node_encoder=hidden,
                        hidden_dim_edge_encoder=hidden, hidden_dim_node_decoder=hidden, aggregation="sum")


def collate(graphs):
    """Merge several surface graphs into one disconnected graph (one forward pass for a mini-batch)."""
    from torch_geometric.data import Data
    sizes = [len(g["x"]) for g in graphs]
    offsets = np.cumsum([0] + sizes[:-1])
    b = {"x": torch.cat([g["x"] for g in graphs]), "e": torch.cat([g["e"] for g in graphs]),
         "G": Data(edge_index=torch.cat([g["G"].edge_index + o for g, o in zip(graphs, offsets)], 1),
                   num_nodes=sum(sizes)),
         "batch": torch.cat([torch.full((n,), k, dtype=torch.long) for k, n in enumerate(sizes)])}
    if "y" in graphs[0]:
        b["y"] = torch.cat([g["y"] for g in graphs])
        b["dz"] = torch.cat([g["dz"] for g in graphs])
    return b


def predict(model, stats, g):
    """Change of loading (one value per node) and of loss coefficient (one per graph) caused by the deviation.

    ``g`` is one graph from :func:`to_graph` or a mini-batch from :func:`collate`.
    """
    xn = (g["x"] - stats["x"][0]) / stats["x"][1]
    en = (g["e"] - stats["e"][0]) / stats["e"][1]
    out = model(xn, en, g["G"])
    batch = g.get("batch", torch.zeros(len(out), dtype=torch.long))
    n_graphs = int(batch.max()) + 1
    counts = torch.bincount(batch, minlength=n_graphs).to(out.dtype)
    dz = torch.zeros(n_graphs, dtype=out.dtype).index_add_(0, batch, out[:, 1]) / counts
    dz = dz if "batch" in g else dz[0]
    return out[:, 0] * stats["y_scale"], dz * stats["z_scale"]


# training ---------------------------------------------------------------------------------------------
def train(out, epochs=400, hidden=64, steps=8, lr=1e-3, n_val=15, seed=0, threads=4, data="geo_v2",
          batch_size=8):
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cases = load_cases(data, ("train",))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(cases))
    val = [cases[i] for i in order[:n_val]]            # held out from the training blades (monitoring only)
    tr = [cases[i] for i in order[n_val:]]
    graphs_tr, graphs_val = [to_graph(c) for c in tr], [to_graph(c) for c in val]
    X = torch.cat([g["x"] for g in graphs_tr])
    E = torch.cat([g["e"] for g in graphs_tr])
    stats = {"x": (X.mean(0), X.std(0) + 1e-6), "e": (E.mean(0), E.std(0) + 1e-6),
             "y_scale": float(torch.cat([g["y"] for g in graphs_tr]).std()),
             "z_scale": float(torch.stack([g["dz"] for g in graphs_tr]).std())}
    model = new_model(hidden, steps)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=lr / 30)
    print(f"surface network: {sum(p.numel() for p in model.parameters()) / 1e3:.0f}k parameters, "
          f"{len(tr)} training / {len(val)} validation blades", flush=True)
    print(f"parametric surrogate alone on the validation blades: loading error "
          f"{np.mean([float(g['y'].abs().mean()) for g in graphs_val]):.4f}, loss error "
          f"{np.mean([abs(float(g['dz'][0])) for g in graphs_val]) * 100:.3f} pp", flush=True)
    history, t0 = [], time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(graphs_tr)).tolist()
        for k in range(0, len(perm), batch_size):
            b = collate([graphs_tr[i] for i in perm[k:k + batch_size]])
            dmis, dz = predict(model, stats, b)
            loss = (torch.mean((dmis - b["y"][:, 0]) ** 2) / stats["y_scale"] ** 2
                    + torch.mean(((dz - b["dz"]) / stats["z_scale"]) ** 2))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        if ep % 20 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                ev = [(float((predict(model, stats, g)[0] - g["y"][:, 0]).abs().mean()),
                       float(abs(predict(model, stats, g)[1] - g["dz"][0])) * 100) for g in graphs_val]
            m = np.mean(ev, axis=0)
            history.append([ep, float(m[0]), float(m[1])])
            print(f"epoch {ep:4d}  validation: loading error {m[0]:.4f}, loss error {m[1]:.3f} pp"
                  f"  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    torch.save({"model": model.state_dict(), "stats": stats, "cfg": {"hidden": hidden, "steps": steps},
                "seed": seed, "data": data}, out / "surface_net.pt")
    (out / "train_history.json").write_text(json.dumps(history))


def load(out):
    """The trained surface networks in ``out``: every ``member*/surface_net.pt`` (an ensemble trained
    with different seeds and validation splits), or a single ``surface_net.pt``."""
    out = Path(out)
    files = sorted(out.glob("member*/surface_net.pt")) or [out / "surface_net.pt"]
    members = []
    for f in files:
        ck = torch.load(f, weights_only=False)
        model = new_model(**ck["cfg"])
        model.load_state_dict(ck["model"])
        members.append((model.eval(), ck["stats"]))
    return Ensemble(members)


class Ensemble:
    """Average of several surface networks; their spread is an estimate of the model uncertainty."""

    def __init__(self, members):
        self.members = members

    @torch.no_grad()
    def __call__(self, c):
        """Change of loading (240,) and of loss coefficient for case ``c``, with their ensemble spread."""
        g = to_graph(c)
        out = [predict(m, s, g) for m, s in self.members]
        dmis = np.stack([o[0].numpy() for o in out])
        dz = np.array([float(o[1]) for o in out])
        return dmis.mean(0), float(dz.mean()), dmis.std(0), float(dz.std())


# linear baseline ------------------------------------------------------------------------------------
class LinearSensitivity:
    """Data-driven linear response to the deviation: Delta = W [delta, delta (M2is - 0.85)].

    The analogue of an adjoint sensitivity map, learnt from the same training blades as the network:
    the predicted change is linear in the deviation (a dent gives exactly the opposite of a bump), with
    a sensitivity that may vary linearly with the exit Mach number. Ridge regression without intercept;
    the regularisation is chosen by leave-one-out cross-validation.
    """

    def fit(self, cases):
        from sklearn.linear_model import RidgeCV
        X = np.array([self._features(c) for c in cases])
        Y = np.array([np.r_[c["mis_cfd"] - c["mis_base"], c["zeta_cfd"] - c["zeta_base"]] for c in cases])
        self.scale = Y.std(0) + 1e-9
        self.model = RidgeCV(alphas=np.logspace(-3, 4, 36), fit_intercept=False).fit(X, Y / self.scale)
        return self

    @staticmethod
    def _features(c):
        d = c["delta"].ravel() * 100
        return np.r_[d, d * (c["M2is"] - 0.85) * 10]

    def __call__(self, c):
        y = self.model.predict(self._features(c)[None])[0] * self.scale
        return y[:-1], float(y[-1]), np.zeros(len(y) - 1), 0.0


# evaluation -----------------------------------------------------------------------------------------
def evaluate_mode(corrector, fit_nominal):
    """Errors of parametric model + ``corrector`` on the test, out-of-family and defect blades.

    ``corrector(case)`` returns (Delta Mis, Delta zeta, std of Delta Mis, std of Delta zeta). The
    nominal shape is taken from the plan or, with ``fit_nominal``, fitted to the outline.
    """
    result = {}
    for name, splits, label in (("geo_v1", ("test",), "test"), ("geo_v1", ("ood",), "ood")):
        cases = load_cases(name, splits, fit_nominal)
        e_base, e_hyb, e_zb, e_zh = [], [], [], []
        for c in cases:
            dmis, dz, _, _ = corrector(c)
            e_base.append(float(np.abs(c["mis_base"] - c["mis_cfd"]).mean()))
            e_hyb.append(float(np.abs(c["mis_base"] + dmis - c["mis_cfd"]).mean()))
            e_zb.append(abs(c["zeta_base"] - c["zeta_cfd"]) * 100)
            e_zh.append(abs(c["zeta_base"] + dz - c["zeta_cfd"]) * 100)
        result[label] = {"n": len(cases), "parametric_wall_mis_mae": float(np.mean(e_base)),
                         "hybrid_wall_mis_mae": float(np.mean(e_hyb)),
                         "parametric_zeta_mae_pp": float(np.mean(e_zb)), "hybrid_zeta_mae_pp": float(np.mean(e_zh)),
                         "per_case": {c["id"]: [b, h, zb, zh]
                                      for c, b, h, zb, zh in zip(cases, e_base, e_hyb, e_zb, e_zh)}}

    # defects (and the blind test, if computed): change relative to the intact blade of the same set
    records = []
    for name, label in (("defect_v1", "defects"), ("defect_blind", "blind")):
        if not (RUNS / "datasets" / name / "cases" / "baseline" / "results" / "scalars.json").exists():
            continue
        result[label], recs = _changes(corrector, name, fit_nominal)
        if label == "defects":
            records = recs
        else:
            result[label]["records"] = [{k: r[k] for k in ("id", "bump", "d_zeta_cfd", "d_zeta_net", "d_zeta_std",
                                                           "dmis_signal", "dmis_error")} for r in recs]
    return result, records


def _changes(corrector, name, fit_nominal):
    """Predicted and CFD changes caused by each defect of data set ``name`` relative to its intact blade."""
    cases = {c["id"]: c for c in load_cases(name, ("defect",), fit_nominal)}
    base = cases.pop("baseline")
    dm0, dz0, _, _ = corrector(base)
    records = []
    for cid, c in cases.items():
        dm, dz, _, dz_std = corrector(c)
        d_cfd = c["mis_cfd"] - base["mis_cfd"]
        # predicted change = change of the parametric part (zero if the nominal shape is known) + correction
        d_net = (c["mis_base"] - base["mis_base"]) + (dm - dm0)
        dz_net = (c["zeta_base"] - base["zeta_base"]) + (dz - dz0)
        region = np.abs(d_cfd) > 0.2 * np.abs(d_cfd).max()
        records.append({"id": cid, "bump": c["bumps"][0],
                        "d_zeta_cfd": c["zeta_cfd"] - base["zeta_cfd"], "d_zeta_net": dz_net, "d_zeta_std": dz_std,
                        "dmis_signal": float(np.abs(d_cfd).mean()), "dmis_error": float(np.abs(d_net - d_cfd).mean()),
                        "dmis_signal_region": float(np.abs(d_cfd[region]).mean()),
                        "dmis_error_region": float(np.abs(d_net - d_cfd)[region].mean()),
                        "dmis_cfd": d_cfd.tolist(), "dmis_net": d_net.tolist()})
    dz_c = np.array([r["d_zeta_cfd"] for r in records])
    dz_n = np.array([r["d_zeta_net"] for r in records])
    summary = {
        "n": len(records),
        "dmis_parametric_error": float(np.mean([r["dmis_signal"] for r in records])),
        "dmis_hybrid_error": float(np.mean([r["dmis_error"] for r in records])),
        "dmis_parametric_error_region": float(np.mean([r["dmis_signal_region"] for r in records])),
        "dmis_hybrid_error_region": float(np.mean([r["dmis_error_region"] for r in records])),
        "dzeta_mean_abs_cfd_pp": float(np.abs(dz_c).mean() * 100),
        "dzeta_hybrid_mae_pp": float(np.abs(dz_n - dz_c).mean() * 100),
        "dzeta_correlation": float(np.corrcoef(dz_c, dz_n)[0, 1]) if len(records) > 2 else None}
    return summary, records


def defect_sweep(corrector, n=111, amp=0.006, width=0.05, M2is=0.90, alpha=0.0):
    """Prediction for a defect slid continuously along each side of the intact blade.

    The same defect as the CFD campaign (amplitude ``amp`` chord, width ``width``), at ``n`` positions
    per side and sign. Returns, for each side and sign, the loss change with its ensemble spread and
    the change of loading, plus the mean time of one prediction.
    """
    pod, _ = parametric_models()
    params = {k: 0.0 for k in NAMES}
    X = np.r_[[0.0] * len(NAMES), M2is, alpha][None]
    base = {"M2is": M2is, "alpha": alpha, "mis_base": pod.predict(X)[0][0], "delta": deviation(params)}
    dm0, dz0, _, _ = corrector(base)
    s0 = np.linspace(0.06, 0.94, n)
    out, t0, n_pred = {"s0": s0.tolist()}, time.perf_counter(), 0
    for side in ("lower", "upper"):
        for sign, kind in ((1, "bump"), (-1, "dent")):
            dz_list, dz_std, dmis_list = [], [], []
            for s in s0:
                bump = [{"side": side, "s0": float(s), "w": width, "amp": sign * amp}]
                dm, dz, _, sd = corrector({**base, "delta": deviation(params, bump)})
                dz_list.append(dz - dz0)
                dz_std.append(sd)
                dmis_list.append((dm - dm0).round(5).tolist())
                n_pred += 1
            out[f"{side}_{kind}"] = {"d_zeta": dz_list, "d_zeta_std": dz_std, "d_mis": dmis_list}
    out["ms_per_prediction"] = (time.perf_counter() - t0) / n_pred * 1e3
    return out


def _strip(res):
    return {k: {q: v for q, v in res[k].items() if q not in ("per_case", "records")}
            for k in ("test", "ood", "defects", "blind") if k in res}


def evaluate(out, data="geo_v2"):
    """Evaluate the trained surface networks and the linear baseline; writes metrics.json and figures."""
    ens = load(out)
    out = Path(out)
    print(f"{len(ens.members)} surface network(s)", flush=True)
    result, records = evaluate_mode(ens, fit_nominal=False)
    fitted, _ = evaluate_mode(ens, fit_nominal=True)
    linear = LinearSensitivity().fit(load_cases(data, ("train",)))
    lin_res, lin_records = evaluate_mode(linear, fit_nominal=False)
    for label, res in (("hybrid, nominal shape known", result), ("hybrid, nominal shape fitted", fitted),
                       ("linear sensitivity, nominal shape known", lin_res)):
        print(label)
        for k, v in _strip(res).items():
            print(f"  {k:8s}", json.dumps({q: round(x, 5) if isinstance(x, float) else x for q, x in v.items()}),
                  flush=True)
    result["n_members"] = len(ens.members)
    result["fitted_nominal"] = _strip(fitted)
    result["linear_baseline"] = {**_strip(lin_res),
                                 "defect_records": [{k: r[k] for k in ("id", "d_zeta_net", "dmis_error")}
                                                    for r in lin_records]}
    (out / "metrics.json").write_text(json.dumps({**result, "defect_records": records}, indent=1))
    sweep = defect_sweep(ens)
    sweep["linear"] = defect_sweep(linear)
    (out / "defect_sweep.json").write_text(json.dumps(sweep))
    print(f"defect sweep: {sweep['ms_per_prediction']:.1f} ms per prediction", flush=True)
    figures(out, records, sweep)


def figures(out, records, sweep=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    side_name = {"upper": "pressure side", "lower": "suction side"}
    fig, axs = plt.subplots(2, 2, figsize=(13, 9.2), sharex=True)
    for row, side in enumerate(("lower", "upper")):
        for col, (sign, kind) in enumerate(((1, "bump (+0.4 mm)"), (-1, "dent (-0.4 mm)"))):
            ax = axs[row, col]
            pts = sorted([r for r in records if r["bump"]["side"] == side and np.sign(r["bump"]["amp"]) == sign],
                         key=lambda r: r["bump"]["s0"])
            s = [r["bump"]["s0"] for r in pts]
            ax.plot(s, [r["d_zeta_cfd"] * 100 for r in pts], "ko-", ms=5, lw=1, label="CFD (OpenFOAM)")
            ax.plot(s, [r["d_zeta_net"] * 100 for r in pts], "s", c="C0", ms=5,
                    label="hybrid surrogate (never trained on these)")
            if sweep is not None:
                key = f"{side}_{'bump' if sign > 0 else 'dent'}"
                z, sd = np.array(sweep[key]["d_zeta"]) * 100, np.array(sweep[key]["d_zeta_std"]) * 100
                ax.plot(sweep["s0"], z, "-", c="C0", lw=1.2, alpha=0.8, label="hybrid surrogate, continuous sweep")
                if sd.any():
                    ax.fill_between(sweep["s0"], z - 2 * sd, z + 2 * sd, color="C0", alpha=0.15, lw=0,
                                    label="ensemble spread (2 standard deviations)")
                if "linear" in sweep:
                    ax.plot(sweep["s0"], np.array(sweep["linear"][key]["d_zeta"]) * 100, "--", c="C2", lw=1.2,
                            label="linear sensitivity (adjoint-like)")
            ax.axhline(0, c="C1", lw=1.5, ls=":", label="parametric surrogate alone (blind)")
            ax.set_title(f"{kind} on the {side_name[side]}", fontsize=11)
            ax.grid(alpha=0.3)
            if col == 0:
                ax.set_ylabel("Δζ [% points]")
            if row == 1:
                ax.set_xlabel("defect position s (0 = leading edge, 1 = trailing edge)")
    h, lab = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=9, ncol=3, loc="lower center")
    fig.suptitle("Damage-tolerance map predicted without CFD: loss change caused by a 0.4 mm defect")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out / "defect_map_hybrid.png", dpi=150)
    plt.close(fig)

    picks = []
    for side in ("lower", "upper"):
        for sign in (1, -1):
            cand = [r for r in records if r["bump"]["side"] == side and np.sign(r["bump"]["amp"]) == sign]
            if cand:
                picks.append(max(cand, key=lambda r: r["dmis_signal_region"]))
    n = len(XS)
    fig, axs = plt.subplots(2, 2, figsize=(13, 8))
    for ax, r in zip(axs.ravel(), picks):
        k = 0 if r["bump"]["side"] == "lower" else 1
        sl = slice(k * n, (k + 1) * n)
        ax.plot(XS, np.array(r["dmis_cfd"])[sl], "k", lw=2, label="CFD")
        ax.plot(XS, np.array(r["dmis_net"])[sl], "C0--", lw=1.8, label="hybrid surrogate")
        ax.axhline(0, c="C1", ls=":", lw=1.5, label="parametric surrogate alone")
        ax.set_title(f"{'bump' if r['bump']['amp'] > 0 else 'dent'} on the {side_name[r['bump']['side']]} at "
                     f"s = {r['bump']['s0']:.2f}", fontsize=11)
        ax.set_xlabel("x / c_ax")
        ax.set_ylabel("ΔMis (defect - intact)")
        ax.grid(alpha=0.3)
    axs[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("Change of the blade loading caused by a defect, on the side of the defect")
    fig.tight_layout()
    fig.savefig(out / "defect_loading_hybrid.png", dpi=150)
    plt.close(fig)


def predict_command(out, params="", bump="", M2is=0.9, alpha=0.0, png=None):
    """CLI: performance of a blade = family member ``params`` + deviation ``bump``, without CFD.

    ``bump`` is side,s0,width,amplitude (side 'lower' = suction, 'upper' = pressure; amplitude in
    chords, e.g. 0.006 = 0.4 mm). Prints the nominal and the hybrid predictions; optionally plots
    the loading.
    """
    ens = load(out)
    pod, gp_zeta = parametric_models()
    p = {k: 0.0 for k in NAMES} | ({k: float(v) for k, v in (kv.split("=") for kv in params.split(",") if kv)}
                                   if params else {})
    bumps = None
    if bump:
        side, s0, w, amp = bump.split(",")
        bumps = [{"side": side, "s0": float(s0), "w": float(w), "amp": float(amp)}]
    X = np.r_[[p[k] for k in NAMES], M2is, alpha][None]
    mis0 = pod.predict(X)[0][0]
    z0, z0_std = (float(v[0]) for v in gp_zeta.predict(X))
    t0 = time.perf_counter()
    dmis, dz, dmis_std, dz_std = ens({"M2is": M2is, "alpha": alpha, "mis_base": mis0, "delta": deviation(p, bumps)})
    ms = (time.perf_counter() - t0) * 1e3
    print(f"nominal blade (parametric GP): zeta = {z0 * 100:.3f} % (+/- {2 * z0_std * 100:.3f})")
    print(f"with the deviation (hybrid):   zeta = {(z0 + dz) * 100:.3f} %  "
          f"(change {dz * 100:+.3f} +/- {2 * dz_std * 100:.3f} points, {ms:.0f} ms)")
    print("RESULT_JSON " + json.dumps({"zeta_nominal_pct": z0 * 100, "zeta_nominal_2sigma_pct": 2 * z0_std * 100,
                                       "zeta_pct": (z0 + dz) * 100, "dzeta_pp": dz * 100,
                                       "dzeta_2sigma_pp": 2 * dz_std * 100, "milliseconds": ms,
                                       "max_abs_dMis": float(np.abs(dmis).max())}))
    if png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        n = len(XS)
        for k, (name, colour) in enumerate((("suction side", "C0"), ("pressure side", "C3"))):
            sl = slice(k * n, (k + 1) * n)
            ax.plot(XS, mis0[sl], c=colour, lw=1, ls="--", label=f"nominal, {name}")
            ax.plot(XS, (mis0 + dmis)[sl], c=colour, lw=2, label=f"with deviation, {name}")
        ax.set_xlabel("x / c_ax")
        ax.set_ylabel("isentropic Mach number")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(png, dpi=150)
        print("wrote", png)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Hybrid surrogate for blades with local deviations.")
    ap.add_argument("cmd", choices=["train", "evaluate", "predict"])
    ap.add_argument("--out", default=str(RESULTS / "hybrid"))
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--data", default="geo_v2", help="training data set (blades with deviations)")
    ap.add_argument("--seed", type=int, default=0, help="train: random seed; the model goes to <out>/member<seed>")
    ap.add_argument("--params", default="", help="predict: family parameters, e.g. a2=0.01,b1=-0.1")
    ap.add_argument("--bump", default="", help="predict: side,s0,width,amplitude, e.g. lower,0.62,0.05,0.006")
    ap.add_argument("--M2is", type=float, default=0.9)
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--png", help="predict: also plot the loading to this file")
    a = ap.parse_args()
    if a.cmd == "train":
        train(Path(a.out) / f"member{a.seed}", a.epochs, seed=a.seed, threads=a.threads, data=a.data)
    elif a.cmd == "evaluate":
        evaluate(a.out, a.data)
    else:
        predict_command(a.out, a.params, a.bump, a.M2is, a.alpha, a.png)
