"""Comprehensive, leakage-free evaluation of the final selected DILI model.

FINAL MODEL: descriptor + Morgan-2048 -> bagged XGBoost (the thesis headline recipe),
trained on the official TDC-DILI train_val set.

LEAKAGE CONTROL
  * The decision threshold (Youden's J) and the Platt probability calibrator are fit
    ONLY on 5-fold out-of-fold (OOF) predictions within train_val. They never see the
    test set or any external labels. The threshold is then FROZEN and applied unchanged
    to the official test set and to every external dataset.
  * Before external evaluation, every molecule overlapping the training set by exact
    SMILES, canonical (standardised) SMILES, InChIKey-14 skeleton, or Bemis-Murcko
    scaffold is removed; the removed counts and final N are reported.

OUTPUTS (results/ and results/figures/)
  final_test_metrics.csv, external_metrics_full.csv, overlap_removal.csv,
  chemspace_similarity.csv, calibration_final.csv, fp_fn_analysis.csv,
  predictions_test.csv, predictions_ext_dilirank.csv, predictions_ext_dilist.csv,
  reproducibility.json, figures/roc_pr_curves.png, figures/confusion_matrices.png,
  figures/calibration_plot.png, figures/chemspace_similarity.png

Run:  python -m src.improved.comprehensive_eval
"""
from __future__ import annotations
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, precision_recall_curve, brier_score_loss
from xgboost import XGBClassifier

from src.improved.tdc_official import build_features
from src.improved import eval_utils as EU

BASE = Path(__file__).resolve().parents[2]
TDC = BASE / "data" / "tdc_benchmark" / "admet_group" / "dili"
RES = BASE / "results"
FIG = RES / "figures"
SEED = 42
N_BAG = 5


def clean(X):
    return np.clip(np.nan_to_num(X, nan=0, posinf=0, neginf=0), -1e6, 1e6).astype(np.float32)


def train_bagged(X_tr, y_tr, X_pred, n_bag=N_BAG, base_seed=0):
    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    pred = np.zeros(X_pred.shape[0])
    for sd in range(n_bag):
        clf = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.85,
                            colsample_bytree=0.7, reg_lambda=1.0, scale_pos_weight=spw,
                            eval_metric="logloss", random_state=base_seed + sd, n_jobs=4, verbosity=0)
        clf.fit(X_tr, y_tr)
        pred += clf.predict_proba(X_pred)[:, 1]
    return pred / n_bag


# ---------------------------------------------------------------- data loading
def load_frame(path, smi_col, y_col, name_col=None):
    df = pd.read_csv(path)
    df = df.rename(columns={smi_col: "smiles", y_col: "label"})
    df["smiles"] = df["smiles"].astype(str)
    df["mol_id"] = df[name_col].astype(str) if name_col else [f"mol_{i}" for i in range(len(df))]
    df = df[df["smiles"].str.strip().astype(bool)]
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"]); df["label"] = df["label"].astype(int)
    # standardise + structural keys
    df["std_smiles"] = df["smiles"].map(EU.standardize_smiles)
    df = df.dropna(subset=["std_smiles"])
    df["ikey14"] = df["std_smiles"].map(EU.inchikey14)
    df["scaffold"] = df["std_smiles"].map(EU.scaffold)
    return df


def dedup_conflicts(df, tag):
    """Drop duplicate InChIKey-14; drop molecules whose duplicates carry conflicting labels."""
    n0 = len(df)
    lab_per_key = df.groupby("ikey14")["label"].nunique()
    conflict_keys = set(lab_per_key[lab_per_key > 1].index)
    df = df[~df["ikey14"].isin(conflict_keys)]
    n_conflict = n0 - len(df)
    df = df.drop_duplicates(subset="ikey14", keep="first")
    n_dup = n0 - n_conflict - len(df)
    print(f"[{tag}] raw={n0}  conflicting-label removed={n_conflict}  duplicate removed={n_dup}  -> {len(df)}")
    return df.reset_index(drop=True), n_conflict, n_dup


def main():
    RES.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
    rng_info = {"seed": SEED, "n_bag": N_BAG}

    # ---- training data (official TDC train_val) ----
    tr = load_frame(TDC / "train_val.csv", "Drug", "Y")
    tr, _, _ = dedup_conflicts(tr, "train_val")
    y_tr = tr["label"].values
    X_tr = clean(build_features(list(tr["std_smiles"]), {}, use_molformer=False))
    train_keys = set(tr["ikey14"]); train_std = set(tr["std_smiles"])
    train_raw = set(tr["smiles"]); train_scaf = set(s for s in tr["scaffold"] if s)

    # ---- Task 4: threshold + calibrator on OOF (leakage-free) ----
    scaf = tr["scaffold"].fillna(tr["std_smiles"]).values
    oof = np.zeros(len(y_tr))
    for a, b in GroupKFold(5).split(X_tr, y_tr, groups=scaf):
        oof[b] = train_bagged(X_tr[a], y_tr[a], X_tr[b])
    thr = EU.youden_threshold(y_tr, oof)
    platt = LogisticRegression().fit(oof.reshape(-1, 1), y_tr)
    print(f"[threshold] Youden's J on train_val OOF = {thr:.4f}")
    rng_info["threshold_youden_J"] = round(thr, 4)
    rng_info["oof_auroc_trainval"] = round(EU.full_panel(y_tr, oof, thr)["AUROC"], 4)

    # ---- final model on ALL train_val ----
    def predict(std_smiles):
        X = clean(build_features(list(std_smiles), {}, use_molformer=False))
        raw = train_bagged(X_tr, y_tr, X)
        cal = platt.predict_proba(raw.reshape(-1, 1))[:, 1]
        return raw, cal

    # ---- official test set ----
    te = load_frame(TDC / "test.csv", "Drug", "Y")
    te, _, _ = dedup_conflicts(te, "test")
    te_raw, te_cal = predict(te["std_smiles"])
    y_te = te["label"].values

    # ---- external datasets ----
    ext_specs = [
        ("DILIrank", BASE / "data" / "dili_with_smiles.csv", "smiles", "label", "drug_name"),
        ("DILIst",   BASE / "data" / "external" / "dilist_official_1279.csv", "smiles", "label", "CompoundName"),
    ]
    overlap_rows, metric_rows, chem_rows, pred_frames = [], [], [], {}

    # test-set metrics row
    def metric_row(name, y, prob, n_removed=None):
        p = EU.full_panel(y, prob, thr)
        ci = EU.bootstrap_ci(y, prob, thr, seed=SEED)
        row = {"dataset": name, "n": p["n"], "pos_rate": round(p["pos"] / p["n"], 3),
               "threshold": round(thr, 4)}
        for k in ["AUROC", "PR_AUC", "ACC", "F1", "MCC", "Sensitivity", "Specificity"]:
            row[k] = round(p[k], 3)
            row[f"{k}_CI"] = f"[{ci[k][0]:.3f}, {ci[k][1]:.3f}]"
        row.update(Precision=round(p["Precision"], 3), NPV=round(p["NPV"], 3),
                   Brier=round(p["Brier"], 3), TP=p["TP"], TN=p["TN"], FP=p["FP"], FN=p["FN"])
        return row

    metric_rows.append(metric_row("TDC_test", y_te, te_raw))

    # save test predictions
    def save_preds(name, df, prob):
        yhat = (prob >= thr).astype(int)
        out = pd.DataFrame({
            "dataset": name, "mol_id": df["mol_id"].values, "SMILES": df["std_smiles"].values,
            "y_true": df["label"].values, "y_prob": np.round(prob, 4), "y_pred": yhat,
            "threshold": round(thr, 4),
            "error_type": [EU.error_type(a, b) for a, b in zip(df["label"].values, yhat)],
        })
        return out
    pred_frames["TDC_test"] = save_preds("TDC_test", te, te_raw)
    pred_frames["TDC_test"].to_csv(RES / "predictions_test.csv", index=False)

    ext_eval = {}
    for name, path, sc, yc, nc in ext_specs:
        ext = load_frame(path, sc, yc, nc)
        ext, n_conf, n_dup = dedup_conflicts(ext, name)
        n_start = len(ext)
        # Task 7: remove train overlap by 4 criteria
        exact = ext["smiles"].isin(train_raw)
        canon = ext["std_smiles"].isin(train_std)
        ikey = ext["ikey14"].isin(train_keys)
        removed_mol = int((exact | canon | ikey).sum())
        ext = ext[~(exact | canon | ikey)].copy()
        scaf_ov = ext["scaffold"].isin(train_scaf)
        removed_scaf = int(scaf_ov.sum())
        ext_novel = ext[~scaf_ov].copy()  # scaffold-disjoint external
        overlap_rows.append({
            "dataset": name, "after_dedup": n_start, "conflicting_labels": n_conf, "duplicates": n_dup,
            "removed_exact_or_canonical_or_inchikey": removed_mol,
            "external_after_molecule_overlap": len(ext),
            "removed_scaffold_overlap": removed_scaf,
            "external_novel_scaffold": len(ext_novel),
        })
        # evaluate on molecule-disjoint external (primary) and novel-scaffold external
        raw, cal = predict(ext["std_smiles"])
        metric_rows.append(metric_row(f"{name}_external", ext["label"].values, raw))
        pred_frames[name] = save_preds(f"{name}_external", ext, raw)
        pred_frames[name].to_csv(RES / f"predictions_ext_{name.lower()}.csv", index=False)
        ext_eval[name] = (ext, raw)
        if len(ext_novel) >= 30 and ext_novel["label"].nunique() == 2:
            rawn, _ = predict(ext_novel["std_smiles"])
            metric_rows.append(metric_row(f"{name}_external_novel_scaffold", ext_novel["label"].values, rawn))
        # Task 8: chemical-space similarity
        simmax = EU.max_tanimoto_to_train(list(ext["std_smiles"]), list(tr["std_smiles"]))
        bins = [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01)]
        for lo, hi in bins:
            mask = (simmax >= lo) & (simmax < hi)
            yb = ext["label"].values[mask]; pb = raw[mask]
            if mask.sum() >= 10 and len(np.unique(yb)) == 2:
                pan = EU.full_panel(yb, pb, thr)
                chem_rows.append({"dataset": name, "sim_range": f"[{lo:.1f},{hi:.1f})", "n": int(mask.sum()),
                                  "AUROC": round(pan["AUROC"], 3), "ACC": round(pan["ACC"], 3),
                                  "MCC": round(pan["MCC"], 3)})
            else:
                chem_rows.append({"dataset": name, "sim_range": f"[{lo:.1f},{hi:.1f})", "n": int(mask.sum()),
                                  "AUROC": np.nan, "ACC": np.nan, "MCC": np.nan})

    # ---- write metric / overlap / chem tables ----
    pd.DataFrame([metric_rows[0]]).to_csv(RES / "final_test_metrics.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(RES / "external_metrics_full.csv", index=False)
    pd.DataFrame(overlap_rows).to_csv(RES / "overlap_removal.csv", index=False)
    pd.DataFrame(chem_rows).to_csv(RES / "chemspace_similarity.csv", index=False)

    # ---- calibration (Task 10) ----
    brier_raw = brier_score_loss(y_te, te_raw); brier_cal = brier_score_loss(y_te, te_cal)
    pd.DataFrame([{"set": "TDC_test", "Brier_uncalibrated": round(brier_raw, 4),
                   "Brier_platt_calibrated": round(brier_cal, 4)}]).to_csv(RES / "calibration_final.csv", index=False)

    # ---- FP/FN analysis (Task 10) ----
    fpfn = []
    for name, pf in pred_frames.items():
        for et in ("FP", "FN"):
            sub = pf[pf["error_type"] == et].copy()
            sub = sub.reindex(sub["y_prob"].sub(thr).abs().sort_values(ascending=False).index).head(10)
            for _, r in sub.iterrows():
                fpfn.append({"dataset": name, "error_type": et, "mol_id": r["mol_id"],
                             "SMILES": r["SMILES"], "y_true": r["y_true"], "y_prob": r["y_prob"]})
    pd.DataFrame(fpfn).to_csv(RES / "fp_fn_analysis.csv", index=False)

    # ---- plots ----
    _plot_roc_pr(y_te, te_raw, ext_eval)
    _plot_confusion(metric_rows)
    _plot_calibration(y_te, te_raw, te_cal, brier_raw, brier_cal)
    _plot_chemspace(pd.DataFrame(chem_rows))

    # ---- reproducibility ----
    import sklearn, xgboost, rdkit, scipy
    rng_info["libraries"] = {"python": platform.python_version(), "numpy": np.__version__,
                             "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
                             "xgboost": xgboost.__version__, "rdkit": rdkit.__version__,
                             "scipy": scipy.__version__}
    rng_info["platform"] = platform.platform()
    (RES / "reproducibility.json").write_text(json.dumps(rng_info, indent=2))

    print("\n[done] wrote metric tables, prediction CSVs, figures and reproducibility.json to results/")
    print(pd.DataFrame(metric_rows)[["dataset", "n", "AUROC", "PR_AUC", "ACC", "F1", "MCC",
                                     "Sensitivity", "Specificity"]].to_string(index=False))


# ---------------------------------------------------------------- plotting
def _plot_roc_pr(y_te, te_raw, ext_eval):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    series = [("TDC test", y_te, te_raw)] + [(f"{n} external", e["label"].values, p) for n, (e, p) in ext_eval.items()]
    for name, y, p in series:
        fpr, tpr, _ = roc_curve(y, p)
        a1.plot(fpr, tpr, label=f"{name} (AUROC {EU.full_panel(y, p, 0.5)['AUROC']:.3f})")
        pr, rc, _ = precision_recall_curve(y, p)
        a2.plot(rc, pr, label=f"{name} (PR-AUC {EU.full_panel(y, p, 0.5)['PR_AUC']:.3f})")
    a1.plot([0, 1], [0, 1], "--", c="grey", lw=0.7); a1.set_xlabel("False positive rate"); a1.set_ylabel("True positive rate"); a1.set_title("ROC curves"); a1.legend(fontsize=8)
    a2.set_xlabel("Recall"); a2.set_ylabel("Precision"); a2.set_title("Precision-Recall curves"); a2.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(FIG / "roc_pr_curves.png", dpi=200); plt.close()


def _plot_confusion(metric_rows):
    rows = [r for r in metric_rows if "novel_scaffold" not in r["dataset"]]
    fig, axes = plt.subplots(1, len(rows), figsize=(3.4 * len(rows), 3.2))
    if len(rows) == 1:
        axes = [axes]
    for ax, r in zip(axes, rows):
        cm = np.array([[r["TN"], r["FP"]], [r["FN"], r["TP"]]])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=11)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["True 0", "True 1"])
        ax.set_title(f"{r['dataset']}\nMCC={r['MCC']:.2f}", fontsize=9)
    plt.tight_layout(); plt.savefig(FIG / "confusion_matrices.png", dpi=200); plt.close()


def _plot_calibration(y, raw, cal, br, bc):
    from sklearn.calibration import calibration_curve
    fig, ax = plt.subplots(figsize=(5, 4.6))
    for prob, lab, b in [(raw, "uncalibrated", br), (cal, "Platt-calibrated", bc)]:
        fx, mx = calibration_curve(y, prob, n_bins=8, strategy="quantile")
        ax.plot(mx, fx, "o-", label=f"{lab} (Brier {b:.3f})")
    ax.plot([0, 1], [0, 1], "--", c="grey", lw=0.8, label="perfect")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration (reliability) on TDC test"); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(FIG / "calibration_plot.png", dpi=200); plt.close()


def _plot_chemspace(df):
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for name in df["dataset"].unique():
        sub = df[df["dataset"] == name]
        ax.plot(sub["sim_range"], sub["AUROC"], "o-", label=f"{name} AUROC")
    ax.axhline(0.5, ls="--", c="grey", lw=0.7)
    ax.set_xlabel("Max Tanimoto similarity to training set"); ax.set_ylabel("AUROC")
    ax.set_title("External performance vs chemical-space similarity"); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(FIG / "chemspace_similarity.png", dpi=200); plt.close()


if __name__ == "__main__":
    main()
