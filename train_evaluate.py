"""
Steps 6 + 7: train a Random Forest and evaluate it honestly.

The same model is evaluated with three cross-validation schemes:

    random   : pixels are split at random. Neighbouring, near-identical
               pixels of the same polygon end up in both train and test.
    polygon  : whole polygons are held out (grouped by polygon ID).
    block    : whole 2 x 2 km blocks are held out (grouped by block ID).
               The model is always tested on areas it has never seen.
               This is the main result.

The gap between "random" and "block" shows how much a naive split
overstates accuracy.

Usage (from the project folder, after extract_pixels.py):
    python train_evaluate.py

Input:   data/processed/pixels.csv
Outputs: results/cv_summary.csv          macro F1 / accuracy per scheme
         results/per_class_f1.csv        per-class F1 per scheme
         results/block_cv_report.txt     full report for the block scheme
         figures/confusion_matrix_block.png
         figures/split_comparison.png
         figures/feature_importance.png
         models/random_forest.joblib     final model (trained on all labels)
"""
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold

PIXELS_PATH = Path("data/processed/pixels.csv")
RESULTS_DIR = Path("results")
FIG_DIR = Path("figures")
MODEL_PATH = Path("models/random_forest.joblib")

FEATURES = ["B02", "B03", "B04", "B08", "B11", "B12", "NDVI", "NDWI", "NDBI"]
CLASS_NAMES = {0: "Water", 1: "Vegetation", 2: "Built-up", 3: "Bare soil"}

MAX_PIXELS_PER_POLYGON = 100  # large polygons are sub-sampled to this size
N_SPLITS = 5                  # folds per cross-validation
N_REPEATS = 3                 # repeat each CV with different random seeds
SEED = 42

RF_PARAMS = dict(
    n_estimators=200,
    class_weight="balanced",  # water has far more pixels than bare soil
    n_jobs=-1,
    random_state=SEED,
)


def load_and_sample() -> pd.DataFrame:
    df = pd.read_csv(PIXELS_PATH)
    df = df.dropna(subset=FEATURES)
    # Cap pixels per polygon so that a few large polygons do not dominate.
    # Every polygon still contributes, so spatial coverage is unchanged.
    # (shuffle once, then keep the first N pixels of each polygon)
    sampled = (
        df.sample(frac=1, random_state=SEED)
        .groupby("polygon_id")
        .head(MAX_PIXELS_PER_POLYGON)
        .reset_index(drop=True)
    )
    print(f"Pixels: {len(df)} extracted -> {len(sampled)} after capping at "
          f"{MAX_PIXELS_PER_POLYGON} per polygon")
    table = sampled.groupby("class").agg(
        pixels=("polygon_id", "size"),
        polygons=("polygon_id", "nunique"),
        blocks=("block_id", "nunique"),
    )
    table.index = table.index.map(CLASS_NAMES)
    print(table.to_string())
    return sampled


def make_splits(scheme: str, X, y, df, seed: int):
    if scheme == "random":
        return StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed).split(X, y)
    if scheme == "polygon":
        return StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=seed).split(
            X, y, groups=df["polygon_id"])
    if scheme == "block":
        return StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=seed).split(
            X, y, groups=df["block_id"])
    raise ValueError(scheme)


def cross_validate(scheme: str, df: pd.DataFrame, seed: int, keep_details: bool = False):
    """Out-of-fold predictions for one scheme and one seed."""
    X, y = df[FEATURES].to_numpy(), df["class"].to_numpy()
    y_pred = np.full_like(y, -1)
    fold_rows, importances = [], []

    for fold, (tr, te) in enumerate(make_splits(scheme, X, y, df, seed)):
        model = RandomForestClassifier(**RF_PARAMS).fit(X[tr], y[tr])
        y_pred[te] = model.predict(X[te])

        test = df.iloc[te]
        fold_rows.append({
            "fold": fold,
            "macro_f1": f1_score(y[te], y_pred[te], average="macro",
                                 labels=list(CLASS_NAMES), zero_division=0),
            **{f"test_polygons_{CLASS_NAMES[k]}": test.loc[test["class"] == k, "polygon_id"].nunique()
               for k in CLASS_NAMES},
        })
        if keep_details:
            # Permutation importance on the held-out fold: how much does the
            # score drop when one feature is shuffled? More honest than the
            # built-in impurity importance, which is measured on training data.
            pi = permutation_importance(model, X[te], y[te], scoring="f1_macro",
                                        n_repeats=5, random_state=seed, n_jobs=-1)
            importances.append(pi.importances_mean)

    assert (y_pred >= 0).all()
    return y, y_pred, pd.DataFrame(fold_rows), importances


def per_class_f1(y, y_pred) -> pd.Series:
    labels = list(CLASS_NAMES)
    return pd.Series(f1_score(y, y_pred, average=None, labels=labels, zero_division=0),
                     index=[CLASS_NAMES[k] for k in labels])


def plot_confusion(y, y_pred, path: Path) -> None:
    labels = list(CLASS_NAMES)
    cm = confusion_matrix(y, y_pred, labels=labels)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    names = [CLASS_NAMES[k] for k in labels]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "white" if cm_norm[i, j] > 0.6 else "black"
            ax.text(j, i, f"{cm_norm[i, j]:.0%}\n({cm[i, j]})", ha="center", va="center",
                    color=color, fontsize=9)
    ax.set_xticks(range(len(names)), names, rotation=20)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class (label)")
    ax.set_title("Spatial block CV: confusion matrix\n(row %, pixel counts in brackets)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_comparison(per_class: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    schemes = list(summary.index)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1, 2]})

    ax = axes[0]
    ax.bar(schemes, summary["macro_f1_mean"], yerr=summary["macro_f1_std"], capsize=5,
           color=["#bbbbbb", "#8fb3d9", "#1f77b4"])
    for i, v in enumerate(summary["macro_f1_mean"]):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Macro F1")
    ax.set_title(f"Macro F1 by split\n(mean ± std over {N_REPEATS} repeats)")

    ax = axes[1]
    width = 0.8 / len(schemes)
    x = np.arange(len(per_class.columns))
    for i, s in enumerate(schemes):
        ax.bar(x + i * width, per_class.loc[s], width, label=s,
               color=["#bbbbbb", "#8fb3d9", "#1f77b4"][i])
    ax.set_xticks(x + width * (len(schemes) - 1) / 2, per_class.columns)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("F1")
    ax.set_title("Per-class F1 by split (first repeat)")
    ax.legend(title="split", loc="lower right")
    for a in axes:
        a.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_importance(importances, path: Path) -> pd.Series:
    imp = pd.DataFrame(importances, columns=FEATURES)
    mean, std = imp.mean().sort_values(), imp.std()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(mean.index, mean, xerr=std[mean.index], capsize=3, color="#1f77b4")
    ax.set_xlabel("Drop in macro F1 when the feature is shuffled")
    ax.set_title("Permutation importance (held-out blocks)")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return mean.sort_values(ascending=False)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)
    df = load_and_sample()

    schemes = ["random", "polygon", "block"]
    summary_rows, per_class_rows = [], {}
    block_details = None

    for scheme in schemes:
        scores, accs, bare_soil = [], [], []
        for r in range(N_REPEATS):
            first = r == 0
            y, y_pred, folds, imps = cross_validate(
                scheme, df, seed=SEED + r, keep_details=(scheme == "block" and first))
            scores.append(f1_score(y, y_pred, average="macro"))
            accs.append(accuracy_score(y, y_pred))
            bare_soil.append(per_class_f1(y, y_pred)["Bare soil"])
            if first:
                per_class_rows[scheme] = per_class_f1(y, y_pred)
                if scheme == "block":
                    block_details = (y, y_pred, folds, imps)
        summary_rows.append({
            "scheme": scheme,
            "macro_f1_mean": np.mean(scores), "macro_f1_std": np.std(scores),
            "accuracy_mean": np.mean(accs),
            "bare_soil_f1_mean": np.mean(bare_soil), "bare_soil_f1_std": np.std(bare_soil),
        })
        print(f"   {scheme:8s} macro F1 {np.mean(scores):.3f} ± {np.std(scores):.3f}")

    summary = pd.DataFrame(summary_rows).set_index("scheme")
    per_class = pd.DataFrame(per_class_rows).T

    # ---- Main result: spatial block CV ----
    y, y_pred, folds, imps = block_details
    report = classification_report(y, y_pred, labels=list(CLASS_NAMES),
                                   target_names=list(CLASS_NAMES.values()), digits=3)
    print("\n=== Spatial block cross-validation (main result, first repeat) ===")
    print(report)
    print("Per fold (macro F1 and number of TEST polygons per class):")
    print(folds.round(3).to_string(index=False))
    if (folds.filter(like="test_polygons_") == 0).any().any():
        print("   NOTE: some folds have no test polygons of a class; "
              "that class's score comes from the other folds.")

    print("\n=== Comparison of split schemes ===")
    print(summary.round(3).to_string())
    print("\nPer-class F1 (first repeat):")
    print(per_class.round(3).to_string())
    gap = summary.loc["random", "macro_f1_mean"] - summary.loc["block", "macro_f1_mean"]
    print(f"\nRandom split overstates macro F1 by {gap:.3f} compared with the block split.")

    plot_confusion(y, y_pred, FIG_DIR / "confusion_matrix_block.png")
    plot_comparison(per_class, summary, FIG_DIR / "split_comparison.png")
    ranking = plot_importance(imps, FIG_DIR / "feature_importance.png")
    print("\nPermutation importance (held-out blocks), highest first:")
    print(ranking.round(3).to_string())

    summary.round(4).to_csv(RESULTS_DIR / "cv_summary.csv")
    per_class.round(4).to_csv(RESULTS_DIR / "per_class_f1.csv")
    with open(RESULTS_DIR / "block_cv_report.txt", "w") as f:
        f.write("Spatial block cross-validation (first repeat)\n\n")
        f.write(report + "\n")
        f.write(folds.round(3).to_string(index=False) + "\n\n")
        f.write("Confusion matrix (rows = true, columns = predicted): "
                f"{list(CLASS_NAMES.values())}\n")
        f.write(np.array2string(confusion_matrix(y, y_pred, labels=list(CLASS_NAMES))) + "\n")

    # ---- Final model on all labels, for the map in step 8 ----
    MODEL_PATH.parent.mkdir(exist_ok=True)
    final = RandomForestClassifier(**RF_PARAMS).fit(df[FEATURES].to_numpy(), df["class"].to_numpy())
    joblib.dump({"model": final, "features": FEATURES, "classes": CLASS_NAMES}, MODEL_PATH)

    print(f"\nSaved results/ (3 files), figures/ (3 plots), {MODEL_PATH}")


if __name__ == "__main__":
    main()
