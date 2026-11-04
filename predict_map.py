"""
Step 8: classify every pixel of the image with the trained Random Forest.

Usage (from the project folder, after train_evaluate.py):
    python predict_map.py

Inputs:  data/processed/stack.tif, models/random_forest.joblib
Outputs: data/processed/landcover_map.tif   class per pixel (0-3, 255 = no-data),
                                            with a colour table for QGIS
         data/processed/confidence.tif      model confidence per pixel (0-100 %)
         figures/landcover_map.png          true colour next to the map
         figures/confidence_map.png         where the model is unsure

"Confidence" is the share of the 200 trees that voted for the chosen class.
It is not a calibrated probability, but low values show where the model is
torn between classes. Those are the places to inspect in QGIS.
"""
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

STACK_PATH = Path("data/processed/stack.tif")
MODEL_PATH = Path("models/random_forest.joblib")
OUT_MAP = Path("data/processed/landcover_map.tif")
OUT_CONF = Path("data/processed/confidence.tif")
FIG_DIR = Path("figures")

NODATA = 255
CHUNK_ROWS = 200          # predict in chunks to keep memory use low
LOW_CONFIDENCE = 0.6      # below this, a pixel counts as "uncertain"

BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]
CLASS_COLORS = {0: "#1f77b4", 1: "#2ca02c", 2: "#d62728", 3: "#e6c27a"}


def features(block: np.ndarray, feature_names: list[str]) -> np.ndarray:
    """block: (6, rows, cols) -> (rows*cols, n_features), same order as training."""
    b = {name: block[i].ravel().astype("float32") for i, name in enumerate(BANDS)}
    with np.errstate(divide="ignore", invalid="ignore"):
        b["NDVI"] = (b["B08"] - b["B04"]) / (b["B08"] + b["B04"])
        b["NDWI"] = (b["B03"] - b["B08"]) / (b["B03"] + b["B08"])
        b["NDBI"] = (b["B11"] - b["B08"]) / (b["B11"] + b["B08"])
    X = np.column_stack([b[f] for f in feature_names])
    X[~np.isfinite(X)] = 0
    return X


def hex_to_rgb255(h: str) -> tuple:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def main() -> None:
    bundle = joblib.load(MODEL_PATH)
    model, feature_names, class_names = bundle["model"], bundle["features"], bundle["classes"]

    with rasterio.open(STACK_PATH) as src:
        if list(src.descriptions) != BANDS:
            raise ValueError(f"Unexpected band order: {src.descriptions}")
        stack = src.read().astype("float32")
        profile = src.profile.copy()

    _, rows, cols = stack.shape
    nodata = (stack == 0).all(axis=0)
    class_map = np.full((rows, cols), NODATA, dtype="uint8")
    conf_map = np.zeros((rows, cols), dtype="float32")

    print(f"Classifying {rows * cols:,} pixels ...")
    for r0 in range(0, rows, CHUNK_ROWS):
        r1 = min(r0 + CHUNK_ROWS, rows)
        X = features(stack[:, r0:r1, :], feature_names)
        proba = model.predict_proba(X)
        pred = model.classes_[proba.argmax(axis=1)]
        class_map[r0:r1] = pred.reshape(r1 - r0, cols)
        conf_map[r0:r1] = proba.max(axis=1).reshape(r1 - r0, cols)
    class_map[nodata] = NODATA
    conf_map[nodata] = 0

    # ---- GeoTIFFs (open these in QGIS) ----
    OUT_MAP.parent.mkdir(parents=True, exist_ok=True)
    p = profile.copy()
    p.update(count=1, dtype="uint8", nodata=NODATA, compress="deflate")
    with rasterio.open(OUT_MAP, "w", **p) as dst:
        dst.write(class_map, 1)
        dst.write_colormap(1, {k: hex_to_rgb255(c) + (255,) for k, c in CLASS_COLORS.items()})
        dst.set_band_description(1, "class: " + ", ".join(f"{k}={v}" for k, v in class_names.items()))

    p.update(dtype="uint8", nodata=NODATA)
    with rasterio.open(OUT_CONF, "w", **p) as dst:
        conf_pct = np.round(conf_map * 100).astype("uint8")
        conf_pct[nodata] = NODATA
        dst.write(conf_pct, 1)
        dst.set_band_description(1, "confidence (% of trees voting for the predicted class)")

    # ---- Area per class ----
    pixel_km2 = abs(profile["transform"].a * profile["transform"].e) / 1e6
    valid = ~nodata
    print("\nPredicted area per class")
    print(f"   {'class':<11}{'km²':>8}{'share':>8}{'uncertain*':>12}")
    for k, name in class_names.items():
        m = (class_map == k) & valid
        unsure = (conf_map[m] < LOW_CONFIDENCE).mean() if m.any() else 0
        print(f"   {name:<11}{m.sum() * pixel_km2:8.1f}{m.sum() / valid.sum():8.1%}{unsure:12.1%}")
    print(f"   * share of that class's pixels with confidence below {LOW_CONFIDENCE:.0%}")
    print(f"   Overall: {(conf_map[valid] < LOW_CONFIDENCE).mean():.1%} of pixels are uncertain")

    # ---- Figures ----
    FIG_DIR.mkdir(exist_ok=True)
    rgb = np.clip(np.dstack([stack[2], stack[1], stack[0]]) / 0.3, 0, 1)
    rgb[nodata] = 1
    cmap = ListedColormap([CLASS_COLORS[k] for k in sorted(class_names)])
    shown = np.ma.masked_equal(class_map, NODATA)

    fig, axes = plt.subplots(2, 1, figsize=(12, 13))
    axes[0].imshow(rgb)
    axes[0].set_title("Sentinel-2 true colour, 29 July 2026")
    axes[1].imshow(shown, cmap=cmap, vmin=-0.5, vmax=len(class_names) - 0.5, interpolation="nearest")
    axes[1].set_title("Random Forest land cover map")
    axes[1].legend(handles=[Patch(color=CLASS_COLORS[k], label=v) for k, v in class_names.items()],
                   loc="lower left", framealpha=0.9)
    for a in axes:
        a.set_axis_off()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "landcover_map.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6.5))
    im = ax.imshow(np.ma.masked_where(nodata, conf_map), cmap="magma", vmin=0.25, vmax=1)
    ax.set_title("Model confidence (share of trees voting for the predicted class)")
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, fraction=0.03, label="confidence")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confidence_map.png", dpi=110)
    plt.close(fig)

    print(f"\nSaved {OUT_MAP}, {OUT_CONF}")
    print(f"Saved {FIG_DIR / 'landcover_map.png'}, {FIG_DIR / 'confidence_map.png'}")


if __name__ == "__main__":
    main()
