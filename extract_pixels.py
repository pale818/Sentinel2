"""
Step 5: extract the labelled pixels from the Sentinel-2 stack.

For every pixel inside a label polygon this script stores:
    polygon ID, class, spatial block ID, the 6 band values, NDVI, NDWI, NDBI

It also
    - masks the no-data border (pixels that are 0 in all bands)
    - assigns each POLYGON to one 2 x 2 km block (used for the spatial
      cross-validation in step 6; a polygon is never split across blocks)
    - prints pixel / polygon / block counts per class
    - prints an outlier report: polygons whose spectrum looks more like
      another class, or is far from the rest of its own class
    - plots the mean spectrum per class and the index distributions

Usage (from the project folder):
    pip install -r requirements.txt
    python extract_pixels.py

Inputs:  data/processed/stack.tif, data/labels.gpkg (layer "labels")
Outputs: data/processed/pixels.csv           one row per labelled pixel
         data/processed/polygon_summary.csv  one row per polygon
         figures/class_spectra.png
         figures/index_boxplots.png

All pixels are kept here. Sub-sampling (e.g. fewer pixels per water polygon)
is a training choice and happens in step 6, so it can be changed without
re-extracting.
"""
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")  # write figures to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize

STACK_PATH = Path("data/processed/stack.tif")
LABELS_PATH = Path("data/labels.gpkg")
LABELS_LAYER = "labels"
OUT_PIXELS = Path("data/processed/pixels.csv")
OUT_POLYGONS = Path("data/processed/polygon_summary.csv")
FIG_DIR = Path("figures")

BLOCK_SIZE_M = 2000   # side length of the spatial blocks, in metres
MIN_PIXELS = 25       # polygons with fewer pixels are reported as "small"

BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]
# Approximate central wavelengths of the Sentinel-2 bands (nm), for plotting
WAVELENGTH_NM = {"B02": 490, "B03": 560, "B04": 665, "B08": 842, "B11": 1610, "B12": 2190}
INDICES = ["NDVI", "NDWI", "NDBI"]
FEATURES = BANDS + INDICES

CLASS_NAMES = {0: "Water", 1: "Vegetation", 2: "Built-up", 3: "Bare soil"}
CLASS_COLORS = {0: "#1f77b4", 1: "#2ca02c", 2: "#7f7f7f", 3: "#c49a6c"}


def normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(a - b) / (a + b), with NaN where a + b == 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        nd = (a - b) / (a + b)
    nd[~np.isfinite(nd)] = np.nan
    return nd


def load_inputs():
    with rasterio.open(STACK_PATH) as src:
        stack = src.read().astype("float32")  # (6, rows, cols)
        transform, crs = src.transform, src.crs
        names = list(src.descriptions)
    if names != BANDS:
        raise ValueError(f"Unexpected band order in {STACK_PATH}: {names}")

    labels = gpd.read_file(LABELS_PATH, layer=LABELS_LAYER, fid_as_index=True)
    labels.index.name = "polygon_id"
    if labels.crs != crs:
        raise ValueError(f"CRS mismatch: labels {labels.crs}, image {crs}")
    if labels["class"].isna().any():
        raise ValueError(f"Polygons without a class: {list(labels.index[labels['class'].isna()])}")
    labels["class"] = labels["class"].astype(int)
    unknown = set(labels["class"]) - set(CLASS_NAMES)
    if unknown:
        raise ValueError(f"Unknown class values: {unknown}")
    return stack, transform, labels


def check_overlaps(labels: gpd.GeoDataFrame) -> None:
    """Overlapping polygons would put the same pixels in the data twice."""
    pairs = []
    sindex = labels.sindex
    for pid, geom in labels.geometry.items():
        for j in sindex.query(geom, predicate="intersects"):
            other = labels.index[j]
            if other > pid and geom.intersection(labels.geometry[other]).area > 0:
                pairs.append((pid, other))
    if pairs:
        raise ValueError(f"Overlapping polygons (fix in QGIS first): {pairs}")


def assign_blocks(labels: gpd.GeoDataFrame) -> pd.Series:
    """Block ID from the polygon centroid, so a polygon is never split."""
    c = labels.geometry.centroid
    bx = (c.x // BLOCK_SIZE_M).astype(int)
    by = (c.y // BLOCK_SIZE_M).astype(int)
    return bx.astype(str) + "_" + by.astype(str)


def extract(stack, transform, labels) -> pd.DataFrame:
    rows_, cols_ = stack.shape[1:]
    # Burn the polygon ID into a raster on the same grid as the image.
    # A pixel belongs to a polygon if its centre lies inside the polygon.
    poly_raster = rasterize(
        ((geom, int(pid)) for pid, geom in labels.geometry.items()),
        out_shape=(rows_, cols_),
        transform=transform,
        fill=0,
        dtype="int32",
    )

    nodata = (stack == 0).all(axis=0)
    in_poly = poly_raster > 0
    n_masked = int((in_poly & nodata).sum())
    if n_masked:
        print(f"Masked {n_masked} labelled pixels that are no-data (0 in all bands)")
    keep = in_poly & ~nodata

    r, c = np.nonzero(keep)
    xs, ys = rasterio.transform.xy(transform, r, c)  # pixel centres

    df = pd.DataFrame({
        "row": r, "col": c,
        "x": np.asarray(xs), "y": np.asarray(ys),
        "polygon_id": poly_raster[r, c],
    })
    for i, band in enumerate(BANDS):
        df[band] = stack[i, r, c]

    df["NDVI"] = normalized_difference(df["B08"].to_numpy(), df["B04"].to_numpy())
    df["NDWI"] = normalized_difference(df["B03"].to_numpy(), df["B08"].to_numpy())
    df["NDBI"] = normalized_difference(df["B11"].to_numpy(), df["B08"].to_numpy())

    df = df.join(labels[["class", "block_id"]], on="polygon_id")
    df["class_name"] = df["class"].map(CLASS_NAMES)
    return df


def print_counts(df: pd.DataFrame, labels: gpd.GeoDataFrame) -> None:
    counts = labels.groupby("class").agg(polygons=("block_id", "size"), blocks=("block_id", "nunique"))
    px = df.groupby("class").agg(pixels=("polygon_id", "size"))
    per_poly = df.groupby("polygon_id").size()
    med = per_poly.groupby(labels.loc[per_poly.index, "class"]).median().rename("median_px_per_polygon")
    table = counts.join(px).join(med)
    table.index = table.index.map(CLASS_NAMES)
    print("\nCounts per class")
    print(table.to_string())
    print(f"Total: {len(labels)} polygons, {len(df)} pixels, "
          f"{labels['block_id'].nunique()} blocks of {BLOCK_SIZE_M / 1000:g} km")

    missing = sorted(set(labels.index) - set(per_poly.index))
    if missing:
        print(f"\nWARNING: polygons with 0 pixels (outside the image or too thin): {missing}")
    small = per_poly[per_poly < MIN_PIXELS]
    if len(small):
        print(f"\nPolygons with fewer than {MIN_PIXELS} pixels (OK, but small):")
        by_class = small.groupby(labels.loc[small.index, "class"])
        for k, s in by_class:
            print(f"   {CLASS_NAMES[k]}: {len(s)} polygons, ids {list(s.index)}")


def polygon_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean features per polygon + outlier flags."""
    poly = df.groupby("polygon_id").agg(
        **{"class": ("class", "first"), "block_id": ("block_id", "first"), "n_pixels": ("B02", "size")},
        **{f: (f, "mean") for f in FEATURES},
        NDVI_min=("NDVI", "min"), NDVI_max=("NDVI", "max"),
    )

    # Standardise polygon means so every feature counts equally in distances
    z = (poly[FEATURES] - poly[FEATURES].mean()) / poly[FEATURES].std()
    centroids = z.groupby(poly["class"]).median()  # median is robust to outliers

    dist = pd.DataFrame({k: np.sqrt(((z - centroids.loc[k]) ** 2).sum(axis=1)) for k in centroids.index})
    own = dist.to_numpy()[np.arange(len(poly)), poly["class"].map(list(centroids.index).index)]
    other = dist.where(dist.columns.to_numpy()[None, :] != poly["class"].to_numpy()[:, None])
    poly["dist_own_class"] = own
    poly["nearest_other_class"] = other.idxmin(axis=1).map(CLASS_NAMES)
    poly["dist_nearest_other"] = other.min(axis=1)

    # Flag 1: spectrum closer to another class than to its own
    poly["closer_to_other_class"] = poly["dist_nearest_other"] < poly["dist_own_class"]
    # Flag 2: far from its own class (above Q3 + 1.5 * IQR within the class)
    def far(s):
        q1, q3 = s.quantile([0.25, 0.75])
        return s > q3 + 1.5 * (q3 - q1)
    poly["far_from_own_class"] = poly.groupby("class")["dist_own_class"].transform(far)
    poly["class_name"] = poly["class"].map(CLASS_NAMES)
    return poly


def print_outliers(poly: pd.DataFrame) -> None:
    flagged = poly[poly["closer_to_other_class"] | poly["far_from_own_class"]]
    print("\nOutlier report (check these polygons in QGIS)")
    if flagged.empty:
        print("   no polygons flagged")
        return
    cols = ["class_name", "n_pixels", "NDVI", "NDWI", "NDBI", "dist_own_class",
            "nearest_other_class", "dist_nearest_other", "closer_to_other_class", "far_from_own_class"]
    print(flagged[cols].round(2).to_string())
    print("\n   closer_to_other_class: the polygon's mean spectrum is nearer the typical")
    print("   spectrum of another class than its own -> likely confused by the model.")
    print("   far_from_own_class: unusual for its own class -> mislabel, or real variety.")
    print("   A flag is a prompt to look again, not proof of an error.")


def plot_spectra(df: pd.DataFrame) -> Path:
    wl = [WAVELENGTH_NM[b] for b in BANDS]
    fig, ax = plt.subplots(figsize=(8, 5))
    for k, name in CLASS_NAMES.items():
        sub = df[df["class"] == k]
        if sub.empty:
            continue
        # Mean of polygon means: every polygon counts equally,
        # so large polygons do not dominate the curve
        poly_means = sub.groupby("polygon_id")[BANDS].mean()
        mean, sd = poly_means.mean(), poly_means.std()
        ax.plot(wl, mean, "o-", color=CLASS_COLORS[k],
                label=f"{name} ({len(poly_means)} polygons)")
        ax.fill_between(wl, mean - sd, mean + sd, color=CLASS_COLORS[k], alpha=0.15)
    for b, w in zip(BANDS, wl):
        ax.annotate(b, (w, 0), textcoords="offset points", xytext=(0, -28),
                    ha="center", fontsize=8, color="0.4", annotation_clip=False)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Surface reflectance")
    ax.set_title("Mean spectrum per class (shading: ±1 std between polygons)")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = FIG_DIR / "class_spectra.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_indices(poly: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
    classes = [k for k in CLASS_NAMES if k in set(poly["class"])]
    for ax, idx in zip(axes, INDICES):
        data = [poly.loc[poly["class"] == k, idx] for k in classes]
        bp = ax.boxplot(data, patch_artist=True)
        ax.set_xticks(range(1, len(classes) + 1), [CLASS_NAMES[k] for k in classes], rotation=20)
        for patch, k in zip(bp["boxes"], classes):
            patch.set_facecolor(CLASS_COLORS[k])
            patch.set_alpha(0.6)
        ax.set_title(f"{idx} (polygon means)")
        ax.axhline(0, color="0.6", lw=0.8)
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = FIG_DIR / "index_boxplots.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    stack, transform, labels = load_inputs()
    check_overlaps(labels)
    labels["block_id"] = assign_blocks(labels)

    df = extract(stack, transform, labels)
    print_counts(df, labels)

    poly = polygon_summary(df)
    print_outliers(poly)

    OUT_PIXELS.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PIXELS, index=False)
    poly.to_csv(OUT_POLYGONS)

    FIG_DIR.mkdir(exist_ok=True)
    f1 = plot_spectra(df)
    f2 = plot_indices(poly)

    print("\nClass means (polygon-weighted)")
    means = poly.groupby("class_name")[FEATURES].mean().round(3)
    print(means.to_string())

    print(f"\nSaved {OUT_PIXELS} ({len(df)} rows)")
    print(f"Saved {OUT_POLYGONS} ({len(poly)} rows)")
    print(f"Saved {f1}")
    print(f"Saved {f2}")


if __name__ == "__main__":
    main()
