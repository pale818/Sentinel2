"""
Stack Sentinel-2 L2A bands exported from the Copernicus Browser
(Analytical download, TIFF 32-bit float, 10 m, UTM 29N) into one GeoTIFF.

Usage (run from your project folder):
    pip install rasterio numpy
    python stack_bands.py

Expects one file per band in data/raw/, e.g. "..._B02_(Raw).tiff".
Writes data/processed/stack.tif with 6 bands in the order of BANDS.
"""
from pathlib import Path

import numpy as np
import rasterio

RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/processed/stack.tif")
BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]


def find_band_file(band: str) -> Path:
    """Return the single TIFF in RAW_DIR whose name contains the band code."""
    matches = [
        p for p in RAW_DIR.glob(f"*{band}*")
        if p.suffix.lower() in (".tif", ".tiff")
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one file for {band} in {RAW_DIR}, found: {matches}"
        )
    return matches[0]


def main() -> None:
    arrays = []
    ref = None  # grid of the first band; all others must match it

    for band in BANDS:
        path = find_band_file(band)
        with rasterio.open(path) as src:
            if src.count != 1:
                print(f"  warning: {path.name} has {src.count} bands, using band 1")
            data = src.read(1).astype("float32")

            if ref is None:
                ref = {
                    "crs": src.crs,
                    "transform": src.transform,
                    "shape": data.shape,
                    "profile": src.profile.copy(),
                }
            else:
                # Same grid is what makes stacking valid: same CRS,
                # same pixel size/origin, same number of rows and columns.
                if src.crs != ref["crs"]:
                    raise ValueError(f"{band}: CRS {src.crs} != {ref['crs']}")
                if src.transform != ref["transform"]:
                    raise ValueError(f"{band}: pixel grid differs from first band")
                if data.shape != ref["shape"]:
                    raise ValueError(f"{band}: shape {data.shape} != {ref['shape']}")

        arrays.append(data)
        print(
            f"{band}: {path.name}\n"
            f"   min={np.nanmin(data):.4f}  mean={np.nanmean(data):.4f}  "
            f"max={np.nanmax(data):.4f}  NaNs={int(np.isnan(data).sum())}"
        )
        # Reflectance sanity check. The max is a poor test: a few pixels of
        # sun glint or bright roofs legitimately exceed 1.0 in L2A. Use the
        # mean (scale) and the 99.9th percentile (bulk of the data) instead.
        p999 = np.nanpercentile(data, 99.9)
        n_above_1 = int((data > 1.0).sum())
        print(f"   p99.9={p999:.4f}  pixels > 1.0: {n_above_1}")
        if np.nanmean(data) > 1.5:
            print("   ERROR?: mean > 1.5, this looks like scaled digital numbers, "
                  "not reflectance (0-1). Check export settings.")
        elif p999 > 1.0:
            print("   warning: p99.9 > 1.0, unusually many very bright pixels. "
                  "Inspect the image.")

    stack = np.stack(arrays)  # shape: (bands, rows, cols)
    print(f"\nStack shape (bands, rows, cols): {stack.shape}")
    print(f"CRS: {ref['crs']}  |  pixel size: {ref['transform'].a} m")

    profile = ref["profile"]
    profile.update(count=len(BANDS), dtype="float32", compress="deflate")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(OUT_PATH, "w", **profile) as dst:
        dst.write(stack)
        for i, band in enumerate(BANDS, start=1):
            dst.set_band_description(i, band)

    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
