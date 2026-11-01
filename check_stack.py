"""
Sanity checks on data/processed/stack.tif (run after stack_bands.py).

1. Per-band distribution (percentiles) and counts of pixels > 1.0 and == 0
2. No-data detection: pixels that are 0 in every band (dataMask was off,
   so no-data is exported as 0, not NaN)
3. Spectral check at two known places (forest, water)

Usage:
    python check_stack.py
"""
import numpy as np
import rasterio
from rasterio.transform import rowcol
from rasterio.warp import transform

STACK = "data/processed/stack.tif"

# (lon, lat) in WGS84, picked from the true colour image and verified:
# forest NDVI ~0.74-0.80, water B08/B11 < 0.02.
POINTS = {
    "Monsanto forest 1": (-9.198, 38.725),
    "Monsanto forest 2": (-9.191, 38.736),
    "Tagus estuary water 1": (-9.106, 38.712),
    "Tagus estuary water 2": (-9.083, 38.725),
}

with rasterio.open(STACK) as src:
    data = src.read()  # shape (6, rows, cols)
    names = src.descriptions
    crs, tr = src.crs, src.transform

# 1. Distribution, not just min/max
print("Per-band distribution")
for name, band in zip(names, data):
    p1, med, p99, p999 = np.percentile(band, [1, 50, 99, 99.9])
    print(f"  {name}: p1={p1:.3f} median={med:.3f} p99={p99:.3f} p99.9={p999:.3f} | "
          f">1.0: {(band > 1).sum()} px | ==0: {(band == 0).sum()} px")

# 2. No-data: zero in every band
allzero = (data == 0).all(axis=0)
print(f"\nPixels zero in ALL bands (likely no-data): {allzero.sum()} "
      f"of {allzero.size} ({100 * allzero.mean():.3f}%)")
anyabove1 = (data > 1).any(axis=0)
print(f"Pixels > 1.0 in ANY band: {anyabove1.sum()} "
      f"({100 * anyabove1.mean():.3f}%)")

# 3. Physical sanity check at known places
for label, (lon, lat) in POINTS.items():
    xs, ys = transform("EPSG:4326", crs, [lon], [lat])
    r, c = rowcol(tr, xs[0], ys[0])
    if 0 <= r < data.shape[1] and 0 <= c < data.shape[2]:
        print(f"\n{label} (row {r}, col {c}):")
        for n, v in zip(names, data[:, r, c]):
            print(f"   {n}: {v:.3f}")
    else:
        print(f"\n{label}: outside the image, check the coordinates")
