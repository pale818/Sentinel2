# Land cover classification of Lisbon with Sentinel-2

> **Status: work in progress.** Data acquisition, stacking and data checks are done.
> Labelling, training and evaluation are next. Sections marked *(planned)* describe
> intended methods, not results.

## Goal

Classify every pixel of a Sentinel-2 image of the Lisbon area into four land cover
classes with a Random Forest, and evaluate the result honestly:

| Class | Value |
|---|---|
| Water | 0 |
| Vegetation | 1 |
| Urban | 2 |
| Bare soil | 3 |

The emphasis is on sound methodology rather than a high accuracy number. Labels
are drawn by hand, the train/test split respects spatial structure, and the
limitations are reported.

## Data

| Item | Value |
|---|---|
| Satellite / product | Sentinel-2 Level-2A (surface reflectance) |
| Acquisition date | 29 July 2026 (~5% cloud cover) |
| Source | Copernicus Browser, *Analytical* download |
| Bands | B02 (blue), B03 (green), B04 (red), B08 (near-infrared), B11, B12 (shortwave infrared) |
| Format | GeoTIFF, 32-bit float |
| Resolution | ~10 m (B11/B12 resampled from 20 m during export) |
| Projection | UTM zone 29N (EPSG:32629) |
| Size | 2211 × 1211 pixels, ~268 km² |
| Area | Monsanto Forest Park (west) → central Lisbon, Tagus estuary, Almada, Barreiro → Montijo / Alcochete (east) |

A true colour image was exported as well. It is used only as a visual reference
for labelling in QGIS, never as a model input.

Resampling, cropping and conversion to reflectance were done by the Copernicus
export service, not by this project. What this project did was verify the result
(see "Data checks").

<!-- TODO: add the exact Sentinel-2 product name (Copernicus Browser, Search tab, 29 July 2026) -->

### Reproducing the download

The imagery is not in this repository (see `.gitignore`). To reproduce it:
1. Copernicus Browser → Sentinel-2 L2A, 29 July 2026, area as above.
2. Download → Analytical: TIFF 32-bit float, custom resolution 10 m,
   UTM 29N, *Clip extra bands* on, *dataMask* off.
3. Select the raw bands B02, B03, B04, B08, B11, B12 (+ True color for labelling).
4. Put the TIFF files directly in `data/raw/`.

## Project structure

```
SentinelProject/
├── data/
│   ├── raw/            # band TIFFs from the export (not committed)
│   └── processed/      # stack.tif (not committed, recreated by stack_bands.py)
├── stack_bands.py      # stacks the 6 bands into one GeoTIFF
├── check_stack.py      # sanity checks on the stack
├── requirements.txt
└── README.md
```

## Setup and usage

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python stack_bands.py   # -> data/processed/stack.tif
python check_stack.py   # prints the data checks below
```

## Pipeline

### 1. Stacking (`stack_bands.py`) — done

- Finds one TIFF per band in `data/raw/` by band name.
- Verifies that all bands share the same CRS, pixel grid and shape. Stacking is
  only valid if every pixel position refers to the same location in every band.
- Writes `data/processed/stack.tif`: 6 bands in the order B02, B03, B04, B08,
  B11, B12, with band names stored in the file.
- Prints per-band statistics and warns if the values do not look like
  reflectance (mean > 1.5 suggests scaled digital numbers; 99.9th percentile
  > 1.0 suggests unusually many saturated pixels).

### 2. Data checks (`check_stack.py`) — done

**Value range.** Values are surface reflectance (0–1).

| Band | Median | 99.9th percentile | Pixels > 1.0 |
|---|---|---|---|
| B02 | 0.050 | 0.493 | 18 |
| B03 | 0.070 | 0.511 | 21 |
| B04 | 0.065 | 0.521 | 22 |
| B08 | 0.154 | 0.557 | 32 |
| B11 | 0.144 | 0.556 | 90 |
| B12 | 0.083 | 0.512 | 70 |

121 pixels (0.005%) exceed 1.0 in at least one band. These are specular
reflections from water and bright surfaces, which are expected in L2A data.
They were left uncorrected.

The maximum was deliberately not used as the test: a handful of glinting
pixels push it above 1.5, so a max-based check raises a false alarm on almost
every scene. Percentiles describe the bulk of the data.

**No-data.** With *dataMask* off, no-data is exported as 0, not NaN.
4,389 pixels (0.16%) are 0 in all bands. They form a 1–2 pixel border at the
image edges. They will be masked out of training and prediction.

**Spectral sanity check.** Four locations of known land cover were sampled.
The points were chosen and checked visually on the true colour image.

| Location | B03 | B04 | B08 | B11 | Expected behaviour |
|---|---|---|---|---|---|
| Monsanto forest 1 | 0.044 | 0.034 | 0.262 | 0.145 | B08 ≫ B04 (vegetation) |
| Monsanto forest 2 | 0.056 | 0.041 | 0.312 | 0.135 | B08 ≫ B04 (vegetation) |
| Tagus estuary 1 | 0.047 | 0.027 | 0.013 | 0.007 | B08, B11 ≈ 0 (water) |
| Tagus estuary 2 | 0.050 | 0.030 | 0.014 | 0.007 | B08, B11 ≈ 0 (water) |

Vegetation reflects strongly in the near-infrared and absorbs red light (NDVI ≈ 0.77
at both forest points). Water absorbs near-infrared and shortwave infrared almost
completely. Both patterns appear where expected, which confirms the
reflectance scale, the band order and the georeferencing.

**Lesson learned.** The first version of this check used coordinates picked
from memory. The "water" point actually fell on fields on the Montijo peninsula,
and the "forest" point fell on a clearing inside Monsanto. Both produced
implausible spectra, which is how the error was caught. Sample locations
(and later, label polygons) are now always checked visually on the image.

**First look at the difficult classes.** Single-pixel samples suggest that
urban areas (e.g. Baixa: B08 0.20, B11 0.30) and dry bare ground have very
similar spectra in every band. Urban/bare soil confusion is therefore
expected. This will be quantified properly with class mean spectra
from the labelled data.

### 3. Labelling in QGIS *(planned)*

Hand-drawn polygons in a GeoPackage layer with an integer `class` field,
roughly 15–30 small polygons per class spread across the image. Polygons are
drawn inside homogeneous areas, away from class boundaries and image edges.

Labels are deliberately **not** created by thresholding spectral indices.
The model would then just relearn those thresholds, and the evaluation would be circular.

### 4. Features and training *(planned)*

- Features: the 6 bands plus NDVI (B08, B04), NDWI (B03, B08) and NDBI (B11, B08).
  The indices are model inputs, not labels.
- Random Forest (~200 trees, `class_weight="balanced"`) because water
  dominates the scene.
- Train/test split **by polygon**, not by pixel. Neighbouring pixels are
  nearly identical (spatial autocorrelation). A random pixel split puts
  near-duplicates into both sets and inflates accuracy.

### 5. Evaluation *(planned)*

Confusion matrix and per-class precision, recall and F1, not only overall
accuracy. A comparison of random-pixel vs polygon-grouped splitting will
measure how much a naive split overstates performance.

## Known limitations

- **Single date.** One summer image. Seasonal differences, e.g. dry
  vegetation vs bare soil, are not captured.
- **Mixed pixels.** At 10 m, many pixels contain more than one land cover
  (street trees in urban blocks, field edges).
- **Urban vs bare soil.** These are spectrally similar in this scene (see above).
- **Limited labels.** Labels are hand-drawn by one person on a single image,
  with no field validation.
- **Ambiguous land cover.** Dry fields in July sit between "vegetation" and
  "bare soil". Ambiguous areas are left unlabelled rather than guessed.

## Attribution

Contains modified Copernicus Sentinel data (2026), processed by Sentinel Hub
through the Copernicus Browser.
