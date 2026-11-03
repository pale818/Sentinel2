# Land cover classification of Lisbon with Sentinel-2

> **Status: work in progress.** Data acquisition, stacking and data checks are done.
> Labelling is done. Sections marked *(planned)* describe intended methods,
> not results. See "Plan" for the full sequence.

## Goal

Classify every pixel of a Sentinel-2 image of the Lisbon area into four land cover
classes with a Random Forest, and evaluate the result honestly:

| Value | Class | Includes | Excludes |
|---|---|---|---|
| 0 | Water | river, estuary, harbour basins | tidal mudflats, salt marsh, salt pans and aquaculture ponds |
| 1 | Vegetation | forest, parks, gardens, green fields | dry yellow fields (ambiguous) |
| 2 | Built-up | buildings, roads, paved areas, ports, airfields | street trees, gardens |
| 3 | Bare soil | bare earth, empty plots, construction sites, beach sand | tidal mudflats |

**Not labelled:** tidal mudflats, salt marsh, shadows, and anything ambiguous.
These areas fit none of the four classes cleanly. The model will still assign
them a class in the final map, and that behaviour is inspected and reported
(see "Known limitations").

The emphasis is on sound methodology rather than a high accuracy number. Labels
are drawn by hand, the train/test split respects spatial structure, and the
limitations are reported.

## Plan

| # | Step | Status |
|---|---|---|
| 1 | Download Sentinel-2 L2A image, stack the bands | done |
| 2 | Data checks (value range, no-data, spectra at known locations) | done |
| 3 | Label polygons in QGIS (spread over the whole image) | done (98 polygons) |
| 4 | Review polygons visually and with a per-polygon spectral check | done |
| 5 | Extract labelled pixels, mask no-data, compute indices, check counts, plot class spectra | planned |
| 6 | Train Random Forest, evaluate with **spatial block cross-validation** | planned |
| 7 | Compare with a random pixel split (how much does a naive split overstate accuracy?) | planned |
| 8 | Predict the full map and inspect it visually (shores, mudflats, built-up vs bare soil) | planned |
| 9 | Final README: results, figures, limitations | planned |

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
│   ├── processed/      # stack.tif (not committed, recreated by stack_bands.py)
│   └── labels.gpkg     # hand-drawn training polygons (committed)
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

### 3. Labelling in QGIS — done

Hand-drawn polygons in `data/labels.gpkg` (layer `labels`, polygon geometry,
EPSG:32629) with an integer `class` field. The automatic `fid` serves as the
polygon ID.

- 15–25 small polygons per class, **spread across the whole image**, so that
  every class appears in many different areas. This is required for the
  spatial cross-validation in step 6.
- Polygons are drawn inside homogeneous areas, away from class boundaries and
  the image edge. Homogeneous means homogeneous **at 10 m**, the Sentinel-2
  pixel size.
- Polygons are labelled by what the pixels show, not by what the area is called
  (a tree in a city district is vegetation).
- Ambiguous areas are skipped, not guessed.
- Labels are not marked as train or test. The split is done in code (step 6).

Labels are deliberately **not** created by thresholding spectral indices.
The model would then just relearn those thresholds, and the evaluation would be circular.

### 4. Polygon review — done

Every polygon was checked in two ways:

1. **Visually**, on the Sentinel-2 image in true colour (B04/B03/B02) and false
   colour (B08/B04/B03). False colour makes vegetation bright red and water
   nearly black, which separates them far better than true colour.
2. **Spectrally**, by computing the pixel values inside each polygon (mean,
   minimum, 10th percentile of NDVI / NDWI) and comparing them with the rest of
   its class.

Polygons were removed when the visual and spectral evidence disagreed with the
assigned class. Examples:
- dark-green salt pans / aquaculture ponds labelled as vegetation
  (NDVI −0.2 to −0.3: near-infrared is absorbed, so this is water, not plants)
- a clearing inside Monsanto labelled as bare soil (NDVI ~0.16, dark: dry grass)
- accidental stacked duplicates created with the copy tool

A reference orthophoto (DGT OrtoSat 2023) was considered as a third source but
could not be loaded reliably as a WMS layer, so it was **not** used.

**Final labels:** 98 polygons.

| Class | Polygons |
|---|---|
| Water | 34 |
| Vegetation | 22 |
| Built-up | 31 |
| Bare soil | 11 |

Clear bare soil is rare in this summer scene, hence the smaller count. Clusters
were thinned (e.g. at most ~8 vegetation polygons in Monsanto, fewer built-up
polygons in central Lisbon), and variety was added on purpose (turbid and clear
water; forest, parks and irrigated fields; residential, industrial and port areas).

**Early finding.** Mean polygon spectra show that bare soil and built-up have
almost identical NDVI (0.09 vs 0.09) and NDBI (0.11 vs 0.13). Bare soil is
mainly brighter in every band. Several bare-soil polygons fall within the
spectral range of the built-up class, so confusion between the two is expected.

### 5. Pixel extraction and features *(planned)*

- Rasterize the polygons onto the image grid (class and polygon ID per pixel).
- Mask the no-data border (pixels that are 0 in all bands).
- Features: the 6 bands plus NDVI (B08, B04), NDWI (B03, B08) and NDBI (B11, B08).
  The indices are model inputs, not labels.
- Count pixels per class and per polygon. Large polygons are capped to a
  maximum number of pixels, so that no single area dominates training.
- Plot the mean spectrum of each class, to see before training how separable
  the classes are, especially built-up vs bare soil.

### 6. Training and evaluation *(planned)*

- Random Forest (~200 trees, `class_weight="balanced"`) because water
  dominates the scene.
- **Spatial block cross-validation.** The image is divided into blocks of about
  2 × 2 km, and each polygon is assigned to the block it falls in.
  `StratifiedGroupKFold` with the block as the group means every fold tests on
  areas the model has never seen, and every polygon is tested exactly once.

  Why blocks and not single pixels or single polygons: neighbouring pixels are
  nearly identical (spatial autocorrelation). Two polygons 100 m apart are
  still near-duplicates. Grouping by block keeps whole neighbourhoods on one
  side of the split. The split is made by code, not chosen by hand, so it is
  not biased towards easy test areas.
- Metrics: confusion matrix and per-class precision, recall and F1, not only
  overall accuracy.

### 7. Random vs spatial split *(planned)*

The same model is also evaluated with a random pixel split. The difference
between the two scores shows how much a naive split overstates accuracy.

### 8. Full map and visual inspection *(planned)*

The model predicts every pixel of the image (no-data masked). The map is
inspected visually, with attention to shorelines, tidal mudflats, the edges of
Monsanto, and areas where built-up and bare soil meet.

## Known limitations

- **Test accuracy measures easy pixels.** Labels are drawn only in clean,
  homogeneous areas. The cross-validation score therefore describes
  performance on pure pixels and overestimates the accuracy of the full map,
  which also contains mixed pixels at boundaries.
- **Single date.** One summer image. Seasonal differences, e.g. dry
  vegetation vs bare soil, are not captured.
- **Mixed pixels.** At 10 m, many pixels contain more than one land cover
  (street trees in built-up blocks, field edges, shorelines).
- **Built-up vs bare soil.** These are spectrally similar in this scene (see above).
- **Unlabelled surface types.** Tidal mudflats and salt marsh belong to none of
  the four classes, but the model must still assign them one.
- **Spectrally screened labels.** Polygons whose spectra did not match their
  class were removed. This removes mislabels, but it also removes *atypical*
  examples, which makes the classes look more separable than they really are.
  Accuracy measured on these labels is therefore optimistic.
- **No independent reference.** Labels were checked only against the same
  Sentinel-2 image the model uses, not against higher-resolution imagery or
  field data.
- **Limited labels.** Labels are hand-drawn by one person on a single image,
  with no field validation.
- **Ambiguous land cover.** Dry fields in July sit between "vegetation" and
  "bare soil". Ambiguous areas are left unlabelled rather than guessed.

## Attribution

Contains modified Copernicus Sentinel data (2026), processed by Sentinel Hub
through the Copernicus Browser.
