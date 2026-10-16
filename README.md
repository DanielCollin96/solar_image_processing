# Solar Image Processing

A Python package for downloading and preprocessing images from the Solar Dynamics Observatory (SDO) for scientific analysis and machine learning applications in space weather forecasting. It provides the image data for PROSWIN, a probabilistic deep-learning model for forecasting the solar wind speed.

---

## PROSWIN

PROSWIN is a probabilistic deep-learning model for forecasting the solar wind speed at L1. It combines SDO/AIA solar images with physical features describing the solar wind conditions of the previous solar rotation and the state of the solar cycle. From these inputs it predicts a conditional log-normal distribution of the solar wind speed with a lead time of 4 days. This repository contains the code for preparing the image data, training the model and evaluating its predictions.

### Data sources for operational forecasting

Definitive JSOC lev1 data is published with a delay of about 7 days. An operational forecast cannot rely on it alone, because the newest available images would already be older than the 4-day lead time. SIDC quicklook images (level 1.5) are available within about 15–20 minutes of observation. The realtime path of this pipeline therefore uses SIDC data as model input, which makes a forecast 4 days ahead possible in practice. JSOC lev1 remains the source of the historical archive used for training.

### Model architecture

PROSWIN consists of an image-encoding backbone and two prediction heads, which predict the two parameters μ and σ of a log-normal distribution. The model is trained in two sequential stages.

**Backbone.** A pretrained Swin Transformer encodes a stack of three single-channel images, i.e. three N×N images combined into one N×N×3 input. The stack can be built from AIA images, HMI magnetograms or a mixture of both; the operational model uses AIA images only, for performance reasons, while magnetograms remain available for architecture experiments. The encoded feature vector is concatenated with the physical feature vector.

**Prediction heads.** Each prediction head consists of two fully connected layers with dropout. One head predicts the μ parameter and the other the σ parameter of the predictive log-normal distribution.

**Two-stage training.**

| Stage | Trained part | Loss | Output |
|---|---|---|---|
| μ | Image encoder + μ head | Weighted MSE | Expected solar wind speed (conditional mean) |
| σ | σ head | Negative log-likelihood | Conditional log-normal distribution |

The image encoder and the μ head are frozen during σ training, so the uncertainty estimate is built on top of a fixed point forecast. This is possible because a log transform maps the targets to a normal distribution, in which mean (μ) and variance (σ) decouple. The weighted MSE emphasises high-speed stream events.

**Predictive distribution.** The network outputs live in a standardised log space. Inverting the scaler and re-parameterising yields a shifted log-normal distribution with parameters (shape, scale, loc).

**Physical features.**

- Solar wind speed from the previous solar rotation (hourly values from 28 to 26 days ago)
- Heliospheric latitude of the Earth relative to the solar equatorial plane
- Monthly sunspot numbers over the last year
- Progress of the solar cycle in years

---

## Publication and Citation

This processing pipeline was developed to create the [dataset](https://doi.org/tbd) for the model and research study [Probabilistic Solar Wind Speed Forecasting Using Deep Distributional Regression From Solar Images](https://doi.org/tbd), published in *TBD* (2026). If you use this software or the data in your research, please cite:

```bibtex
@article{Collin2026journal,
  title   = {TBD},
  author  = {Collin, Daniel and Shprits, Yuri and Chiarabini, Luca and Hofmeister, Stefan J. and Klein, Nadja and Gallego, Guillermo},
  journal = {TBD},
  year    = {2026},
  doi     = {TBD}
}

@misc{Collin2026dataset,
  title        = {TBD},
  author       = {Collin, Daniel and Shprits, Yuri and Chiarabini, Luca and Hofmeister, Stefan J. and Klein, Nadja and Gallego, Guillermo},
  publisher    = {GFZ Data Services},
  year         = {2026},
  howpublished = {Data Set},
  doi          = {10.5880/GFZ.2.7.2026.XXX}
}
```

[Journal Publication](https://doi.org/tbd) | [Data Publication](https://doi.org/tbd)

---

## Pipeline Overview

The pipeline processes images from **two data sources** with a single set of scripts:

| Source | Product | Latency | Used for |
|---|---|---|---|
| **JSOC** | Definitive AIA lev1 science data, hourly | about 7 days | Historical archive, model training |
| **SIDC** | AIA quicklook data (level 1.5), every 30 minutes | about 15–20 minutes | Realtime model input |

Each source passes through three stages:

| Stage | Script | Description |
|---|---|---|
| **Download** | `scripts/download_solar_images.py` | Download raw FITS files from JSOC and SIDC |
| **Preprocess** | `scripts/preprocess_solar_images.py` | Apply instrument and data corrections to produce 1024×1024 full-disk images |
| **Crop** | `scripts/crop_solar_images.py` | Downsample, crop and resize the preprocessed images to the model input size |

Supported instruments:

- **AIA** (Atmospheric Imaging Assembly): EUV channels 171, 193 and 211 Å. Downloaded, preprocessed and cropped.
- **HMI** (Helioseismic and Magnetic Imager): line-of-sight magnetograms. HMI is no longer downloaded. The preprocessing code (`hmi_preprocessor.py`) is retained so that historical magnetograms on disk can still be processed.

---

## Project Structure

```
solar_image_processing/
├── README.md
├── pyproject.toml              # Package metadata and dependencies
├── uv.lock                     # Locked dependency versions
├── run_pipeline_jsoc.sh        # Nightly JSOC chain (download → preprocess → crop)
├── run_pipeline_sidc.sh        # Nightly SIDC chain (download → preprocess → crop)
├── configs/
│   └── pipeline_config.yaml    # Paths, channels, date range and per-step options
├── scripts/                    # Command-line entry points (argument parsing only)
│   ├── download_solar_images.py
│   ├── preprocess_solar_images.py
│   ├── crop_solar_images.py
│   └── maintenance/            # Diagnostic and maintenance tools
│       └── check_year_coverage.py
├── src/
│   └── solar_image_processing/
│       ├── cropping/
│       │   └── solar_image_cropper.py
│       ├── downloading/
│       │   ├── solar_image_downloader.py   # Download logic for both sources
│       │   └── jsoc_download.py            # Low-level JSOC client (drms search and export)
│       ├── preprocessing/
│       │   ├── aia_preprocessor.py
│       │   ├── hmi_preprocessor.py
│       │   ├── preprocessing_functions.py
│       │   └── solar_image_preprocessor.py
│       ├── psf_deconvolution/
│       │   ├── deconvolve_image.py
│       │   └── rebin_psf.py
│       └── utils/
│           ├── helper_functions.py
│           └── pipeline_config.py
├── logs/                       # Cron logs (cron_jsoc_YYYY-MM-DD.log, cron_sidc_YYYY-MM-DD.log)
└── data/                       # Created at runtime
    ├── unprocessed_images/               # JSOC raw FITS
    ├── unprocessed_images_realtime/      # SIDC raw FITS
    ├── preprocessed_images/
    │   ├── uncropped/                     # JSOC preprocessed (1024×1024)
    │   └── proswin/                       # JSOC cropped (224×224, model input)
    ├── preprocessed_images_realtime/
    │   ├── uncropped/                     # SIDC preprocessed (1024×1024)
    │   └── proswin/                       # SIDC cropped (224×224, model input)
    └── instrument_data/                  # Cached PSF, degradation and pointing tables
```

The download logic is split into two layers. `solar_image_downloader.py` decides what to fetch for each source and channel and retrieves the SIDC files over HTTP. JSOC requests are delegated to `jsoc_download.py`, a thin client that performs the search and export through `drms`.

All data trees are organised day-wise (`.../{channel}/YYYY/MM/DD/`).

---

## Installation

### 1. Install uv

[uv](https://docs.astral.sh/uv/) is a fast Python package and project manager.

**Linux / macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify the installation with `uv --version`.

### 2. Create the virtual environment

From the project root, run:
```bash
uv sync
```

This reads `pyproject.toml` and `uv.lock`, creates a `.venv/` directory and installs all pinned dependencies.

### 3. Activate the environment (optional)

**Linux / macOS:**
```bash
source .venv/bin/activate
```

**Windows:**
```powershell
.venv\Scripts\activate
```

Alternatively, prefix every command with `uv run`, as in the examples below; no activation is needed then.

### 4. Register with JSOC

Downloading JSOC data requires a registered email address:

1. Visit <http://jsoc.stanford.edu/ajax/exportdata.html>
2. Enter your email address and click **Check Email**
3. Confirm the registration link sent to your inbox
4. Set `download.email` in `configs/pipeline_config.yaml` to this address

---

## Configuration

The pipeline is controlled in two places:

| | `configs/pipeline_config.yaml` | Command-line flags |
|---|---|---|
| Defines | **What** is processed and **where** it is stored | **Which source** and **which time window** a run covers |
| Contains | Data folders, channels, JSOC settings, preprocessing and cropping options, archive date range | `--jsoc`/`--sidc`, `--source`, `--days`, `--start`/`--end`, `--all` |
| Changes | Rarely | Per run |

Every folder a script reads from or writes to comes from the configuration file; no flag sets a folder.

**Locating the configuration.** All scripts load `configs/pipeline_config.yaml` by default. The download script looks for it relative to the current working directory and must therefore be run from the project root; the preprocessing and cropping scripts locate it relative to their own location and run from any directory. All three accept `--config PATH` to load a different file instead. This is useful for test runs, e.g. a copy of the configuration that writes to separate scratch folders, covers a short date range or processes a single channel, so that new code or settings can be tried without touching the production data. A separate configuration is also the way to preprocess historical HMI data (see [Channels](#channels)).

**Folders are created on load.** When the configuration is loaded, every directory listed under `paths` is created if it does not exist. A wrong `base_dir` therefore does not raise an error but silently creates a new, empty `data/` tree. The scripts then find nothing on disk: the JSOC downloader restarts from `start_date`, the SIDC downloader fetches the last 14 days, and preprocessing finds no raw files. If a run unexpectedly starts from scratch, check `base_dir` and the working directory first.

### Base directory and paths

```yaml
base_dir: null   # null = the folder containing configs/ (the project root)

paths:
  unprocessed:            data/unprocessed_images/SDO                  # JSOC raw FITS
  unprocessed_realtime:   data/unprocessed_images_realtime/SDO         # SIDC raw FITS
  preprocessed:           data/preprocessed_images/uncropped           # JSOC preprocessed
  preprocessed_realtime:  data/preprocessed_images_realtime/uncropped  # SIDC preprocessed
  cropped:                data/preprocessed_images/proswin             # JSOC cropped (model input)
  cropped_realtime:       data/preprocessed_images_realtime/proswin    # SIDC cropped (model input)
  instrument_data:        data/instrument_data                         # calibration caches
```

> **Note:** the paths above are relative to the project and move with it, but the `cd` line in `run_pipeline_*.sh` and the crontab entries use absolute paths. These must be updated by hand when the project is moved; otherwise cron keeps running the old copy, whose configuration writes to the old location.

### Date range

```yaml
start_date: "2026-03-01 00:00:00"
end_date:   "2026-07-29 23:00:00"
```

Supported formats are `"YYYY-MM-DD"` and `"YYYY-MM-DD HH:MM:SS"`. The archive range is used only in these cases:

- **Download (JSOC):** `start_date` is the starting point when a channel folder is still empty, or with `--backfill --jsoc`. The downloader ignores `end_date` and always runs up to the present.
- **Preprocess and crop:** with `--all`.

Routine runs do not use it: the downloaders resume from the data on disk, and preprocessing and cropping cover the last 14 days.

### Channels

```yaml
channels:
  - aia_171
  - aia_193
  - aia_211
```

The channels to download and process. List only AIA channels in the main configuration. To preprocess historical HMI data, use a separate configuration that lists `hmi` and run it with `--source jsoc`. Adding `hmi` to the main configuration would make the SIDC run look for HMI quicklook data, which does not exist.

### Download options

```yaml
download:
  email: your@email.com # JSOC-registered email address
  rebin_factor: 4       # JSOC rebins before download: 1 = 4096×4096 px, 4 = 1024×1024 px
  jsoc_series:
    aia:
      series: AIA.lev1_euv_12s
      segment: image
```

These settings apply to JSOC only. SIDC needs no settings: its files are always 1024×1024 px and are fetched over plain HTTP.

```yaml
download_workers:
  - email: your@email.com
    channels:
      - aia_171
      - aia_193
      - aia_211
```

`download_workers` is intended for large-volume JSOC downloads, for example re-downloading the full archive or many channels. It distributes the channels across several JSOC-registered email addresses so that several export requests can run in parallel, one worker per address; every channel must appear in exactly one worker. The routine download script does not read this block and uses the single `download.email` above.

### Preprocessing options

```yaml
preprocessing:
  use_gpu: false                   # true requires a CUDA 12 GPU and cupy
  differential_rotation: true      # rotate substitute images to the target time
  target_rsun_arcsec: 976.0        # normalise all solar disks to this radius
  overwrite_existing: false        # reprocess hours that already have output
  load_preprocessing_fails: false  # load the list of earlier failures and skip them
```

- **`overwrite_existing`**: with `false`, hours that already have output are skipped. Set it to `true` to force reprocessing, for example to apply an updated correction.
- **`load_preprocessing_fails`**: each month folder contains a `preprocessing_fails.csv` listing the hours that could not be processed (bad quality or no raw file). With `true`, this list is loaded and those hours are skipped. With `false` (default), it is not loaded, so every missing hour is attempted again on each run. This key is the only control for retrying failed hours and works identically for both sources. Entries for hours outside a run's time window are kept unchanged.

### Cropping options

```yaml
cropping:
  downsample_resolution: 512  # downsample to this size before cropping
  crop_mode: square           # 'square' or 'disk'
  crop_pixels: 300            # side length of the square crop
  resize_cropped: 224         # final size; null = no resize
```

---

## Usage

Run the scripts from the project root. Each step is incremental: it checks what already exists on disk and performs only the missing work, so re-running a command is always safe.

### 1. Download

```bash
uv run python scripts/download_solar_images.py            # both sources
uv run python scripts/download_solar_images.py --jsoc     # JSOC only
uv run python scripts/download_solar_images.py --sidc     # SIDC only
```

**Resuming.** Routine downloads need no dates. For each channel, the downloader finds the newest file on disk and continues from there up to the present:

- **JSOC:** starts on the day after the newest file and requests one image per hour. Each day is first requested as a single batch; if that fails, the hours are requested individually. Because JSOC publishes about 6–7 days behind real time, the most recent days return no data and are retried automatically on the next run.
- **SIDC:** restarts on the day of the newest file and skips files that already exist, so a partially downloaded day is completed.

If a channel folder is empty, JSOC starts from `start_date` and SIDC from 14 days before the present.

**SIDC 30-minute selection.** The SIDC archive provides a frame every 3 minutes, of which the downloader keeps one per **:00 and :30 slot**: the frame closest to the slot within ±6 minutes. If the exact frame is missing, the nearest neighbour (e.g. :57 or :03) is used; if no frame lies within ±6 minutes, the slot is skipped. Slots younger than 30 minutes are deferred to the next run, because SIDC publishes with a delay of 15–20 minutes and the exact frame may not yet be available. The selection applies to all SIDC downloads, including backfills, and is set by the constants `SLOT_MINUTES`, `SLOT_TOLERANCE_MINUTES` and `SLOT_SETTLE_MINUTES` in `SIDCDownloader`.

**Flags**

| Flag | Default | Description |
|---|---|---|
| `--jsoc` | off | Download JSOC only. |
| `--sidc` | off | Download SIDC only. Without `--jsoc` or `--sidc`, both sources are downloaded. |
| `--backfill` | off | Ignore the data on disk; requires `--jsoc` or `--sidc`. `--backfill --jsoc` requests every day from `start_date` onwards, which can be the full archive. `--backfill --sidc` crawls the SIDC server month by month from `--backfill-start` to the present and downloads all missing files. |
| `--backfill-start YYYY-MM` | `2026-01` | First month for `--backfill --sidc`; ignored for JSOC. |
| `--coverage` | off | Report the newest downloaded date and the resume date per channel for both sources, then exit without downloading. |
| `--dry-run` | off | Report which days each channel would request, then exit without downloading or contacting the servers. |
| `--config PATH` | `./configs/pipeline_config.yaml` | Use a different configuration file. |

**Output:** raw FITS files and metadata pickles under `data/unprocessed_images/SDO/` (JSOC) and `data/unprocessed_images_realtime/SDO/` (SIDC), organised as `.../AIA/{171|193|211}/YYYY/MM/DD/`.

### Choosing source and time window

Preprocessing and cropping use the same flags. Each run handles one source and one time window:

| Flag | Default | Description |
|---|---|---|
| `--source {jsoc,sidc}` | required | `jsoc` processes JSOC lev1 data, `sidc` processes SIDC quicklook data. |
| `--days N` | `14` | Process the last N days up to the present. |
| `--start DATE --end DATE` | — | Process an explicit range. Both must be given; a date without a time (`YYYY-MM-DD`) as `--end` includes that whole day. |
| `--all` | off | Process the archive range `start_date` → `end_date` from the configuration. |
| `--config PATH` | `configs/pipeline_config.yaml` | Use a different configuration file. |

Only one of `--days`, `--start`/`--end` and `--all` may be given; combinations are rejected with an error. Without any of them, the last 14 days are processed, which covers the JSOC publication delay. A longer gap, for example after a server outage, is filled with a single run over an explicit range:

```bash
uv run python scripts/preprocess_solar_images.py --source jsoc --start 2026-09-01 --end 2026-09-30
uv run python scripts/crop_solar_images.py --source jsoc --start 2026-09-01 --end 2026-09-30
```

### 2. Preprocess

```bash
uv run python scripts/preprocess_solar_images.py --source jsoc   # JSOC, last 14 days
uv run python scripts/preprocess_solar_images.py --source sidc   # SIDC, last 14 days
uv run python scripts/preprocess_solar_images.py --source jsoc --all
```

**Processing a window.** For every channel, the preprocessor builds the hourly target times inside the window, up to the newest raw file. Hours that already have output are skipped, so a routine run processes the new hours plus any older gaps that now have raw data.

**Gap filling.** For each missing hour, the temporally closest raw file that passes the quality checks is used, searched within ±24.5 h. If it lies more than 6 minutes from the target hour, it is differentially rotated to the target time (if `differential_rotation` is enabled). If no usable file exists within ±24.5 h, the hour is recorded in `preprocessing_fails.csv` and left as a gap.

**Quality checks.** A frame is used only if it contains the full solar disk, has valid WCS keywords and passes the `QUALITY` screen. JSOC lev1 frames require `QUALITY == 0`. SIDC quicklook frames always carry bit 30 ("quicklook image"); for SIDC, bits 30, 2 and 21 are treated as benign and any other bit rejects the frame.

**AIA processing steps (JSOC lev1):**
1. Upsample to 4096 px and update the pointing
2. Apply differential rotation (if the gap exceeds 6 minutes)
3. Downsample to 1024 px and deconvolve with the instrument PSF
4. Register to the solar disk centre
5. Normalise the solar disk radius to `target_rsun_arcsec`
6. Correct for instrument degradation
7. Normalise by exposure time (DN/s)
8. Flip to solar north up

**SIDC quicklook (lev1.5)** data is already pointed, registered and binned to 1024 px. It therefore skips the upsampling and pointing update (step 1) and the registration (step 4); all other steps, including the differential rotation of substitute frames, are applied as for JSOC. The data level is read from the `LVL_NUM` FITS header keyword. Before frames are matched to target hours, the SIDC frame closest to each full hour is selected.

**HMI processing steps** (historical data only): upsample → differential rotation → downsample → register with zero fill → replace NaNs with zero → normalise disk radius → flip.

**Calibration data** (PSF, degradation correction and pointing tables) is loaded from `instrument_data/` on first use and cached there as pickle files. The pointing table is used for JSOC lev1 only.

**Output:** one file pair per hour for both sources, under `data/preprocessed_images/uncropped/{channel}/YYYY/MM/DD/` (JSOC) or `data/preprocessed_images_realtime/uncropped/{channel}/YYYY/MM/DD/` (SIDC):
```
{channel}_{YYYY-MM-DD_HH:MM}.npy          # image array
{channel}_{YYYY-MM-DD_HH:MM}_meta.pickle  # FITS header metadata
```

### 3. Crop

```bash
uv run python scripts/crop_solar_images.py --source jsoc   # JSOC, last 14 days
uv run python scripts/crop_solar_images.py --source sidc   # SIDC, last 14 days
```

The cropper processes the preprocessed images inside the window that do not have a cropped counterpart yet. Each image is downsampled by block-sum reduction, which preserves the total flux, then cropped and optionally resized. The crop is set by `crop_mode` in the configuration:

- **`square`**: a symmetric crop of `crop_pixels` centred on the image, followed by an optional bicubic resize to `resize_cropped`.
- **`disk`**: a crop to the solar disk boundary, using the disk radius stored in the metadata pickle.

Cropping has been validated for the AIA channels.

**Output:** `float32` arrays under `data/preprocessed_images/proswin/` (JSOC) or `data/preprocessed_images_realtime/proswin/` (SIDC), organised day-wise with the same file names as the preprocessed input.

---

## Routine Operation

In production, the three steps run once per night for each source, chained in a shell script. Each script changes into the project directory, runs download → preprocess → crop (each step only if the previous one succeeded) and appends all output to a dated log in `logs/`.

```bash
# run_pipeline_jsoc.sh (simplified)
cd /export/solardl/solar_image_processing || exit 1
uv run python scripts/download_solar_images.py --jsoc
uv run python scripts/preprocess_solar_images.py --source jsoc
uv run python scripts/crop_solar_images.py --source jsoc

# run_pipeline_sidc.sh (simplified)
cd /export/solardl/solar_image_processing || exit 1
uv run python scripts/download_solar_images.py --sidc
uv run python scripts/preprocess_solar_images.py --source sidc
uv run python scripts/crop_solar_images.py --source sidc
```

Crontab (`crontab -e`):

```
PATH=/home/ravito/.local/bin:/usr/local/bin:/usr/bin:/bin
30 2 * * * /export/solardl/solar_image_processing/run_pipeline_jsoc.sh
0 5 * * * /export/solardl/solar_image_processing/run_pipeline_sidc.sh
```

The 14-day window overlaps generously with the nightly schedule, so a missed night or late-arriving JSOC data is filled in automatically by the next run. To catch up immediately, run both scripts manually in a `screen` session:

```bash
screen -S catchup bash -c './run_pipeline_sidc.sh; ./run_pipeline_jsoc.sh'
tail -f logs/cron_sidc_$(date +%F).log
```

A run has finished when its log shows `=== ... pipeline end ... (exit 0) ===`. If the project is moved, update the `cd` line in both scripts and both crontab entries.

---

## GPU Acceleration

PSF deconvolution (AIA only) can be accelerated with a CUDA-capable GPU:

1. Ensure a CUDA 12 compatible GPU and driver are available.
2. The `cupy-cuda12x` package is listed as a dependency and installed by `uv sync`.
3. Set `use_gpu: true` in the `preprocessing` section of the configuration.

Without a GPU, deconvolution runs on the CPU and is considerably slower.

---

## PROSWIN Model: Training and Evaluation

### Data

The model expects its input data in the following layout:

```
<data_root>/
├── solar_images/              # SDO/AIA EUV images (.npy, one file per hour)
│   ├── 171/                   # AIA 171 Å channel
│   │   └── <year>/<month>/
│   ├── 193/                   # AIA 193 Å channel
│   ├── 211/                   # AIA 211 Å channel
│   └── HMI/                   # HMI line-of-sight magnetograms
├── machine_learning/
│   ├── ml_dataset.csv         # Feature values
│   └── literature_models.csv  # Required for the literature comparison
├── physical_features/         # Scalar time series (CSV)
│   ├── solar_wind_speed.csv   # Required for the baseline model comparison
│   ├── sunspot_number.csv     # Required for the solar cycle plot
│   └── ch_area.csv            # Required for the baseline model comparison
└── solar_events/              # Event catalogues (CSV)
    ├── cme_list.csv
    ├── hss_list.csv
    └── enhancement_list.csv
```

To reuse or evaluate trained models or predictions, copy them into the `results` directory:

```
<results_root>/
├── models/                     # Trained models
│   ├── full_model/             # Trained with all features, one model per CV fold
│   │   └── <model>_<fold>.pth
│   └── image_only/             # Trained without physical features, one model per CV fold
│       └── <model>_<fold>.pth
└── predictions/                # Computed predictions
    ├── <model>_full_model.csv  # Full-model predictions aggregated over all CV folds
    └── <model>_image_only.csv  # Image-only predictions aggregated over all CV folds
```

### Configuration

All model scripts read exclusively from `configs/config.yaml` and take no command-line arguments. Edit the configuration before each run:

- Set all `paths` entries to your data directories
- Configure the dataset and input data (e.g. image channels) under `dataset`
- Set the model hyperparameters and the model name, which identifies the outputs of a run, under `model`
- Set the training hyperparameters for the μ and σ stages under `training`
- Set `use_multiple_gpus: true` to train with DDP on multiple GPUs
- Optionally set `wandb_project` for experiment tracking

### Training

To train on a fixed train/validation/test split, set `TRAIN_DATES`, `VAL_DATES` and `TEST_DATES` in `scripts/train_single_split.py`, then run:

```bash
uv run scripts/train_single_split.py
```

### Prediction

To predict with a trained model, set `MODEL_PATH` to the saved checkpoint and the prediction period with `PREDICTION_START` and `PREDICTION_END` in `scripts/predict.py`, then run:

```bash
uv run scripts/predict.py
```

### Evaluation

To evaluate saved predictions and produce evaluation plots, set `PREDICTION_PATH` in `scripts/evaluate_predictions.py`, then run:

```bash
uv run scripts/evaluate_predictions.py
```

### Working with probabilistic predictions

The predictions are based on `scipy.stats.lognorm`. The distribution parameters are available as `prediction.shape`, `prediction.scale` and `prediction.loc`, and derived quantities can be computed directly:

```python
prediction.get_expected_values()             # mean
prediction.get_quantiles(0.5)                # quantiles, e.g. the median
prediction.get_quantiles([0.025, 0.975])     # 95% interval
prediction.get_cdfs(500)                     # CDF at 500 km/s
prediction.get_pdfs(500)                     # PDF at 500 km/s
prediction.get_crps(observation)             # per-sample CRPS
```

See `predictions.py` for the full API.

### Reproducing the experiments

**Cross-validation.** The full cross-validation is run with:

```bash
uv run scripts/cross_validation.py
```

This builds and caches the dataset, generates the 5-fold cross-validation split, trains each fold in two stages (μ, then σ), evaluates each fold, aggregates the predictions across all test periods and saves the results in the `results/` subdirectories.

**Model comparison.** All model variants from the paper and the literature models are compared with:

```bash
uv run scripts/compare_models.py
```

---

## Code Developers

[Daniel Collin](https://www.linkedin.com/in/daniel-collin-52abb0205/) — original pipeline.

For questions or bug reports, please contact Daniel at collin@gfz.de.

---

## License

This work is released under the [MIT License](https://github.com/DanielCollin96/INSERT_LATER).

---

## References

- Publication (please cite): **Insert when submitted**
- SDO Mission: <https://sdo.gsfc.nasa.gov/>
- JSOC Data Center: <http://jsoc.stanford.edu/>
- SIDC / Royal Observatory of Belgium: <https://www.sidc.be/>
- SunPy: <https://sunpy.org/>
- aiapy: <https://aiapy.readthedocs.io/>
- uv: <https://docs.astral.sh/uv/>
