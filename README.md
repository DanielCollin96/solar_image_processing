# Solar Image Processing

A Python software for downloading and preprocessing SDO (Solar Dynamics Observatory)
solar images for scientific analysis and machine learning applications in space weather
forecasting.

## Journal Publication and Citation

This processing pipeline was developed to create the [dataset](https://doi.org/10.5880/GFZ.OJSJ.2026.001) for the model and research study [Probabilistic Solar Wind Speed Forecasting Using Deep Distributional Regression From Solar Images](https://doi.org/tbd) published in *TBD* (2026). If you use this software or the data in your research, please cite it as follows:


```bibtex
@article{Collin2026journal,
  title   = {TBD},
  author = {Collin, Daniel and Shprits, Yuri and  Chiarabini, Luca and Hofmeister, Stefan J. and Klein, Nadja and Gallego, Guillermo},
  journal = {TBD},
  year    = {2026},
  doi     = {TBD}
}

@misc{Collin2026dataset,
  title = {Solar Images and Physical Features for Probabilistic Solar Wind Speed Prediction},
  author = {Collin, Daniel and Shprits, Yuri and  Chiarabini, Luca and Hofmeister, Stefan J. and Klein, Nadja and Gallego, Guillermo},
  publisher = {GFZ Data Services},
  year = {2026},
  howpublished = {Data Set},
  doi = {10.5880/GFZ.OJSJ.2026.001}
}
```
[Journal Publication](https://doi.org/tbd) | [Data Publication](https://doi.org/10.5880/GFZ.OJSJ.2026.001) 

The deconvolution algorithm with the instrumental point-spread functions uses the works of Stefan J. Hofmeister ([[1]](https://doi.org/10.3847/1538-4365/adbaed), [[2]](https://doi.org/10.7910/DVN/DYT4ZL), [[3]](https://doi.org/10.1007/s11207-024-02321-y)) listed below in the References section.

## Overview

This package provides a three-stage pipeline:

| Stage | Script | What it does |
|---|---|---|
| **Download** | `scripts/download_solar_images.py` | Download raw FITS files from JSOC at hourly cadence |
| **Preprocess** | `scripts/preprocess_solar_images.py` | Apply instrument- and data related corrections |
| **Crop** | `scripts/crop_solar_images.py` | Downsample, crop, and optionally resize preprocessed images |

Supported instruments:

- **AIA** (Atmospheric Imaging Assembly): EUV channels 171, 193, 211 Å
- **HMI** (Helioseismic and Magnetic Imager): line-of-sight magnetograms

---

## Project Structure

```
solar_images/
├── README.md
├── pyproject.toml              # Package metadata and dependencies
├── uv.lock                     # Locked dependency versions
├── configs/
│   └── pipeline_config.yaml    # All hyperparameters (edit this)
├── scripts/                    # Executable scripts
│   ├── download_solar_images.py
│   ├── preprocess_solar_images.py
│   └── crop_solar_images.py
├── src/                        # Source code
│   └── solar_image_processing/
│       ├── cropping/
│       │   └── solar_image_cropper.py
│       ├── downloading/
│       │   ├── jsoc_download.py
│       │   └── solar_image_downloader.py
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
├── tests/
│   ├── data/                   # Data for testing (see below)
│   └── test_pipeline.py        # Test script
└── data/                       # Created during script execution
    ├── unprocessed_images/
    ├── preprocessed_images/
    └── instrument_data/
```

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

Verify the installation:
```bash
uv --version
```

### 2. Create the virtual environment

From the project root directory, run:
```bash
uv sync
```

This reads `pyproject.toml` and `uv.lock`, creates a `.venv/` directory, and
installs all pinned dependencies. To also install the optional test dependencies:
```bash
uv sync --extra dev
```

### 3. Activate the virtual environment

**Linux / macOS:**
```bash
source .venv/bin/activate
```

**Windows:**
```powershell
.venv\Scripts\activate
```

After activation, `python` resolves to the project's interpreter and all
dependencies are available. Deactivate with `deactivate`.


### 4. Register with JSOC

Downloading data requires a JSOC-registered email address:

1. Visit <http://jsoc.stanford.edu/ajax/exportdata.html>
2. Enter your email and click **Check Email**
3. Confirm the registration link sent to your inbox
4. Set `email` in `configs/pipeline_config.yaml` to this address

---

## Configuration

All hyperparameters are controlled by a single YAML file:

```
configs/pipeline_config.yaml
```

Edit this file before running any script. The full set of options is documented
inline in the file. Key sections are:

### Paths

```yaml
paths:
  unprocessed:    data/unprocessed_images/SDO               # raw FITS files
  preprocessed:   data/preprocessed_images/deep_learning    # preprocessed files ready for deep learning
  cropped:        data/preprocessed_images/sws_prediction   # further downsampled and cropped files for certain applications
  instrument_data: data/instrument_data                     # instument-related data for preprocessing
```

All paths are relative to the project root (or to `base_dir` if set).
Directories are created automatically on first run.

### Date range

```yaml
start_date: "2010-05-01 00:00:00"
end_date:   "2024-06-30 23:00:00"
```

Put here the time range you want to download or process. Supported formats: `"YYYY-MM-DD"` or `"YYYY-MM-DD HH:MM:SS"`.

### Channels

```yaml
channels:
  - aia_171
  - aia_193
  - aia_211
  - hmi
```

Put here the channels you want to download or process. Remove channels you do not need.

### Download options

```yaml
download:
  rebin_factor: 4       # downsampling factor before downloading (1 = 4096×4096 px, 4 = 1024×1024 px)
  email: your@email.com # email for JSOC download
```

### Preprocessing options

```yaml
preprocessing:
  use_gpu: false                   # true requires a CUDA 12 GPU + cupy
  differential_rotation: true      # rotate substitute images to target time
  target_rsun_arcsec: 976.0        # normalise all disks to this radius
  overwrite_existing: false        # reprocess files that already exist
  load_preprocessing_fails: false  # skip dates that failed previously
```

### Cropping options

```yaml
cropping:
  downsample_resolution: 512  # downsample to this size before cropping
  crop_mode: square           # 'square' or 'disk' (either crop a square of certain pixels or crop at solar disk boundaries)
  crop_pixels: 300            # pixel length of square crop
  resize_cropped: 224         # final size; null = no resize
```

---

## Usage

All scripts read configuration from `configs/pipeline_config.yaml` and must be
run **from inside the `scripts/` directory** so the config file is located
correctly:

```bash
cd scripts
```

### 1. Download

```bash
python download_solar_images.py
```

Downloads images at hourly cadence for all channels listed in `channels` over
the configured date range. For each day, a single batch request is attempted
first; individual hourly requests are used as a fallback if the batch fails.

**Output:** raw FITS files and per-request metadata pickles saved to:
```
data/unprocessed_images/SDO/
├── AIA/
│   ├── 171/YYYY/MM/
│   ├── 193/YYYY/MM/
│   └── 211/YYYY/MM/
└── HMI/
    └── magnetogram/YYYY/MM/
```

### 2. Preprocess

```bash
python preprocess_solar_images.py
```

Processes all channels over the configured date range. Missing hourly
observations are filled using the temporally closest available raw file
within a ±24.5 h window; differential rotation is applied when the time
gap exceeds 6 minutes (if enabled).

**AIA pipeline per image:**
1. Upsample to 4096 px and update pointing
2. Apply differential rotation (if gap > 6 min)
3. Downsample to 1024 px and deconvolve with the instrument PSF
4. Register to solar disk centre
5. Normalise solar disk radius to `target_rsun_arcsec`
6. Correct for instrument degradation
7. Normalise by exposure time (DN/s)
8. Flip to solar north-up orientation

**HMI pipeline per image:**
1. Upsample to 4096 px
2. Apply differential rotation (if gap > 6 min)
3. Downsample to 1024 px
4. Register with zero fill for off-disk regions
5. Replace NaN values with zero
6. Normalise solar disk radius
7. Flip to solar north-up orientation

**Calibration data** (PSF, degradation correction, pointing tables) is loaded
from `instrument_data/` on first use and cached there as pickle files for
subsequent runs.

**Output:** one file pair per hourly timestep saved to
`data/preprocessed_images/deep_learning/{channel}/YYYY/MM/`:
```
{channel}_{YYYY-MM-DD_HH:MM}.npy          # image array (float32)
{channel}_{YYYY-MM-DD_HH:MM}_meta.pickle  # FITS header metadata
```

### 3. Crop

```bash
python crop_solar_images.py
```

Reads preprocessed `.npy` files, downsamples them using block-sum reduction
(which preserves total flux), crops, and optionally resizes.

Two crop modes are available:

- **`square`** — symmetric pixel crop centred on the image, followed by an
  optional bicubic resize.
- **`disk`** — crop to the solar disk boundary, using the disk radius stored
  in the accompanying metadata pickle.

**Output:** cropped arrays saved to `data/preprocessed_images/sws_prediction/`
(or the path set under `paths.cropped`) with the same filename as the
preprocessed input.

---

## GPU Acceleration

PSF deconvolution (AIA only) can be accelerated with a CUDA-capable GPU:

1. Ensure you have a CUDA 12 compatible GPU and driver.
2. The `cupy-cuda12x` package is already listed as a dependency and installed
   by `uv sync`.
3. Set `use_gpu: true` in the `preprocessing` section of the config.

Without a GPU, deconvolution runs on the CPU and is significantly slower.

---

## Running Tests

The integration test runs the full pipeline against a small reference dataset
stored in `tests/data/reference/` and compares results to stored reference
outputs. Reference data can be downloaded here **(insert test data download link)**.

```bash
cd tests
pytest test_pipeline.py -v
```

---

## Code developer

[Daniel Collin](https://www.linkedin.com/in/daniel-collin-52abb0205/). 

In case of questions or bugs, please contact Daniel at collin@gfz.de.

---

## License

This work is released under [MIT License](https://github.com/DanielCollin96/INSERT_LATER).

---

## References

- Publication (please cite): **Insert when submitted**
- Point-spread functions: [1] [Hofmeister et al., (2025). ApJS, 278(1), 8](https://doi.org/10.3847/1538-4365/adbaed); [2] Dataset: [Hofmeister (2024)](https://doi.org/10.7910/DVN/DYT4ZL); 
- Deconvolution: [3] [Hofmeister (2024). Sol Phys 299, 77](https://doi.org/10.1007/s11207-024-02321-y)
- SDO Mission: <https://sdo.gsfc.nasa.gov/>
- JSOC Data Center: <http://jsoc.stanford.edu/>
- SunPy: <https://sunpy.org/>
- aiapy: <https://aiapy.readthedocs.io/>
- uv: <https://docs.astral.sh/uv/>
