# Solar Image Processing

A Python software for downloading and preprocessing SDO (Solar Dynamics Observatory)
solar images for scientific analysis and machine learning applications in space weather
forecasting.

Solar images are sourced from the [Joint Science Operations Center (JSOC)](http://jsoc.stanford.edu/) and in near real-time from the [Solar Influences Data analysis Center (SIDC)](https://sdo.oma.be/data/aia_quicklook/). 

If you use this software or the data in your research, please [cite it as shown below](#citation).

---
## Overview

This package provides a three-stage pipeline that handles two data sources from a single set of scripts:

- **JSOC lev1** — definitive science data, published with a latency of roughly one week.
- **SIDC quicklook** — near-real-time data (level 1.5), available within minutes.

This package provides a three-stage pipeline:

| Stage | Script | What it does                                                |
|---|---|-------------------------------------------------------------|
| **Download** | `scripts/download_solar_images.py` | Download raw FITS from JSOC (lev1) and SIDC (quicklook) |
| **Preprocess** | `scripts/preprocess_solar_images.py` | Apply instrument- and data related corrections              |
| **Crop** | `scripts/crop_solar_images.py` | Downsample, crop, and optionally resize preprocessed images |

Supported instruments:

- **AIA** (Atmospheric Imaging Assembly): EUV channels 171, 193, 211 Å
- **HMI** (Helioseismic and Magnetic Imager): line-of-sight magnetograms

---

## Getting Started

This project uses [uv](https://docs.astral.sh/uv/) for extremely fast Python package and project management.

### 1. Install uv

If you don't have uv installed yet, install it:

#### Linux / MacOS

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Windows (Powershell)

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### Verify the installation:

```bash
uv --version
```

### 2. Project Setup

After cloning the repository, navigate to the project directory and sync the dependencies. 
This will automatically create a virtual environment (`.venv`) and install all required packages from the `uv.lock` file.

```bash
# Clone the repository
git clone https://github.com/DanielCollin96/solar_image_processing.git
cd solar_image_processing

# Create a virtual environment and install all dependencies
uv sync
```


### 3. Running Scripts

You do not need to activate the virtual environment manually. 
Use `uv run` to execute scripts:

```bash
uv run <script>.py
```


### 4. Register with JSOC

Downloading data from JSOC requires a JSOC-registered email address:

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
  unprocessed:           data/unprocessed_images/SDO                       # JSOC raw FITS
  unprocessed_realtime:  data/unprocessed_images_realtime/SDO              # SIDC quicklook raw FITS
  preprocessed:          data/preprocessed_images/uncropped                # JSOC preprocessed
  preprocessed_realtime: data/preprocessed_images_realtime/uncropped       # SIDC preprocessed
  cropped:               data/preprocessed_images/proswin                  # JSOC cropped
  cropped_realtime:      data/preprocessed_images_realtime/proswin         # SIDC cropped
  instrument_data:       data/instrument_data                              # calibration caches
```

All paths are relative to the project root (or to `base_dir` if set).
Directories are created automatically on first run.

### Date range

```yaml
start_date: "2026-03-01 00:00:00"
end_date:   "2026-07-29 23:00:00"
```

The time range used when a run is not bounded by `--days` or `--start/--end`. Supported formats: `"YYYY-MM-DD"` or `"YYYY-MM-DD HH:MM:SS"`.

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
`overwrite_existing` matters when re-running over dates that already have output: with `false`, existing files are skipped; set it to `true` to force reprocessing (for example, to re-apply an updated correction to already-processed frames).

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

All scripts read configuration from `configs/pipeline_config.yaml` (resolved relative to the package, so they can be run from the project root). Each stage takes a required `--source {jsoc, sidc}` selecting the data source.

### 1. Download

```bash
uv run python scripts/download_solar_images.py --source jsoc
uv run python scripts/download_solar_images.py --source sidc
```

Downloads images for all channels listed in `channels`. JSOC (lev1) is published roughly a week behind real time and downloaded at an hourly cadence. SIDC (quicklook, level 1.5) is downloaded at a 30-minute cadence (two files per hour).

**Output:** raw FITS files and per-request metadata pickles under `data/unprocessed_images/SDO/` (JSOC) and `data/unprocessed_images_realtime/SDO/` (SIDC), organised day-wise as `.../AIA/{171|193|211}/YYYY/MM/DD/`.

### 2. Preprocess

```bash
# JSOC lev1 — recent window (default 14 days)
uv run python scripts/preprocess_solar_images.py --source jsoc

# SIDC quicklook — recent window
uv run python scripts/preprocess_solar_images.py --source sidc

# bound an explicit range, or process everything available
uv run python scripts/preprocess_solar_images.py --source jsoc --start 2026-03-01 --end 2026-03-31
uv run python scripts/preprocess_solar_images.py --source jsoc --all
```

Common flags: `--days N` (rolling recent window, default 14), `--start/--end` (explicit date range), `--all` (ignore the window, process the full configured range), `--config PATH` (use an alternate config).

Missing hourly observations are filled using the temporally closest available raw file within a ±24.5 h window; differential rotation is applied when the time gap exceeds 6 minutes (if enabled).

**AIA pipeline per image (JSOC lev1):**

1. Upsample to 4096 px and update pointing
2. Apply differential rotation (if gap > 6 min)
3. Downsample to 1024 px and deconvolve with the instrument PSF
4. Register to solar disk centre
5. Normalise solar disk radius to `target_rsun_arcsec`
6. Correct for instrument degradation
7. Normalise by exposure time (DN/s)
8. Flip to solar north-up orientation

SIDC quicklook (lev1.5) is already pointed, registered and binned to 1024, so it skips steps 1, 2 (pointing) and 4 and runs the remaining corrections (deconvolution, disk-radius normalisation, degradation correction, exposure normalisation, flip).

**HMI pipeline per image (historical data only):** upsample → differential rotation → downsample → register with zero fill → replace NaNs with zero → normalise disk radius → flip.

Calibration data (PSF, degradation correction, pointing tables) is loaded from `instrument_data/` on first use and cached there as pickle files. The pointing table is used for JSOC lev1 only; SIDC quicklook skips it.

**Output:** one file pair per timestep, day-wise under `data/preprocessed_images/uncropped/{channel}/YYYY/MM/DD/` (JSOC) or `data/preprocessed_images_realtime/uncropped/...` (SIDC):

```
{channel}_{YYYY-MM-DD_HH:MM}.npy           # image array (float32)
{channel}_{YYYY-MM-DD_HH:MM}_meta.pickle   # FITS header metadata
```

### 3. Crop

```bash
uv run python scripts/crop_solar_images.py --source jsoc --start 2024-06-01 --end 2024-06-30
uv run python scripts/crop_solar_images.py --source sidc --start 2026-01-01 --end 2026-04-30
```

Reads preprocessed `.npy` files, downsamples them using block-sum reduction (which preserves total flux), crops, and optionally resizes. Two crop modes are available via `crop_mode` in the config: `square` (symmetric pixel crop centred on the image, then an optional bicubic resize) and `disk` (crop to the solar disk boundary using the disk radius in the metadata pickle).

**Output:** cropped arrays saved day-wise under `data/preprocessed_images/proswin/` (JSOC) or `data/preprocessed_images_realtime/proswin/` (SIDC), with the same filename as the preprocessed input.



---

## GPU Acceleration

PSF deconvolution (AIA only) can be accelerated with a CUDA-capable GPU:

1. Ensure you have a CUDA 12 compatible GPU and driver.
2. The `cupy-cuda12x` package is already listed as a dependency and installed
   by `uv sync`.
3. Set `use_gpu: true` in the `preprocessing` section of the config.

Without a GPU, deconvolution runs on the CPU and is significantly slower.

---

## Citation

This processing pipeline was developed to create the [dataset](https://doi.org/10.5880/GFZ.OJSJ.2026.001) for the model and research study [PROSWIN: Probabilistic Solar Wind Speed Forecasting Using Deep Distributional Regression From Solar Images](https://doi.org/10.1029/2026AV002524) published in *AGU Advances* (2026). If you use this software or the data in your research, please cite it as follows:


```bibtex
@article{Collin2026,
    author = {Collin, Daniel and Shprits, Yuri and Chiarabini, Luca and Hofmeister, Stefan J. and Klein, Nadja and Gallego, Guillermo},
    title = {PROSWIN: Probabilistic Solar Wind Speed Forecasting Using Deep Distributional Regression From Solar Images},
    journal = {AGU Advances},
    volume = {7},
    number = {8},
    pages = {e2026AV002524},
    doi = {https://doi.org/10.1029/2026AV002524},
    year = {2026}
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
[Journal Publication](https://doi.org/10.1029/2026AV002524) | [Data Publication](https://doi.org/10.5880/GFZ.OJSJ.2026.001) 

The deconvolution algorithm with the instrumental point-spread functions uses the works of Stefan J. Hofmeister ([[1]](https://doi.org/10.3847/1538-4365/adbaed), [[2]](https://doi.org/10.7910/DVN/DYT4ZL), [[3]](https://doi.org/10.1007/s11207-024-02321-y)) listed below in the References section.


---

## Code developer

- [Daniel Collin](https://www.linkedin.com/in/daniel-collin-52abb0205/). 
- [Ravi Shankar Chaurasia](https://www.linkedin.com/in/ravishankarchaurasia/)

In case of questions or bugs, please contact Daniel at collin@gfz.de or Ravi at ravito@gfz.de.

---

## License

This work is released under [MIT License](https://github.com/DanielCollin96/solar_image_processing/blob/main/LICENSE).

---

## References

- Publication (please cite): [Collin et al., (2026). AGU Advances, 7, e2026AV002524.](https://doi.org/10.1029/2026AV002524)
- Point-spread functions: [1] [Hofmeister et al., (2025). ApJS, 278(1), 8](https://doi.org/10.3847/1538-4365/adbaed); [2] Dataset: [Hofmeister (2024)](https://doi.org/10.7910/DVN/DYT4ZL); 
- Deconvolution: [3] [Hofmeister (2024). Sol Phys 299, 77](https://doi.org/10.1007/s11207-024-02321-y)
- SDO Mission: <https://sdo.gsfc.nasa.gov/>
- JSOC Data Center: <http://jsoc.stanford.edu/>
- SIDC / Royal Observatory of Belgium: <https://www.sidc.be/>
- SunPy: <https://sunpy.org/>
- aiapy: <https://aiapy.readthedocs.io/>
- uv: <https://docs.astral.sh/uv/>
