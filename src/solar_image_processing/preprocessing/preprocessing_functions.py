from datetime import datetime
from typing import Optional

import astropy.units as u
import numpy as np
import sunpy.map
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.wcs import WCS
from sunpy.coordinates import Helioprojective, propagate_with_solar_surface


def _extract_or_pad_data(data: np.ndarray, target_shape: int) -> np.ndarray:
    """
    Crop the centre or zero-pad a 2D array to reach ``target_shape``.

    Parameters
    ----------
    data : np.ndarray
        Input square 2D array.
    target_shape : int
        Target side length for the square output.

    Returns
    -------
    np.ndarray
        Array with shape ``(target_shape, target_shape)``.
    """
    current_shape = data.shape[0]

    if current_shape > target_shape:
        # Compute symmetric centre crop bounds
        center_pixel = current_shape / 2 - 0.5
        half_size = target_shape / 2
        cutout_start = int(center_pixel - half_size + 0.5)
        cutout_end = int(center_pixel + half_size + 0.5)
        return data[cutout_start:cutout_end, cutout_start:cutout_end]

    elif current_shape < target_shape:
        pad_width = int((target_shape - current_shape) / 2)
        return np.pad(data, pad_width, mode='constant', constant_values=0.0)

    return data


def scale_solar_disk_radius(
    smap: sunpy.map.Map,
    rsun_target: float = 976.0,
    missing: Optional[float] = None,
) -> sunpy.map.Map:
    """
    Rescale a solar image map so the solar disk radius matches ``rsun_target``.

    Normalising all images to the same apparent disk size removes the
    effect of varying Earth–Sun distance throughout the year.

    Parameters
    ----------
    smap : sunpy.map.Map
        Input registered map.
    rsun_target : float, optional
        Target solar radius in arcseconds. Default is 976.0, corresponding
        to approximately one solar radius at 1 AU.
    missing : float, optional
        Fill value for pixels outside the original field of view after
        scaling. Defaults to the minimum value of the input data.

    Returns
    -------
    sunpy.map.Map
        Rescaled map with solar disk radius normalised to ``rsun_target``.
    """
    orig_shape = smap.data.shape[0]
    # Scale factor < 1 shrinks the disk (observer far from Sun), > 1 expands it
    scale_factor = rsun_target / smap.meta['RSUN_OBS']
    missing = smap.data.min() if missing is None else missing

    temp_map = smap.rotate(
        scale=scale_factor,
        order=3,
        missing=missing,
        method='scipy',
    )

    # Restore original image size after rotation may have changed it
    new_img_data = _extract_or_pad_data(temp_map.data, orig_shape)
    return sunpy.map.Map(new_img_data, temp_map.meta)


def register_image(
    smap: sunpy.map.Map,
    missing: Optional[float] = None,
    arcsec_pix_target: Optional[float] = None,
) -> sunpy.map.Map:
    """
    Register an SDO image to a common reference frame.

    Rotates to solar north up, scales to the target pixel scale, and
    centres the solar disk. The reference pixel (``crpix1``, ``crpix2``)
    is set to the image centre in the returned metadata.

    Parameters
    ----------
    smap : sunpy.map.Map
        Input SunPy map to register.
    missing : float, optional
        Fill value for interpolated pixels. Defaults to the map minimum.
    arcsec_pix_target : float, optional
        Target pixel scale in arcseconds per pixel. Defaults to
        ``0.6 * (4096 / image_size)`` arcsec/pix, matching the AIA
        native scale at 4096 px scaled to the working resolution.

    Returns
    -------
    sunpy.map.Map
        Map with original data and metadata updated so that ``crpix1``
        and ``crpix2`` point to the image centre.

    Notes
    -----
    The rotation, scaling, and submap operations are computed internally
    but the returned data array is taken from the original ``smap``.
    Only ``crpix1`` and ``crpix2`` are updated in the output metadata.
    """
    orig_shape = smap.data.shape[0]

    # Default pixel scale: AIA native 0.6 arcsec/pix scaled by downsample factor
    if arcsec_pix_target is None:
        downsample_factor = 4096 / orig_shape
        arcsec_pix_target = 0.6 * downsample_factor

    scale = arcsec_pix_target * u.arcsec
    scale_factor = smap.scale[0] / scale
    missing = smap.min() if missing is None else missing

    # Rotate and scale; recenter=True places the disk centre at crpix
    tempmap = smap.rotate(
        recenter=True,
        scale=scale_factor.value,
        order=3,
        missing=missing,
        method='scipy',
    )

    # Extract a sub-region of the original size from the (possibly larger) rotated map
    # crpix1 == crpix2 because recenter=True produces a symmetric output
    center = np.floor(tempmap.meta["crpix1"])
    range_side = (center + np.array([-1, 1]) * smap.data.shape[0] / 2) * u.pix
    newmap = tempmap.submap(
        u.Quantity([range_side[0], range_side[0]]),
        top_right=u.Quantity([range_side[1], range_side[1]]) - 1 * u.pix,
    )

    # Update intermediate metadata (overwritten below; kept for reference)
    newmap.meta["r_sun"] = newmap.meta["rsun_obs"] / newmap.meta["cdelt1"]
    newmap.meta["lvl_num"] = 1.5
    newmap.meta["bitpix"] = -64

    # Reconstruct map from original data to handle any size mismatch
    newmap = sunpy.map.Map(_extract_or_pad_data(smap.data, orig_shape), smap.meta)

    # Set reference pixel to image centre
    newmap.meta["crpix1"] = orig_shape / 2 + 0.5
    newmap.meta["crpix2"] = orig_shape / 2 + 0.5

    return newmap


def compute_differential_rotation(
    smap: sunpy.map.Map,
    target_date: datetime,
) -> sunpy.map.Map:
    """
    Reproject a map to account for solar differential rotation.

    Solar features rotate at latitude-dependent rates. This function
    shifts an image to the position features would occupy at
    ``target_date``, allowing a nearby-time image to substitute for
    a missing observation.

    Parameters
    ----------
    smap : sunpy.map.Map
        Input map to rotate.
    target_date : datetime
        Target observation time to rotate to.

    Returns
    -------
    sunpy.map.Map
        Reprojected map with the original metadata preserved.

    Notes
    -----
    Uses SunPy's ``propagate_with_solar_surface`` context manager for
    accurate differential rotation modelling.
    """
    # Output frame: same observer position, shifted to target_date
    out_frame = Helioprojective(
        observer=smap.observer_coordinate,
        obstime=Time(target_date),
        rsun=smap.coordinate_frame.rsun,
    )
    out_center = SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=out_frame)

    # Construct output WCS matching the input image geometry
    header = sunpy.map.make_fitswcs_header(
        smap.data.shape,
        out_center,
        reference_pixel=u.Quantity(smap.reference_pixel),
        scale=u.Quantity(smap.scale),
        rotation_matrix=smap.rotation_matrix,
        instrument=smap.instrument,
        exposure=smap.exposure_time,
    )
    out_wcs = WCS(header)

    # Reproject pixels using differential rotation to propagate coordinates
    with propagate_with_solar_surface():
        smap_reprojected = smap.reproject_to(out_wcs)

    # Return reprojected data with original metadata (observer, instrument, etc.)
    return sunpy.map.Map(smap_reprojected.data, smap.meta)
