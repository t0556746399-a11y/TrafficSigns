import warnings

import numpy as np
import pandas as pd


def _resolve_input_column(df: pd.DataFrame, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def calc_absolute_azimuth(df_detections: pd.DataFrame, HFOV: float) -> pd.DataFrame:
    """
    Calculate relative and absolute azimuth from bounding boxes and vehicle heading.

    This version is schema-tolerant and works with both the older column names
    and the current detection file layout.
    """
    if not isinstance(df_detections, pd.DataFrame):
        raise TypeError("df_detections must be a pandas DataFrame")

    result = df_detections.copy()
    result["relative_angle"] = np.nan
    result["absolute_azimuth"] = np.nan

    if not np.isscalar(HFOV) or pd.isna(HFOV):
        return result

    hfov_value = float(HFOV)
    if hfov_value <= 0:
        return result

    x1_col = _resolve_input_column(result, ["bbox_x1", "bbox_left", "x1"])
    x2_col = _resolve_input_column(result, ["bbox_x2", "bbox_right", "x2"])
    width_col = _resolve_input_column(result, ["image_width", "imageWidth", "width", "frame_width"])
    azimuth_col = _resolve_input_column(result, ["vehicle_azimuth", "vehicle_heading", "heading", "azimuth"])

    if x1_col is None or x2_col is None or width_col is None or azimuth_col is None:
        return result

    x1 = pd.to_numeric(result[x1_col], errors="coerce").to_numpy(dtype=float)
    x2 = pd.to_numeric(result[x2_col], errors="coerce").to_numpy(dtype=float)
    image_width = pd.to_numeric(result[width_col], errors="coerce").to_numpy(dtype=float)
    vehicle_azimuth = pd.to_numeric(result[azimuth_col], errors="coerce").to_numpy(dtype=float)

    valid_mask = (
        np.isfinite(x1)
        & np.isfinite(x2)
        & np.isfinite(image_width)
        & np.isfinite(vehicle_azimuth)
        & (image_width > 0)
    )

    if not np.any(valid_mask):
        return result

    x1_valid = np.clip(x1[valid_mask], 0.0, image_width[valid_mask])
    x2_valid = np.clip(x2[valid_mask], 0.0, image_width[valid_mask])
    image_width_valid = image_width[valid_mask]
    vehicle_azimuth_valid = vehicle_azimuth[valid_mask]

    zero_width_mask = (x1_valid == x2_valid)
    if np.any(zero_width_mask):
        warnings.warn("Zero-width bounding box encountered; treating it as a point center.", RuntimeWarning)

    bbox_center_x = (x1_valid + x2_valid) / 2.0
    image_center_x = image_width_valid / 2.0

    hfov_rad = np.deg2rad(hfov_value)
    focal_length = image_center_x / np.tan(hfov_rad / 2.0)

    relative_rad = np.arctan((bbox_center_x - image_center_x) / focal_length)
    relative_deg = np.rad2deg(relative_rad)
    absolute_deg = np.mod(vehicle_azimuth_valid + relative_deg, 360.0)

    result.loc[valid_mask, "relative_angle"] = relative_deg
    result.loc[valid_mask, "absolute_azimuth"] = absolute_deg

    return result