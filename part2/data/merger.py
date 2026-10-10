import pandas as pd
import numpy as np

REQUIRED_DETECTION_COLUMNS = [
    "frame_timestamp",
    "sign_instance_id",
    "sign_type",
    "confidence",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "bbox_area",
    "image_width",
    "image_height",
]


def _ensure_required_columns(df, required_columns, source_name):
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise KeyError(f"{source_name} is missing required columns: {missing}")


def load_detection_results(detections_df):
    """Normalize detection rows and remove duplicates per instance ID."""
    if not isinstance(detections_df, pd.DataFrame):
        raise TypeError("detections_df must be a pandas DataFrame")

    _ensure_required_columns(detections_df, REQUIRED_DETECTION_COLUMNS, "detections_df")

    normalized = detections_df.copy()
    normalized["frame_timestamp"] = pd.to_datetime(normalized["frame_timestamp"], errors="coerce")

    for col in ["confidence", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "bbox_area"]:
        normalized[col] = pd.to_numeric(normalized[col], errors="coerce")

    for col in ["image_width", "image_height"]:
        normalized[col] = pd.to_numeric(normalized[col], errors="coerce")

    # הסרת כפילויות של אותו תמרור (למנוע ניפוח נתונים)
    return normalized.drop_duplicates(subset=['sign_instance_id'])


def merge_data(
        gps_df,
        detections_df,
        time_col="timestamp",
        detection_time_col="frame_timestamp",
        tolerance="5s",
        distance_threshold=20.0
):
    """
    Merge telemetry with detection rows using time window AND geographical distance.
    """
    if not isinstance(gps_df, pd.DataFrame):
        raise TypeError("gps_df must be a pandas DataFrame")

    gps = gps_df.copy()
    gps[time_col] = pd.to_datetime(gps[time_col], errors="coerce")

    # מיון להבטחת תקינות
    gps = gps.sort_values(time_col).reset_index(drop=True)

    detections = load_detection_results(detections_df)

    if "drive_uuid" not in gps.columns:
        gps["drive_uuid"] = ""
    gps["drive_uuid"] = gps["drive_uuid"].astype(str)

    output_columns = list(gps.columns) + [col for col in REQUIRED_DETECTION_COLUMNS if col != detection_time_col] + [
        detection_time_col]
    rows = []

    for _, gps_row in gps.iterrows():
        gps_time = gps_row[time_col]

        # סינון לפי זמן
        time_mask = (abs(detections[detection_time_col] - gps_time) <= pd.Timedelta(tolerance))
        potential_matches = detections[time_mask].copy()

        # סינון גיאוגרפי אם קיימות קואורדינטות מתאימות
        if not potential_matches.empty and 'final_x' in potential_matches.columns and 'cam_x' in gps_row:
            potential_matches['dist'] = np.sqrt(
                (potential_matches['final_x'] - gps_row['cam_x']) ** 2 +
                (potential_matches['final_y'] - gps_row['cam_y']) ** 2
            )
            potential_matches = potential_matches[potential_matches['dist'] <= distance_threshold]

        if not potential_matches.empty:
            for _, det_row in potential_matches.iterrows():
                row = gps_row.to_dict()
                for col in REQUIRED_DETECTION_COLUMNS:
                    row[col] = det_row[col]
                rows.append(row)
        else:
            # הוספת שורת GPS ללא זיהוי
            row = gps_row.to_dict()
            for col in REQUIRED_DETECTION_COLUMNS:
                row[col] = pd.NA
            rows.append(row)

    return pd.DataFrame(rows, columns=output_columns)