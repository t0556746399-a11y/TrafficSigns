import pandas as pd
import numpy as np

# הגדרות תצוגה
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


def _ensure_group_key(df, trip_col, temp_col='_single_trip_key'):
    """
    Safe-get מרכזי לעמודת הקיבוץ (trip_col).
    לא מניח שקיימת trip_col בכל שלב/תת-קבוצה. בודק במפורש trip_col is None,
    שהיא מחרוזת, ושהיא באמת קיימת בעמודות ה-DataFrame הנוכחי.
    אם לא - יוצרים עמודת קיבוץ זמנית קבועה (כל השורות = "נסיעה" אחת).
    מחזיר: (df [אולי עם עמודה זמנית], שם עמודת הקיבוץ בפועל, is_temp)
    """
    has_valid_trip_col = isinstance(trip_col, str) and trip_col in df.columns
    if has_valid_trip_col:
        return df, trip_col, False
    df = df.copy()
    df[temp_col] = 0
    return df, temp_col, True


def calculate_distance_and_speed(df):
    df = df.sort_values(['timestamp']).reset_index(drop=True)
    lat1 = np.radians(df['latitude'].shift(1))
    lon1 = np.radians(df['longitude'].shift(1))
    lat2 = np.radians(df['latitude'])
    lon2 = np.radians(df['longitude'])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = 6371 * 1000 * c

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    time_diff = (df['timestamp'] - df['timestamp'].shift(1)).dt.total_seconds()

    speed = distance / time_diff.replace(0, np.nan)
    speed = speed.fillna(0)

    df['speed_mps'] = speed
    df['speed_kmh'] = speed * 3.6
    df['distance_meters'] = distance.fillna(0)
    return df


def clean_gps_data(df, trip_col=None, time_col='timestamp', lat_col='latitude', lon_col='longitude',
                   speed_col='speed_kmh', max_physical_speed_kmh=140.0, max_gap_seconds=5.0, max_interpolation_gap=5):
    df_clean, group_key, is_temp_key = _ensure_group_key(df.copy(), trip_col)

    df_clean = df_clean.sort_values([group_key, time_col]).reset_index(drop=True)
    grouped = df_clean.groupby(group_key, group_keys=False)

    df_clean['time_gap_seconds'] = grouped[time_col].diff().dt.total_seconds()
    df_clean['is_dropout'] = df_clean['time_gap_seconds'] > max_gap_seconds

    df_clean['is_spike_point'] = (df_clean[speed_col] > max_physical_speed_kmh) & \
                                 (grouped[speed_col].shift(-1) > max_physical_speed_kmh).fillna(False)

    points_to_null = (df_clean['is_spike_point'] | df_clean[lat_col].isna()) & (~df_clean['is_dropout'])
    df_clean.loc[points_to_null, [lat_col, lon_col]] = np.nan

    def _interpolate_group(g):
        g = g.set_index(time_col)
        g[[lat_col, lon_col]] = g[[lat_col, lon_col]].interpolate(method='time', limit=max_interpolation_gap,
                                                                  limit_area='inside')
        return g.reset_index()

    # pandas גרסאות חדשות מוציאות בשקט את עמודת הקיבוץ מתוצאת groupby(...).apply(...)
    # - שומרים את הערכים לפני ומחברים בחזרה אחרי (בטוח פוזיציונית כי ה-df ממוין
    # לפי [group_key, time_col] וסדר השורות נשמר).
    group_key_values = df_clean[group_key].to_numpy()
    df_clean = grouped.apply(_interpolate_group).reset_index(drop=True)
    df_clean[group_key] = group_key_values
    df_clean['is_unrecoverable'] = df_clean[lat_col].isna() | df_clean[lon_col].isna()
    if is_temp_key:
        df_clean = df_clean.drop(columns=[group_key], errors='ignore')
    return df_clean


def handle_gps_stops(df, lat_col='latitude', lon_col='longitude', speed_col='speed_kmh', stop_speed_threshold=1.0,
                     trip_col='drive_uuid'):
    df, group_col, is_temp_key = _ensure_group_key(df.copy(), trip_col)

    # is_stopped נבדק גם "אחורה" (המהירות לתוך השורה) וגם "קדימה" (המהירות
    # מהשורה הזו לבאה אחריה). זה נחוץ כי שורת הכניסה לאשכול עצירה (למשל
    # כשיש קפיצה/רעש חד-צדדי בכניסה) יכולה להראות מהירות "אחורה" גבוהה
    # ומטעה, אף שבפועל התזוזה *קדימה* שלה מראה שהיא כבר בעצירה.
    is_stopped_backward = df[speed_col] < stop_speed_threshold
    is_stopped_forward = df[speed_col].shift(-1) < stop_speed_threshold
    is_stopped = is_stopped_backward | is_stopped_forward.fillna(False)

    trip_id = df[group_col]
    boundary = (is_stopped != is_stopped.shift()) | (trip_id != trip_id.shift())
    run_id = boundary.cumsum()

    grp = df.groupby(run_id)
    df.loc[is_stopped, lat_col] = grp[lat_col].transform('mean')[is_stopped]
    df.loc[is_stopped, lon_col] = grp[lon_col].transform('mean')[is_stopped]
    df.loc[is_stopped, speed_col] = 0.0
    if is_temp_key:
        df = df.drop(columns=[group_col], errors='ignore')
    return df


def _calc_single_trip_azimuth(df_trip, speed_stop_threshold=0.1):
    df = df_trip.copy().reset_index(drop=True)
    df['next_lat'] = df['latitude'].shift(-1)
    df['next_lon'] = df['longitude'].shift(-1)

    lat1, lon1 = np.radians(df['latitude']), np.radians(df['longitude'])
    lat2, lon2 = np.radians(df['next_lat']), np.radians(df['next_lon'])
    d_lon = lon2 - lon1

    x = np.sin(d_lon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(d_lon)

    azimuth_deg = np.degrees(np.arctan2(x, y))
    df['azimuth'] = (azimuth_deg + 360) % 360

    # azimuth[i] מתאר את הכיוון *מ*-i *אל* i+1 (מחושב מ-next_lat/next_lon).
    # לכן יש לבטל אותו לפי המהירות *קדימה* (shift(-1)) ולא לפי speed_kmh[i]
    # שמתארת את התנועה *לתוך* i (מ-i-1). שימוש ב-speed_kmh[i] גרם לכך שהשורה
    # הראשונה של כל נסיעה (שתמיד speed_kmh=0 בה, כי אין נקודה קודמת) תמיד
    # תאבד את האזימוט שלה - גם כשהרכב באמת נע לשורה הבאה.
    forward_speed = df['speed_kmh'].shift(-1)
    df.loc[forward_speed < speed_stop_threshold, 'azimuth'] = np.nan
    df['azimuth'] = df['azimuth'].ffill().bfill()
    # אם נסיעה שלמה נייחת לחלוטין - אין אף שורה עם כיוון תקף למלא ממנה;
    # ברירת מחדל סבירה היא 0 מעלות במקום להשאיר NaN.
    df['azimuth'] = df['azimuth'].fillna(0.0)
    return df.drop(columns=['next_lat', 'next_lon'], errors='ignore')


def process_full_pipeline(df, trip_col='drive_uuid'):
    print("--- מתחיל עיבוד וניתוח נתונים מלא ---")
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    def has_trip_col(current_df):
        return isinstance(trip_col, str) and trip_col in current_df.columns

    df = calculate_distance_and_speed(df)
    df = clean_gps_data(df, trip_col=trip_col if has_trip_col(df) else None)

    unusable_mask = df['is_unrecoverable'] | (df['is_dropout'] & df['time_gap_seconds'].gt(5))
    df = df.loc[~unusable_mask].reset_index(drop=True)

    df = df.drop(columns=['speed_mps', 'speed_kmh', 'distance_meters'], errors='ignore')
    df = calculate_distance_and_speed(df)

    df = handle_gps_stops(df, trip_col=trip_col if has_trip_col(df) else None)

    df = calculate_distance_and_speed(df)

    if has_trip_col(df):
        df = df.sort_values([trip_col, 'timestamp']).reset_index(drop=True)
        trip_col_values = df[trip_col].to_numpy()
        df = df.groupby(trip_col, group_keys=False).apply(_calc_single_trip_azimuth).reset_index(drop=True)
        df[trip_col] = trip_col_values
    else:
        df = _calc_single_trip_azimuth(df)

    diff = df['azimuth'].diff()
    df['angular_change'] = (diff + 180) % 360 - 180

    conditions = [(df['speed_kmh'] < 1.0), (np.abs(df['angular_change']) > 15.0), (np.abs(df['angular_change']) > 2.0)]
    df['behavior'] = np.select(conditions, ['עצירה', 'פנייה חדה', 'פנייה מתונה'], default='נסיעה ישר')

    print("--- העיבוד הושלם בהצלחה ---")
    return df