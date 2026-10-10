import numpy as np
import pandas as pd


def _calc_single_trip_azimuth(
        df_trip,
        speed_stop_threshold=3.0,
        speed_reverse_threshold=25.0,
        angle_flip_threshold=160.0
):
    """
    מחשבת אזימוט גיאוגרפי מתוקן עבור נסיעה בודדת ומבודדת.

    --- לוגיקה ושלבי עבודה במילים ---
    1. חישוב וקטורי של אזימוט גולמי (Raw Azimuth) בין כל נקודה לבאה בתור
       באמצעות נוסחה גיאודזית (arctan2) ונרמול לטווח של 0-360 מעלות.
    2. טיפול במקרה קצה 1 (אפס תנועה): מחיקת אזימוט במקומות שבהם ה-GPS מראה
       שנשארנו בדיוק באותן קואורדינטות, כדי למנוע עיוותים מתמטיים.
    3. טיפול במקרה קצה 2 (נסיעה לאחור / רוורס):
       - שימוש בלוגיקה היברידית העוקבת אחר שינוי הכיוון משנייה לשנייה (shift).
       - זיהוי קפיצה חדה ופתאומית בזווית (מעל 160 מעלות בשנייה בודדת) המעידה על שילוב רוורס.
       - ניהול מונה מהפכים (cumsum). מספר אי-זוגי של קפיצות חדות מסמן שהרכב ברוורס ומנקה את הזווית.
       - לוגיקה זו חסינה מפני פניות פרסה איטיות, שכן בהן שינוי הזווית הוא הדרגתי ולא פתאומי.
    4. טיפול במקרה קצה 3 (עצירה / GPS Drift): מחיקת זוויות המצפן בכל זמן
       שהמהירות נמוכה מ-3 קמ"ש, שכן רכיבי GPS נוטים "לרקוד" ולזייף כיוון בעמידה.
    5. החלקה והשלמה (Imputation): ביצוע ffill ו-bfill על ה-NaN שנוצרו.
       המשמעות: בזמן עצירה או רוורס, כיוון הרכב "קופא" על האזימוט התקין האחרון שלו.

    --- פרמטרי קלט בדאטה (Columns) ---
    :param df_trip: DataFrame המכיל את העמודות הבאות:
                    - 'latitude', 'longitude': קואורדינטות גיאוגרפיות (חובה).
                    - 'speed_kmh': מהירות הרכב בקמ"ש, מחושבת מראש ע"י פונקציה חיצונית (חובה).

    --- ערכי סף והגדרות אלגוריתמיות (Thresholds) ---
    :param speed_stop_threshold: מהירות (בקמ"ש) שמתחתיה הרכב נחשב בעצירה מוחלטת. ברירת מחדל: 3.0.
    :param speed_reverse_threshold: מהירות מקסימלית הגיונית לרוורס (למניעת זיהוי שגוי במהירות גבוהה). ברירת מחדל: 25.0.
    :param angle_flip_threshold: זווית (במעלות) המגדירה קפיצה חדה בשנייה אחת כאירוע היפוך כיוון. ברירת מחדל: 160.0.

    :return: DataFrame עם עמודת 'azimuth' סופית, נקייה ומיושרת.
    """
    df = df_trip.copy()

    # יצירת קואורדינטות יעד (הנקודה הבאה בתור על ציר הזמן)
    df['next_lat'] = df['latitude'].shift(-1)
    df['next_lon'] = df['longitude'].shift(-1)

    # המרה לרדיאנים לצורך החישוב הטריגונומטרי
    lat1 = np.radians(df['latitude'])
    lon1 = np.radians(df['longitude'])
    lat2 = np.radians(df['next_lat'])
    lon2 = np.radians(df['next_lon'])

    # חישוב רכיבי הנוסחה הגיאודזית (Great-Circle Bearing)
    d_lon = lon2 - lon1
    x = np.sin(d_lon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(d_lon)

    # חילוץ הזווית, המרה חזרה למעלות ונרמול לטווח מצפן סטנדרטי (0-360)
    azimuth_deg = np.degrees(np.arctan2(x, y))
    df['raw_azimuth'] = (azimuth_deg + 360) % 360
    df['azimuth'] = df['raw_azimuth'].copy()

    # ---------------------------------------------------------------------
    # מקרה קצה 1: אפס תנועה מוחלט (מונע שגיאות מתמטיות של arctan2 על אפס)
    # ---------------------------------------------------------------------
    same_position_mask = np.isclose(df['latitude'], df['next_lat'], atol=1e-9) & \
                         np.isclose(df['longitude'], df['next_lon'], atol=1e-9)
    df.loc[same_position_mask, 'azimuth'] = np.nan

    # ---------------------------------------------------------------------
    # מקרה קצה 2: נסיעה לאחור (רוורס) - לוגיקת הפרשים חסינת פניות פרסה
    # ---------------------------------------------------------------------
    # א. מייצרים ציר אזימוטים רציף שמתעלם מרגעי עצירה (מונע באגים של NaN משתרשר)
    last_moving_azimuth = df['raw_azimuth'].where(df['speed_kmh'] >= speed_stop_threshold).ffill()
    prev_moving_azimuth = last_moving_azimuth.shift(1)

    # ב. חישוב ההפרש הזוויתי המדויק משנייה לשנייה (בטווח של 180- עד 180 מעלות)
    angular_diff = (last_moving_azimuth - prev_moving_azimuth + 180) % 360 - 180

    # ג. זיהוי רגע המהפך: קפיצה חדה ופתאומית (כמעט 180 מעלות בשנייה אחת) במהירות נמוכה.
    # פניות פרסה (אפילו איטיות מאוד) קורות בהדרגה לאורך כמה שניות ולכן לא יפעילו את זה.
    flip_moments = (np.abs(angular_diff) > angle_flip_threshold) & (df['speed_kmh'] < speed_reverse_threshold)

    # ד. ניהול סטטוס הרוורס: סופרים כמה מהפכים היו. מספר אי-זוגי = הרכב נוסע כרגע לאחור.
    df['flip_count'] = flip_moments.cumsum()
    reverse_mask = (df['flip_count'] % 2 == 1) & (df['speed_kmh'] < speed_reverse_threshold)

    # ה. מחיקת האזימוט השגוי במקטעי הרוורס (הפיכה ל-NaN)
    df.loc[reverse_mask, 'azimuth'] = np.nan

    # ---------------------------------------------------------------------
    # מקרה קצה 3: רמזור / עצירה (GPS Drift)
    # ---------------------------------------------------------------------
    # מבוצע בסוף כדי לא לפגוע במטריצות החישוב הקודמות שדרשו אזימוט תנועה רציף
    df.loc[df['speed_kmh'] < speed_stop_threshold, 'azimuth'] = np.nan

    # ---------------------------------------------------------------------
    # שלב ההחלקה והשלמת הנתונים הריקים (Imputation)
    # ---------------------------------------------------------------------
    # ffill - נועל ומעתיק קדימה את האזימוט התקין האחרון לאורך כל העצירות והרוורסים
    df['azimuth'] = df['azimuth'].ffill()
    # bfill - למקרה קצה קיצוני שהקובץ/הנסיעה נפתחים ישר בתוך מצב של עמידה
    df['azimuth'] = df['azimuth'].bfill()

    # הסרת עמודות עזר מתמטיות על מנת להחזיר דאטהסט נקי למשתמש
    # הסרת עמודות עזר מתמטיות על מנת להחזיר דאטהסט נקי למשתמש (משאירים את raw_azimuth)
    df.drop(columns=['next_lat', 'next_lon', 'flip_count'], inplace=True, errors='ignore')
    return df


def calc_azimuth(df):
    """
    הפונקציה העטיפה הראשית (Wrapper).
    מוודאת סדר כרונולוגי ומבודדת את החישוב לכל נסיעה בנפרד (לפי drive_uuid)
    על מנת למנוע זליגת נתונים ושגיאות חישוב בגבולות שבין נסיעות שונות.
    """
    # מבטיח סדר כרונולוגי ומונע בעיות אינדקסים
    df = df.sort_values('timestamp').reset_index(drop=True).copy()

    # אם הדאטה מכיל יותר מנסיעה אחת, נחשב עצמאית לכל קבוצה
    if 'drive_uuid' in df.columns and df['drive_uuid'].nunique() > 1:
        return df.groupby('drive_uuid', group_keys=False).apply(_calc_single_trip_azimuth)
    else:
        return _calc_single_trip_azimuth(df)