import pandas as pd
import numpy as np
import os


def create_mock_data():
    os.makedirs('data', exist_ok=True)

    # 1. יצירת נתוני GPS (תואם למבנה ה-safety_events)
    periods = 100
    start_time = pd.Timestamp('2026-07-07 15:00:00')
    time_range = pd.date_range(start=start_time, periods=periods, freq='s')

    # יצירת מסלול עם "רעש" מכוון
    data = {
        'timestamp': time_range,
        'drive_uuid': ['test-drive-001'] * periods,
        'latitude': [31.7683 + (i * 0.0001) for i in range(periods)],
        'longitude': [35.2137 + (i * 0.0001) for i in range(periods)],
        'speed_kmh': [50.0] * periods  # מהירות בסיס
    }

    df_gps = pd.DataFrame(data)

    # הזרקת מקרי קצה לבדיקת הניקוי:
    df_gps.loc[20, 'speed_kmh'] = 300.0  # ספייק מהירות (צריך לזהות כ-True)
    df_gps.loc[50:60, 'speed_kmh'] = 0.0  # עצירה (צריך לזהות כ-עצירה)

    # 2. יצירת נתוני תמרורים (תואם למבנה ה-results)
    # יצירת רשימת תמרורים לדוגמה
    signs = [
        {'timestamp': start_time + pd.Timedelta(seconds=5), 'class_name': 'Speed_Limit_60', 'id': 2},
        {'timestamp': start_time + pd.Timedelta(seconds=55), 'class_name': 'Stop', 'id': 3}
    ]
    df_signs = pd.DataFrame(signs)

    # שמירה
    df_gps.to_csv('data/clean_trip_data.csv', index=False)
    df_signs.to_csv('data/results.csv', index=False)

    print("נתוני דמה נוצרו בהצלחה בתיקיית data!")
    print("מוכנים להרצת ה-Pipeline!")


if __name__ == '__main__':
    create_mock_data()