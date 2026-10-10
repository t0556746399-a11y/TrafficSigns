import pandas as pd
import sys
import os
from geometry_processor import calc_absolute_azimuth

# הוספת התיקייה ל-path כדי שפייתון ימצא את הקבצים
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from processor import process_full_pipeline
from merger import merge_data


def run():
    try:
        print("מתחיל עיבוד נתונים...")
        # טעינת הקובץ מהנתיב המלא
        raw_gps = pd.read_csv('data/clean_trip_data.csv')

        # הרצת ה-Pipeline
        enriched_gps = process_full_pipeline(raw_gps)

        enriched_gps.to_csv('data/enriched_trip_data.csv', index=False)

        results = pd.read_csv('data/results.csv')
        final_df = merge_data(enriched_gps, results)
        # מדפיס את השורות שבהן המערכת זיהתה עצירה
        print(final_df[final_df['behavior'] == 'עצירה'][['timestamp', 'speed_kmh', 'behavior']])
        final_df.to_csv('data/final_output.csv', index=False)
        file_path = os.path.join("data", "final_output.csv") # או .parquet בהתאם לפורמט הקובץ
        
        # טעינת קובץ הנתונים ל-DataFrame
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"הקובץ לא נמצא בנתיב: {file_path}")
        
        df_detections = pd.read_csv(file_path)
        
        # הגדרת שדה ה-HFOV של המצלמה (לדוגמה: 90 מעלות)
        hfov_value = 90.0 
        
        # הפעלת הפונקציה
        df_result = calc_absolute_azimuth(df_detections, HFOV=hfov_value)
        
        # שמירת התוצאה המעודכנת (אופציונלי - לדוגמה חזרה לאותה התיקייה או לקובץ חדש)
        output_path = os.path.join("data", "final_output.csv")
        df_result.to_csv(output_path, index=False)
        
        print(f"看到了 עיבוד הושלם. נוספו העמודות לטבלה ושובצו בנתיב: {output_path}")
        print("המערכת סיימה בהצלחה!")

    except Exception as e:
        print(f"שגיאה בתהליך: {e}")


if __name__ == '__main__':
    run()