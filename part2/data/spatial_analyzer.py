import pandas as pd
import numpy as np
from pyproj import Transformer
from sklearn.neighbors import KDTree



# --- הגדרות קבועות ---
EPSG_WGS84 = "EPSG:4326"
EPSG_ITM = "EPSG:2039"

def build_sign_index(df_signs):
    """
    בונה אינדקס מהיר לחיפוש תמרורים.
    df_signs חייב להכיל את העמודות final_x, final_y.
    """
    coords = df_signs[['final_x', 'final_y']].values
    tree = KDTree(coords)
    return tree

def find_nearby_signs(tree, df_signs, car_x, car_y, radius=30.0):
    """
    מוצאת אילו תמרורים נמצאים ברדיוס של X מטרים מהרכב.
    """
    # חיפוש תמרורים בטווח הרדיוס
    indices = tree.query_radius([[car_x, car_y]], r=radius)

    # מחזירה את השורות מה-DataFrame של התמרורים שנמצאו
    return df_signs.iloc[indices[0]]
def get_transformer():
    """יוצר אובייקט טרנספורמציה למערכת הקואורדינטות של ישראל."""
    return Transformer.from_crs(EPSG_WGS84, EPSG_ITM, always_xy=True)


def project_and_model(df):
    """
    שלב ב': המרת GPS למטרים (ITM) ויצירת וקטורי כיוון.
    """
    transformer = get_transformer()
    x, y = transformer.transform(df['longitude'].values, df['latitude'].values)

    # שימוש ב-assign לעבודה נקייה על DataFrame
    df = df.assign(cam_x=x, cam_y=y)

    az_rad = np.radians(df['azimuth'])
    df = df.assign(ray_dx=np.sin(az_rad), ray_dy=np.cos(az_rad))
    return df


def triangulate_signs(df):
    """
    שלב ג': חישוב מיקום התמרור באמצעות Least Squares עם סינון מרחבי.
    """
    # הגדרת גבולות גיאוגרפיים גסים לירושלים (למשל, בערכי ITM)
    # ערכים אלו תלויים במיקום הנסיעה שלך
    MIN_X, MAX_X = 170000, 250000
    MIN_Y, MAX_Y = 600000, 680000

    def _solve_intersection(g):
        if len(g) < 2:
            return pd.Series({'final_x': np.nan, 'final_y': np.nan})

        A = np.column_stack((g['ray_dx'], -g['ray_dy']))
        b = (g['ray_dx'] * g['cam_x'] - g['ray_dy'] * g['cam_y']).values
        point, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        # סינון: אם הנקודה מחוץ לגבולות שהגדרנו, נחזיר NaN
        if not (MIN_X <= point[0] <= MAX_X and MIN_Y <= point[1] <= MAX_Y):
            return pd.Series({'final_x': np.nan, 'final_y': np.nan})

        return pd.Series({'final_x': point[0], 'final_y': point[1]})

    results = df.groupby('sign_instance_id', group_keys=False).apply(_solve_intersection)
    # הסרת ערכים לא תקינים סופית
    return results.dropna()
def run_spatial_pipeline(input_path, output_path):
    """פונקציית הרצה מרכזית לכל שלבי העיבוד המרחבי."""
    try:
        print("--- מתחיל עיבוד מרחבי (שלב ב' ו-ג') ---")
        df = pd.read_csv(input_path)

        # ביצוע המרה ומודלינג
        df = project_and_model(df)

        # ביצוע טריאנגולציה
        df_locations = triangulate_signs(df)

        # שמירה
        df_locations.to_csv(output_path, index=False)
        print(f"העיבוד הסתיים בהצלחה. הקובץ נשמר ב: {output_path}")

    except Exception as e:
        print(f"שגיאה במהלך העיבוד המרחבי: {e}")


if __name__ == "__main__":
    run_spatial_pipeline('data/final_output.csv', 'data/final_sign_locations.csv')