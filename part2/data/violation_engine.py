import pandas as pd
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from spatial_analyzer import build_sign_index, find_nearby_signs
import os

# 1. הגדרת סוגי התנהגות ותמרורים
BEHAVIOR_RULES = {
    'Speed_Limit': 'speed_check',
    'Stop': 'stop_check',
    'Yield': 'slow_check',
    'Crosswalk': 'slow_check',
    'No_Entry': 'direction_check',
    'No_U_Turn': 'direction_check'
}


def evaluate_sign(row, df, idx):
    sign = str(row.get('sign_type', 'Unknown'))
    speed = row.get('speed_kmh', 0)
    behavior = next((v for k, v in BEHAVIOR_RULES.items() if k in sign), 'unknown')

    if behavior == 'speed_check':
        limit = int(''.join(filter(str.isdigit, sign)) or 0)
        return "Speeding" if speed > limit else "Normal"
    elif behavior == 'stop_check':
        return "Normal" if speed == 0 else "No Stop"
    elif behavior == 'slow_check':
        return "Normal" if speed < 20 else "No Deceleration"
    elif behavior == 'direction_check':
        change = abs(row.get('angular_change', 0))
        return "Direction Violation" if change < 10 else "Normal"
    return "Normal"


def generate_pdf_report(violations, summary_stats, filename="safety_report.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", 'B', 18)
    pdf.cell(200, 15, text="Safety Violation Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')

    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(200, 10, text="Executive Summary:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    if summary_stats:
        for violation_type, count in summary_stats.items():
            pdf.cell(200, 8, text=f"- {violation_type}: {count} occurrences", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.ln(10)
    pdf.cell(45, 8, text="Timestamp", border=1)
    pdf.cell(55, 8, text="Detected Sign", border=1)
    pdf.cell(45, 8, text="Vehicle Speed", border=1)
    pdf.cell(45, 8, text="Status", border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_font("Helvetica", size=9)
    for _, row in violations.iterrows():
        pdf.cell(45, 8, text=str(row.get('timestamp', 'N/A')), border=1)
        pdf.cell(55, 8, text=str(row.get('sign_type', 'N/A')), border=1)
        pdf.cell(45, 8, text=f"{row.get('speed_kmh', 0):.1f} km/h", border=1)
        pdf.cell(45, 8, text=str(row.get('status', 'N/A')), border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.output(filename)


def run_pipeline(input_csv, signs_csv):
    try:
        print("--- מתחיל ניתוח עבירות עם ולידציה מרחבית ---")

        df_signs = pd.read_csv(signs_csv)
        tree = build_sign_index(df_signs)
        df = pd.read_csv(input_csv)

        # ניקוי הנתונים והמרת עמודות לפורמט מספרי
        cols_to_numeric = ['speed_kmh', 'latitude', 'longitude']
        for col in cols_to_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        df['status'] = "Normal"

        for idx, row in df.iterrows():
            # ולידציה מרחבית באמצעות העמודות הנכונות
            nearby = find_nearby_signs(tree, df_signs, row['latitude'], row['longitude'], radius=10.0)

            if not nearby.empty:
                df.at[idx, 'status'] = evaluate_sign(row, df, idx)

        violations = df[df['status'] != "Normal"].copy()
        summary_stats = violations['status'].value_counts().to_dict()

        violations.to_csv("violations_report.csv", index=False)
        generate_pdf_report(violations, summary_stats)

        print(f"הניתוח הסתיים! נמצאו {len(violations)} עבירות.")

    except Exception as e:
        print(f"שגיאה בהרצת ה-Pipeline: {e}")


if __name__ == "__main__":
    run_pipeline('data/final_output.csv', 'data/final_sign_locations.csv')