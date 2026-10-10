import pandas as pd
import numpy as np


class DetectionLogger:
    """
    אוספת רשומת-זיהוי מספרית מכל פריים (לא קרופים!), ברמת bbox גולמי,
    כדי שאפשר יהיה בהמשך למזג (merge_asof) מול טבלת הטלמטריה ולחשב זווית.

    tracker_id (מ-ByteTrack) הוא בדיוק ה-sign_instance_id שדיברנו עליו -
    המזהה שמקשר בין כמה פריימים לאותו תמרור פיזי.
    """

    def __init__(self):
        self._records = []

    def log_frame(self, frame_timestamp, detections, image_width, image_height,
                  drive_uuid=None):
        """
        קוראים לזה פעם אחת לכל פריים, מיד אחרי YOLOEngine.predict(),
        עם ה-detections שחזרו (אחרי טרקינג).

        frame_timestamp: זמן הפריים המדויק (datetime, כולל מילישניות אם אפשר)
                          - זה מה שהיה חסר לגמרי ב-predict() עצמה.
        """
        if detections is None or len(detections) == 0:
            return

        for i in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[i]
            self._records.append({
                'drive_uuid': drive_uuid,
                'frame_timestamp': frame_timestamp,
                'sign_instance_id': int(detections.tracker_id[i]),
                'sign_type': detections.data['class_name'][i],
                'confidence': float(detections.confidence[i]),
                'bbox_x1': float(x1), 'bbox_y1': float(y1),
                'bbox_x2': float(x2), 'bbox_y2': float(y2),
                'bbox_area': float((x2 - x1) * (y2 - y1)),
                'image_width': image_width,
                'image_height': image_height,
            })

    def log_detection(self, frame_timestamp, tracker_id, class_name, confidence,
                       x1, y1, x2, y2, image_width, image_height, drive_uuid=None):
        """
        רישום זיהוי בודד - מתאים לזרימה שבה מעבדים detection-אחר-detection
        בתוך לולאה (כמו ב-main.py), לא batch שלם של detections בבת אחת.
        """
        self._records.append({
            'drive_uuid': drive_uuid,
            'frame_timestamp': frame_timestamp,
            'sign_instance_id': int(tracker_id),
            'sign_type': class_name,
            'confidence': float(confidence),
            'bbox_x1': float(x1), 'bbox_y1': float(y1),
            'bbox_x2': float(x2), 'bbox_y2': float(y2),
            'bbox_area': float((x2 - x1) * (y2 - y1)),
            'image_width': image_width,
            'image_height': image_height,
        })

    def to_dataframe(self):
        return pd.DataFrame(self._records)

    @classmethod
    def from_dataframe(cls, df):
        """בונה DetectionLogger מ-DataFrame קיים (למשל אחרי סינון חיצוני)."""
        instance = cls()
        instance._records = df.to_dict('records')
        return instance

    def best_n_per_track(self, n=2, strategy='area'):
        """
        בוחרת n רשומות לכל sign_instance_id, לצורך טריאנגולציה בהמשך.

        strategy='area'      : n הרשומות עם ה-bbox הכי גדול (הכי קרובות
                                למצלמה = הכי ברורות לסיווג).
        strategy='max_spread' : הרשומה הראשונה והאחרונה בזמן - זה מה שנותן
                                את הבסיס (baseline) הכי רחב לטריאנגולציה.
                                'area' עלול לבחור 2 פריימים סמוכים מדי בזמן
                                (שניהם קרוב לרגע החלוף-על-פני התמרור), מה
                                שנותן זוויות דומות מדי -> טריאנגולציה רועשת.
        """
        df = self.to_dataframe()
        if df.empty:
            return df

        if strategy == 'area':
            return (df.sort_values('bbox_area', ascending=False)
                      .groupby('sign_instance_id', group_keys=False)
                      .head(n)
                      .sort_values(['sign_instance_id', 'frame_timestamp'])
                      .reset_index(drop=True))

        elif strategy == 'max_spread':
            g = df.sort_values('frame_timestamp').groupby('sign_instance_id', group_keys=False)
            first = g.head(1)
            last = g.tail(1)
            return (pd.concat([first, last])
                      .drop_duplicates()
                      .sort_values(['sign_instance_id', 'frame_timestamp'])
                      .reset_index(drop=True))

        raise ValueError(f"strategy לא מוכר: {strategy}")

    def save(self, path):
        self.to_dataframe().to_csv(path, index=False)
