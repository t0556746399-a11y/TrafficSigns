import os
import httpx
import base64
import cv2
import supervision as sv
import numpy as np
from dotenv import load_dotenv
import urllib3
import logging
from translator_service import translator_service
# הגדרות לוגינג
logger = logging.getLogger(__name__)

# עקיפת אזהרות SSL לטובת עבודה מול נטפרי
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()

class YOLOEngine:
    def __init__(self):
        self.api_key = os.getenv("ROBOFLOW_API_KEY")
        self.model_id = os.getenv("ROBOFLOW_MODEL_ID")
        self.api_url = f"https://detect.roboflow.com/{self.model_id}?api_key={self.api_key}&format=json"

        # 🚀 אתחול טרקר מעודכן: זיכרון ארוך יותר (60) ורגישות מותאמת לפניות
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.20,
            lost_track_buffer=60,  # זיכרון מורחב לאובייקטים אבודים בסיבובים
            minimum_consecutive_frames=2
        )

        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

        logger.info(f"YOLOEngine initialized successfully for model: {self.model_id} with continuous ByteTrack.")
    def predict(self, frame, skip_api=False, min_confidence=0.15):
        """
        חיזוי וטרקינג - מותאם אישית למהירות ויציבות.
        """
        if skip_api or frame is None:
            return self.tracker.update_with_detections(sv.Detections.empty())

        # דחיסה אגרסיבית (30) להפחתת עומס הרשת ומניעת Timeout
        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 30])
        img_str = base64.b64encode(buffer).decode("ascii")

        try:
            # שימוש ב-httpx במקום requests ליציבות גבוהה יותר
            with httpx.Client(verify=False, timeout=8.0) as client:
                response = client.post(
                    self.api_url,
                    data=img_str,
                    headers={"Content-Type": "application/x-www-form-urlencoded"}
                )

            if response.status_code != 200:
                logger.warning(f"Roboflow API returned {response.status_code}")
                return self.tracker.update_with_detections(sv.Detections.empty())
            result = response.json()
            predictions = result.get('predictions', [])

            if not predictions:
                return self.tracker.update_with_detections(sv.Detections.empty())

            # המרת התוצאות לפורמט Supervision
            xyxy = np.array([[p['x'] - p['width'] / 2, p['y'] - p['height'] / 2,
                              p['x'] + p['width'] / 2, p['y'] + p['height'] / 2] for p in predictions])
            conf = np.array([float(p['confidence']) for p in predictions])
            cls = np.array([int(p['class_id']) for p in predictions])

            # שילוב התרגום: העברת שם המחלקה דרך שירות התרגום הסטטי לפני יצירת המערך
            # תרגום שמות המחלקות - הבדיקות והלוגים מבוצעים אוטומטית בתוך ה-translator_service
            class_names = np.array([
                translator_service.translate(p['class_id'], fallback_name=p['class'])
                for p in predictions
            ])

            # יצירת אובייקט ה-Detections של supervision עם השמות המעודכנים
            detections = sv.Detections(
                xyxy=xyxy,
                confidence=conf,
                class_id=cls,
                data={'class_name': class_names}
            )
            filtered_detections = detections[detections.confidence > min_confidence]
            return self.tracker.update_with_detections(filtered_detections)
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return self.tracker.update_with_detections(sv.Detections.empty())

    def annotate(self, frame, detections):
        if detections is None or len(detections) == 0:
            return frame

        # תצוגה כוללת ID
        labels = [
            f"#{detections.tracker_id[i]} {detections.data['class_name'][i]} {detections.confidence[i]:.2f}"
            for i in range(len(detections))
        ]

        annotated_frame = self.box_annotator.annotate(scene=frame.copy(), detections=detections)
        return self.label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)