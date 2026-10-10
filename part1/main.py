import os

os.environ["HTTPX_CLIENT_VERIFY"] = "False"
os.environ["GRPC_NOT_USE_SYSTEM_SSL_ROOT_CERTS"] = "true"
os.environ["PYTHONHTTPSVERIFY"] = "0"

import cv2
import logging
import pandas as pd
import supervision as sv
from concurrent.futures import ThreadPoolExecutor
from yolo_engine import YOLOEngine
from state_manager import StateManager
from gemini_agent import GeminiAgent
from optical_flow_processor import OpticalFlowProcessor
from detection_logger import DetectionLogger  # <-- חדש

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FRAME_STRIDE = 3
CONF_LOW = 0.42
CONF_HIGH = 0.81
MIN_AREA_FOR_GEMINI = 600

# 🚀 חדש: העוגן בין זמן-וידאו לזמן-אמת. חייב להגיע ממקור אמיתי אצלך
# (שם קובץ, EXIF, לוג נפרד) - כרגע רק placeholder.
VIDEO_START_WALLTIME = pd.Timestamp("2026-07-07 15:00:00")
DRIVE_UUID = "messy-trip-001"


def main():
    yolo = YOLOEngine()
    sm = StateManager()
    gemini = GeminiAgent()
    flow = OpticalFlowProcessor()
    detection_logger = DetectionLogger()  # <-- חדש

    executor = ThreadPoolExecutor(max_workers=3)
    pending_gemini_checks = set()

    cap = cv2.VideoCapture("video.mp4")
    if not cap.isOpened():
        logger.error("❌ לא ניתן לפתוח את קובץ הוידאו.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)  # <-- חדש, קריטי לחישוב הזמן האמיתי
    if not fps or fps <= 0:
        logger.warning("⚠️ FPS לא תקין מהווידאו, נופל חזרה ל-30 כברירת מחדל")
        fps = 30.0

    frame_idx = 0
    logger.info("🚀 מערכת התחילה לעבוד במצב ריצה אוטומטי (ללא תצוגה).")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logger.info("🎥 הסרטון הסתיים.")
            break

        frame_idx += 1
        is_stride_frame = (frame_idx % FRAME_STRIDE == 0)

        # 🚀 חדש: תרגום frame_idx לזמן אמיתי, מסונכרן עם ה-GPS
        frame_timestamp = VIDEO_START_WALLTIME + pd.Timedelta(seconds=frame_idx / fps)

        detections = None
        if is_stride_frame and flow.is_vehicle_moving(frame):
            detections = yolo.predict(frame)

        if detections is not None and detections.tracker_id is not None:
            active_ids_this_call = set(detections.tracker_id)

            for i in range(len(detections)):
                tid = detections.tracker_id[i]

                # 🚀 חדש: רישום קורה ראשון, לפני כל בדיקת is_rejected/is_verified -
                # אחרת אנחנו מפסיקים לרשום בדיוק ברגע שתמרור מאושר, ומפספסים
                # את הפריימים הכי טובים (הכי קרובים, עם ה-bbox הגדול ביותר)
                # שממשיכים להגיע אחרי האישור.
                x1_raw, y1_raw, x2_raw, y2_raw = map(int, detections.xyxy[i])
                h_raw, w_raw = frame.shape[:2]
                detection_logger.log_detection(
                    frame_timestamp=frame_timestamp, tracker_id=tid,
                    class_name=detections.data['class_name'][i],
                    confidence=float(detections.confidence[i]),
                    x1=x1_raw, y1=y1_raw, x2=x2_raw, y2=y2_raw,
                    image_width=w_raw, image_height=h_raw, drive_uuid=DRIVE_UUID,
                )

                if sm.is_rejected(tid) or sm.is_verified(tid):
                    continue

                class_name = detections.data['class_name'][i]
                conf = float(detections.confidence[i])
                x1, y1, x2, y2 = x1_raw, y1_raw, x2_raw, y2_raw
                h, w = h_raw, w_raw

                crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                area = (x2 - x1) * (y2 - y1)

                if conf < CONF_LOW:
                    sm.mark_seen(tid)
                    continue

                if conf <= CONF_HIGH and tid not in pending_gemini_checks:
                    if area > MIN_AREA_FOR_GEMINI:
                        logger.info(f"🤖 נשלח לבדיקת רקע: ID={tid}")
                        pending_gemini_checks.add(tid)
                        sm.mark_pending(tid)  # 🚀 חדש - מונע race עם tick_missing
                        executor.submit(gemini.analyze_crop_async, tid, crop, class_name, sm)
                    else:
                        sm.mark_seen(tid)
                else:
                    sm.update_id(tid, class_name, frame, area)

            sm.tick_missing(active_ids_this_call)

    cap.release()
    executor.shutdown(wait=True)
    sm.flush_all()

    # 🚀 חדש: סינון - שמירת רק זיהויים ששרדו כ"מאושרים" בסוף (לא נדחו),
    # ואז בחירת 2 הפריימים עם ההפרש הזמני/זוויתי הכי גדול לטריאנגולציה
    # 🚀 עודכן: לא לפי is_verified (זה שקרי עבור זיהויים שדילגו על Gemini
    # ומעולם לא סומנו gemini_verified=True) - אלא לפי מי שבאמת נכתב
    # ל-results.csv בפועל (כלומר לא נפסל).
    final_ids = sm.get_final_ids()
    all_detections = detection_logger.to_dataframe()
    verified_detections = all_detections[all_detections['sign_instance_id'].isin(final_ids)]

    best_pairs = DetectionLogger.from_dataframe(verified_detections)
    best_pairs.best_n_per_track(2, strategy='max_spread').to_csv('sign_detections_for_angle.csv', index=False)

    logger.info("🏁 סיום. כל הנתונים נשמרו ב-CSV.")


if __name__ == "__main__":
    main()
