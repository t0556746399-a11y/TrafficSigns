import os
import cv2
import pandas as pd
import logging
import threading

logger = logging.getLogger(__name__)


class StateManager:
    """
    מנהל את הזיכרון לטווח-ארוך של כל תמרור שזוהה.
    תומך ב-Multithreading ומבטיח שמירה של התמונות האיכותיות ביותר בלבד.
    """

    def __init__(self, csv_path="results.csv", crops_dir="detected_crops", missing_frames_grace=3):
        self.state_dict = {}
        self.rejected_ids = set()
        self.pending_gemini_ids = set()  # 🚀 חדש - מרוכז כאן, תחת אותו lock
        self.csv_path = csv_path
        self.crops_dir = crops_dir
        self.missing_frames_grace = missing_frames_grace
        self.lock = threading.Lock()

        if not os.path.exists(self.crops_dir):
            os.makedirs(self.crops_dir)

        if not os.path.exists(self.csv_path):
            pd.DataFrame(
                columns=["id", "class_name", "best_1", "best_2"]
            ).to_csv(self.csv_path, index=False)

    def _save_image(self, tracker_id, image, area):
        # שמירת התמונה עם שם הכולל את גודל השטח כדי להבדיל בקלות
        filename = os.path.join(self.crops_dir, f"id_{tracker_id}_area_{int(area)}.jpg")
        cv2.imwrite(filename, image)
        return filename

    def update_id(self, tracker_id, class_name, frame, area):
        with self.lock:
            if tracker_id in self.rejected_ids:
                return

            if tracker_id not in self.state_dict:
                path = self._save_image(tracker_id, frame, area)
                self.state_dict[tracker_id] = {
                    "class_name": class_name,
                    "best_images": [(area, path)],
                    "gemini_verified": False,
                    "missing_count": 0,
                }
            else:
                entry = self.state_dict[tracker_id]
                entry["missing_count"] = 0
                best_list = entry["best_images"]

                # אם התמונה הנוכחית גדולה יותר מהכי קטנה ששמרנו, נחליף ביניהן
                if len(best_list) < 2 or area > best_list[-1][0]:
                    path = self._save_image(tracker_id, frame, area)
                    best_list.append((area, path))
                    best_list.sort(key=lambda x: x[0], reverse=True)

                    # אם יש יותר מ-2, נוציא את הכי קטנה ונגמחק אותה מהדיסק
                    if len(best_list) > 2:
                        removed_item = best_list.pop()  # מוציא את האיבר האחרון (הכי קטן)
                        if os.path.exists(removed_item[1]):
                            os.remove(removed_item[1])  # מחיקה פיזית מהדיסק

                    entry["best_images"] = best_list
    def mark_pending(self, tracker_id):
        """קוראים לזה כשה-ID נשלח ל-Gemini, במקום pending_gemini_checks חיצוני."""
        with self.lock:
            self.pending_gemini_ids.add(tracker_id)

    def clear_pending(self, tracker_id):
        """
        קוראים לזה כש-Gemini סיים (אישר או פסל). אם ה-ID כבר "פג זמנו"
        (missing_count עבר את הסף) בזמן שחיכינו - סוגרים אותו רק עכשיו,
        אחרי שאנחנו סוף-סוף יודעים את התשובה.
        """
        with self.lock:
            self.pending_gemini_ids.discard(tracker_id)
            entry = self.state_dict.get(tracker_id)
            if entry and entry["missing_count"] >= self.missing_frames_grace:
                self._close_id_unlocked(tracker_id)

    def mark_seen(self, tracker_id):
        with self.lock:
            if tracker_id in self.state_dict:
                self.state_dict[tracker_id]["missing_count"] = 0

    def set_verified(self, tracker_id):
        with self.lock:
            if tracker_id in self.state_dict:
                self.state_dict[tracker_id]["gemini_verified"] = True
                logger.info(f"✅ ID {tracker_id} אושר על ידי ג'מיני!")

    def is_verified(self, tracker_id):
        with self.lock:
            return self.state_dict.get(tracker_id, {}).get("gemini_verified", False)

    def reject(self, tracker_id):
        with self.lock:
            self.rejected_ids.add(tracker_id)
            # מוציאים את המידע מהמילון
            data = self.state_dict.pop(tracker_id, None)

            # אם היה מידע (כלומר היו תמונות), נמחק את הקבצים
            if data:
                for img_data in data["best_images"]:
                    path = img_data[1]
                    if os.path.exists(path):
                        os.remove(path)

            logger.info(f"❌ ID {tracker_id} נפסל. התמונות הוסרו מהדיסק.")
    def is_rejected(self, tracker_id):
        with self.lock:
            return tracker_id in self.rejected_ids

    def tick_missing(self, active_ids_this_yolo_call):
        with self.lock:
            active_ids_set = set(active_ids_this_yolo_call)
            for tid in list(self.state_dict.keys()):
                if tid not in active_ids_set:
                    self.state_dict[tid]["missing_count"] += 1
                    if self.state_dict[tid]["missing_count"] >= self.missing_frames_grace:
                        # 🚀 חדש: אם עדיין ממתינים לתשובת Gemini על ה-ID הזה,
                        # לא סוגרים אותו עכשיו - clear_pending תסגור אותו
                        # אחר כך, ברגע שהתשובה תגיע (ראה למעלה).
                        if tid in self.pending_gemini_ids:
                            continue
                        self._close_id_unlocked(tid)

    def close_id(self, tracker_id):
        with self.lock:
            self._close_id_unlocked(tracker_id)

    def _close_id_unlocked(self, tracker_id):
        if tracker_id in self.state_dict:
            data = self.state_dict.pop(tracker_id)
            best_imgs = data["best_images"]
            p1 = best_imgs[0][1] if len(best_imgs) > 0 else ""
            p2 = best_imgs[1][1] if len(best_imgs) > 1 else ""

            new_row = pd.DataFrame([{
                "id": tracker_id,
                "class_name": data["class_name"],
                "best_1": p1,
                "best_2": p2,
            }])
            new_row.to_csv(self.csv_path, mode="a", header=False, index=False)
            print(f"✅ נשמר ל-CSV: תמרור מסוג '{data['class_name']}' (ID: {tracker_id})")
            logger.info(f"💾 ID {tracker_id} נשמר ל-CSV.")

    def flush_all(self):
        # בשלב הזה main.py כבר עשה executor.shutdown(wait=True), אז אין
        # בפועל שום pending_gemini_ids שרירותי שנשאר תלוי - סוגרים הכל.
        with self.lock:
            for tid in list(self.state_dict.keys()):
                self._close_id_unlocked(tid)

    def get_final_ids(self):
        """מחזיר את כל ה-IDs שבאמת נכתבו ל-results.csv (לא נפסלו)."""
        with self.lock:
            if not os.path.exists(self.csv_path):
                return set()
            return set(pd.read_csv(self.csv_path)['id'])