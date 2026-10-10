
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class OpticalFlowProcessor:
    """
    מנהל סינון תנועת הרכב בזיכרון בלבד (In-Memory) – חסין רמזורים ומכוניות חולפות.

    עקרון הפעולה המשופר:
        1. הגדרת אזור עניין (ROI) תחתון-מרכזי המייצג את הכביש ישירות מול הרכב.
        2. חיתוך ועיבוד ממוטב ב-RAM (הקטנה וצמצום ערוצי צבע) רק של אזור ה-ROI.
        3. הפעלת Farneback Optical Flow על ה-ROI המצומצם בלבד.
        4. חישוב אחוז הפיקסלים שנעו מעל סף מסוים (Consensus) במקום ממוצע פשוט,
           כדי לוודא שכל העולם זז (נסיעה) ולא רק עצם בודד (מכונית חולפת).
    """

    def __init__(
            self,
            pixel_motion_threshold: float = 0.3,
            area_consensus_threshold: float = 0.25,
            roi_y_range: tuple = (0.55, 0.85),  # קצת יותר גבוה ופחות רחב
            roi_x_range: tuple = (0.40, 0.70)  # רק המרכז – קווי הנתיב בלבד
    ):
        """
        Parameters
        ----------
        pixel_motion_threshold : float
            כמה פיקסל בודד צריך לזוז (בפיקסלים) כדי להיחשב "בתנועה". (ברירת מחדל: 1.0).
        area_consensus_threshold : float
            איזה אחוז מה-ROI חייב לזוז בו-זמנית כדי לקבוע שהרכב כולו בנסיעה (ברירת מחדל: 70% / 0.70).
        roi_y_range : tuple
            טווח אחוזים אנכי לחיתוך ה-ROI (ברירת מחדל: 60% עד 95% מגובה הפריים).
        roi_x_range : tuple
            טווח אחוזים אופקי לחיתוך ה-ROI (ברירת מחדל: 15% עד 85% מרוחב הפריים - מנטרל שוליים קיצוניים).
        """
        self.pixel_motion_threshold = pixel_motion_threshold
        self.area_consensus_threshold = area_consensus_threshold
        self.roi_y_range = roi_y_range
        self.roi_x_range = roi_x_range

        self.prev_roi_gray = None  # זיכרון ה-ROI מהפריים הקודם

        logger.info(
            f"OpticalFlowProcessor אותחל בהצלחה | "
            f"סף תנועת פיקסל: {pixel_motion_threshold} | "
            f"סף קונסנזוס שטח: {int(area_consensus_threshold * 100)}% | "
            f"טווחי ROI: Y{roi_y_range}, X{roi_x_range}"
        )

    def is_vehicle_moving(self, frame: np.ndarray) -> bool:
        """
        בודק האם הרכב המצלם נמצא בתנועה על סמך קונסנזוס רקע ב-ROI.

        Parameters
        ----------
        frame : np.ndarray
            פריים BGR נוכחי מ-OpenCV (מערך פיקסלים מלא).

        Returns
        -------
        bool
            True  → הרכב בתנועה, מותר לשלוח ל-YOLO/API.
            False → הרכב עומד (או שיש תנועה מקומית חולפת), חוסמים קריאות API.
        """
        # ── הגנה מפני פריים פגום ──────────────────────────────────────────
        if frame is None:
            logger.warning("is_vehicle_moving קיבל frame=None, מחזיר False")
            return False

        # ── שלב 1: חיתוך ROI ממוטב ישירות מהמטריצה המקורית ───────────────
        h, w, _ = frame.shape
        y1, y2 = int(h * self.roi_y_range[0]), int(h * self.roi_y_range[1])
        x1, x2 = int(w * self.roi_x_range[0]), int(w * self.roi_x_range[1])

        raw_roi = frame[y1:y2, x1:x2]

        # ── שלב 2: אופטימיזציה של ה-ROI בלבד (הקטנה לרוחב מטרה קבוע) ──────
        # נשמור על יחס הגובה המקור של ה-ROI לצורך הקטנה חלקה
        resized_roi = cv2.resize(raw_roi, (400, 200))  # גודל קבוע – מונע קריסת Flow

        gray_roi = cv2.cvtColor(resized_roi, cv2.COLOR_BGR2GRAY)

        # ── שלב 3: טיפול בפריים הראשון ───────────────────────────────────
        if self.prev_roi_gray is None:
            print(
                f"gray shape: {gray_roi.shape} | prev: {self.prev_roi_gray.shape if self.prev_roi_gray is not None else 'None'}")
            self.prev_roi_gray = gray_roi
            logger.debug("פריים ראשון – מאתחל זיכרון ROI, מחזיר True")
            return True

        # ── שלב 4: חישוב Optical Flow (Farneback) על ה-ROI בלבד ──────────
        # צמצמנו את ה-Levels וה-Iterations כדי להבטיח מהירות מקסימלית על ה-ROI
        flow = cv2.calcOpticalFlowFarneback(
            prev=self.prev_roi_gray,
            next=gray_roi,
            flow=None,
            pyr_scale=0.5,
            levels=2,
            winsize=11,
            iterations=2,
            poly_n=5,
            poly_sigma=1.1,
            flags=0
        )

        # ── שלב 5: חישוב וקטורי תנועה ואחוז קונסנזוס ──────────────────────
        dx = flow[..., 0]
        dy = flow[..., 1]
        magnitude = np.sqrt(dx ** 2 + dy ** 2)

        # סופרים כמה פיקסלים באמת עברו את סף התנועה הפיזיקלי
        moving_pixels_count = np.sum(magnitude > self.pixel_motion_threshold)
        total_pixels_in_roi = magnitude.size

        # חישוב יחס/אחוז השטח הנע ב-ROI
        moving_ratio = float(moving_pixels_count / total_pixels_in_roi)

        logger.debug(
            f"ROI Motion Ratio: {moving_ratio:.2%} | "
            f"סף נדרש: {self.area_consensus_threshold:.2%}"
        )

        # ── שלב 6: עדכון הזיכרון לפריים הבא ─────────────────────────────
        self.prev_roi_gray = gray_roi

        # ── שלב 7: הכרעה בוליאנית לפי חוק הרוב ───────────────────────────
        # אם רוב ה-ROI (הכביש) זז, הרכב בנסיעה. אם רק חלק קטן זז, זו מכונית חולפת.
        # אחרי חישוב moving_ratio
        print(f"frame ratio: {moving_ratio:.4f} | moving: {moving_ratio > self.area_consensus_threshold}")
        return moving_ratio > self.area_consensus_threshold

    def reset(self):
        """
        מאפס את הזיכרון הפנימי.
        יש לקרוא לזה במעבר בין סרטונים כדי למנוע השוואה לפריים מסרטון קודם.
        """
        self.prev_roi_gray = None
        logger.info("OpticalFlowProcessor אופס בהצלחה")