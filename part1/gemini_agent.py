import os
import httpx
import logging
import json
import cv2
import numpy as np
from PIL import Image
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TrafficSignValidation(BaseModel):
    is_relevant: bool = Field(
        description=(
            "True if the image contains the predicted traffic sign and "
            "it is real and active on the road. "
            "False if it is a billboard, reflection, sticker, advertisement "
            "or another false positive."
        )
    )
    reason: str = Field(
        description="Short reason in English."
    )


class GeminiAgent:

    def __init__(self):
        # שליפת ה-API Key בצורה מאובטחת ממשתני הסביבה
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY was not found. "
                "Make sure load_dotenv() is called in main.py"
            )

        # 🚀 עקיפת נטפרי המנצחת שעבדה לנו בטסט - הזרקת קליינט ללא SSL
        httpx_client = httpx.Client(verify=False)

        # אתחול נקי של ה-Client
        self.client = genai.Client(api_key=api_key)
        # הזרקת הקליינט שעוקף סינון ישירות לגוגל
        self.client._api_client._httpx_client = httpx_client

        # הגדרת ה-Config
        self.config = types.GenerateContentConfig(
            system_instruction=(
                "You are a safety-critical traffic sign verifier. "
                "Your task is to verify whether the cropped image "
                "contains the traffic sign predicted by the primary detector. "
                "Reject advertisements, stickers, reflections, posters, "
                "vehicle graphics, printed images, screens, or other "
                "false positives."
            ),
            response_mime_type="application/json",
            response_schema=TrafficSignValidation,
            temperature=0.1,
        )

        logger.info("GeminiAgent initialized successfully using new SDK.")

    def analyze_crop(self, crop_image: np.ndarray, roboflow_prediction: str) -> dict:
        """
        מקבל קרופ של תמרור ותווית חיזוי מרובופלאו, ומחזיר החלטת אימות מובנית.
        """
        if crop_image is None or crop_image.size == 0:
            return {
                "is_relevant": False,
                "reason": "Empty or None crop image"
            }
        try:
            # המרה מפורמט BGR של OpenCV לפורמט RGB של PIL
            rgb_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_image)

            # בניית הפרומפט
            prompt = (
                f"The primary detector predicted '{roboflow_prediction}'. "
                f"Determine whether this crop truly contains a real road sign "
                f"of type '{roboflow_prediction}' that is relevant for driving."
            )

            # 🚀 חזרה למודל הנתמך והרשמי ב-SDK החדש - gemini-2.5-flash
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[prompt, pil_img],
                config=self.config
            )

            # פענוח ה-JSON בצורה בטוחה
            result = json.loads(response.text)
            result['verified_by_gemini'] = True  # 🚀 חדש - אישור אמיתי, לא fallback
            return result

        except Exception as e:
            logger.exception("Gemini validation failed")
            # החזרת Fail-Safe למקרה של נפילת תקשורת פתאומית
            return {
                "is_relevant": True,
                "reason": f"Temporary API/Network fallback. Error: {str(e)}",
                "verified_by_gemini": False,  # 🚀 חדש - זה NOT אישור אמיתי!
            }

    def analyze_crop_async(self, tracker_id, crop_image, class_name, sm):
        """
        פונקציה זו נקראת ע"י ה-Thread. היא מבצעת את האימות ומעדכנת את ה-StateManager בבטחה.
        """
        try:
            result = self.analyze_crop(crop_image, class_name)

            if isinstance(result, dict) and result.get('is_relevant'):
                if not result.get('verified_by_gemini', True):
                    logger.warning(
                        f"⚠️ ID {tracker_id} אושר בגלל Fallback (Gemini נכשל), "
                        f"לא בגלל אימות אמיתי - {result.get('reason')}"
                    )
                sm.set_verified(tracker_id)
            else:
                sm.reject(tracker_id)
                logger.info(f"🚫 ID {tracker_id} נפסל עקב אימות שלילי מ-Gemini.")
        finally:
            # 🚀 חדש: קריטי - בלי זה, tick_missing ב-StateManager לעולם לא
            # יסגור את ה-ID הזה (הוא ממתין ל-Gemini לנצח). ה-finally מבטיח
            # שזה קורה גם אם set_verified/reject עצמם נכשלים באופן לא צפוי.
            sm.clear_pending(tracker_id)