import os
import json
import logging

logger = logging.getLogger(__name__)


class LabelTranslator:
    def __init__(self, map_filename="labels_map.json"):
        """
        מחלקה האחראית על קריאת שמות מחלקות כבר תורגמים.
        טוען מילון מ-labels_map.json שמכיל צרפתית ואנגלית.
        """
        self.translation_map = {}
        
        # טעינת המילון מהקובץ ברגע אתחול המחלקה
        if os.path.exists(map_filename):
            try:
                with open(map_filename, "r", encoding="utf-8") as f:
                    self.translation_map = json.load(f)
                logger.info("Translation map loaded from {} ({} labels)".format(map_filename, len(self.translation_map)))
            except Exception as e:
                logger.error("Failed to load translation map file: {}".format(e))
        else:
            logger.warning("Translation map file {} not found".format(map_filename))

    def translate(self, class_id, fallback_name=None):
        """
        קריאת התרגום האנגלית מהמילון המקומי לפי class_id.
        labels_map.json כבר מכיל צרפתית ואנגלית (כבר תורגם בזמן fetch_labels.py).
        
        Args:
            class_id: The class ID (int or str)
            fallback_name: Optional fallback if translation not found
        
        Returns:
            English label string
        """
        class_id_str = str(class_id)

        # בדיקה האם המזהה קיים במילון הנטען מקובץ ה-JSON
        if class_id_str in self.translation_map:
            label_data = self.translation_map[class_id_str]
            
            # אם זה מילון עם "en" key, החזר את התרגום לאנגלית
            if isinstance(label_data, dict) and "en" in label_data:
                return label_data["en"]
            
            # backwards compatibility: אם זה string, החזר אותו
            if isinstance(label_data, str):
                return label_data
        
        # במקרה שהמפתח לא קיים, שימוש בגיבוי
        if fallback_name:
            logger.warning("Class ID {} not found. Using fallback: {}".format(class_id_str, fallback_name))
            return fallback_name
        
        return str(class_id)


# אתחול מופע גלובלי של השירות לשימוש ברחבי הפרויקט
translator_service = LabelTranslator()