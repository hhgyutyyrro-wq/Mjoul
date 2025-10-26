# EmotionalState.py - تخزين الحالة والذاكرة

import sqlite3
import json
import time
from typing import Dict, Any

class EmotionalState:
    """تدير تخزين حالة الكائن العاطفية في قاعدة بيانات SQLite."""

    def init(self, db_path: str = 'emotions.db'):
        """تهيئة الكلاس وتحميل الحالة الحالية من DB أو تهيئتها."""
        self.db_path = db_path
        self.initialize_db()
        self.state = self.load_state()

        # القيم الافتراضية
        initial_state = {
            'temperament_bias': 0.5, # الانحياز المزاجي (ثابت عادة)
            'maturity': 1.0,         # النضج (يزداد بمرور الوقت)
            'joy': 0.5,
            'fear': 0.0,
            'calm': 0.5,
            'anxiety': 0.0,
            'pride': 0.0,
            'guilt': 0.0,
            # يمكن إضافة المزيد من المشاعر هنا
        }
        
        # تحميل الحالة أو استخدام الافتراضيات
        if not self.state:
            self.state = initial_state
            self.save_state(self.state) # حفظ الحالة الأولية

    def initialize_db(self):
        """التطور 1: إنشاء قاعدة بيانات وتهيئة الجدول الرئيسي."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # جدول 'state' لتخزين بيانات الحالة الرئيسية (قيمة واحدة لكل مفتاح)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS state (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)

        # جدول 'log' لتسجيل التفاعلات (اختياري)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS log (
                id INTEGER PRIMARY KEY,
                timestamp INTEGER,
                data TEXT
            )
        """)
        
        conn.commit()
        conn.close()

    def load_state(self) -> Dict[str, float]:
        """تحميل الحالة العاطفية من قاعدة بيانات SQLite."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        rows = cursor.execute("SELECT key, value FROM state").fetchall()
        conn.close()
        
        state: Dict[str, float] = {}
        if not rows:
            return {}

        for key, value_str in rows:
            try:
                # محاولة تحويل القيمة إلى Float
                state[key] = float(value_str)
            except ValueError:
                # إذا لم يكن رقمًا، يمكن تخزينه كسلسلة أو تجاهله
                state[key] = value_str # يجب أن تكون معظم قيمنا Float
        
        return state

    def save_state(self, new_state: Dict[str, Any]):
        """حفظ الحالة العاطفية في قاعدة بيانات SQLite."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for key, value in new_state.items():
            # تحويل القيمة إلى سلسلة قبل الحفظ
            value_str = str(value) 
            
            cursor.execute(
                "INSERT OR REPLACE INTO state (key, value) VALUES (?, ?)",
                (key, value_str)
            )
            
        conn.commit()
        conn.close()

        # تحديث الحالة الداخلية
        self.state.update(new_state)

    def log_interaction(self, data: Dict[str, Any]):
        """تسجيل تفاعل المستخدم في جدول log (اختياري)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = int(time.time())
        data_json = json.dumps(data)
        
        cursor.execute(
            "INSERT INTO log (timestamp, data) VALUES (?, ?)",
            (timestamp, data_json)
        )
        
        conn.commit()
        conn.close()
