# app.py

# نقطة الدخول الرئيسية لتطبيق FastAPI
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

# استخدام الاستيراد المطلق لضمان عمله في بيئات النشر
from EmotionalProcessorV4 import EmotionalEngine
from EmotionalState import EmotionalState
from PromptBuilder import PromptBuilder

# تهيئة Firebase (هذه الخطوة غير ضرورية حاليًا ما دمنا نستخدم SQLite محليًا، لكنها خطوة جيدة)
# سنقوم بالتهيئة الأساسية هنا، لكن التطبيق يعتمد على SQLite حاليًا
# app = initialize_firebase_app() 

# تهيئة محرك العواطف (يستخدم حالة عاطفية محددة)
state_manager = EmotionalState()
# هنا يتم تهيئة EmotionalEngine، حيث يتم تمرير state_manager إليه
# ويتم تهيئة LLM client بداخله بشكل آمن
engine = EmotionalEngine(state_manager=state_manager)

app = FastAPI(
    title="Emotional Chat API",
    description="API for the emotionally aware chat companion.",
    version="1.0.0"
)

# تفعيل CORS للسماح بالوصول من أي مصدر (مهم للتطبيقات الـ Frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # يسمح لأي نطاق بالوصول
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    """ نقطة وصول أساسية للتحقق من أن الـ API يعمل. """
    return {"status": "Operational", "message": "Emotional Chat API is running."}

@app.post("/chat")
def chat_endpoint(user_prompt: str):
    """ نقطة وصول لمعالجة طلبات الدردشة مع المستخدم. """
    try:
        # معالجة الطلب عبر محرك العواطف
        response_text, state_update = engine.process_message(user_prompt)
        
        return {
            "response": response_text,
            "current_state": state_update
        }
    except Exception as e:
        # معالجة الأخطاء وإرسال رسالة خطأ واضحة
        return {"response": f"An error occurred: {str(e)}", "current_state": "Error"}

@app.get("/state")
def get_state():
    """ نقطة وصول للحصول على الحالة العاطفية الحالية. """
    return engine.get_current_state()

# @app.post("/reset")
# def reset_state_endpoint():
#     """ نقطة وصول لإعادة تعيين حالة العواطف (اختياري). """
#     # يمكن إضافة منطق إعادة تعيين الحالة هنا
#     return {"status": "State reset placeholder"}
