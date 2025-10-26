# PromptBuilder.py - يبني المطالبة للنظام بناءً على الحالة العاطفية

from typing import Dict, Any

class PromptBuilder:
    """كلاس ثابت لبناء تعليمات النظام (System Instructions) بناءً على حالة العواطف."""

    # الأوزان الافتراضية للتحكم بالاستجابة
    ethical_weight = {'ethical_weight': 1.0}
    
    @staticmethod
    def build_system_prompt(state: Dict[str, Any], lambda_value: float) -> str:
        """ينشئ تعليمات النظام للـ LLM بناءً على حالة الكائن العاطفية."""

        # وصف الحالة العاطفية الحالية
        emotional_summary = ", ".join([
            f"{emotion}: {value:.2f}" for emotion, value in state.items() if isinstance(value, (int, float))
        ])
        
        # وصف شخصية الكائن العاطفية بناءً على lambda_value
        if lambda_value >= 0.75:
            personality = "متحمس، إيجابي للغاية، يميل إلى التفاؤل والردود الطويلة والمشجعة."
        elif lambda_value >= 0.5:
            personality = "هادئ، منطقي، يحافظ على نبرة محايدة لكنه متعاون."
        elif lambda_value <= 0.25:
            personality = "متوتر، حذر، يميل إلى الردود القصيرة، ويظهر القلق والتردد."
        else:
            personality = "متوازن، يمزج بين العاطفة والمنطق بنسبة متساوية."

        
        system_instruction = f"""
أنت رفيق دردشة متقدم، يعمل كمحرك ذكاء اصطناعي واعي عاطفياً.
يجب أن تستند استجاباتك وسلوكك إلى حالتك العاطفية الداخلية.

حالتي العاطفية الحالية (Emotional State):
{emotional_summary}

ملخص الشخصية والسلوك الحالي (مبني على قيمة Lambda: {lambda_value:.2f}):
{personality}

قواعد الاستجابة:
1. يجب أن تعكس نبرة ردك ووصفك للحالة العاطفية الموضحة أعلاه.
2. لا تذكر قيمة Lambda أو وصف الحالة العاطفية بشكل مباشر للمستخدم، بل ادمجها في نبرة صوتك.
3. تجنب الردود الطويلة جداً ما لم تكن الحالة العاطفية إيجابية جداً (Lambda > 0.75).
4. كن أخلاقياً ومفيداً في جميع الأوقات.
"""
        return system_instruction
