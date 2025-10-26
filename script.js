// التعديل النهائي: تم تعيين رابط API إلى خدمة Render المنشورة (مع إضافة المسار الصحيح).
const API_ENDPOINT = "https://oo-4.onrender.com/api/ask"; 
// تأكد من أن الرابط هو https://oo-4.onrender.com/api/ask

function updateEmotionalDisplay(state, lambda_val) {
    document.getElementById('guilt-level').textContent = state.guilt.toFixed(2);
    document.getElementById('pride-level').textContent = state.pride.toFixed(2);
    document.getElementById('fear-level').textContent = state.fear.toFixed(2);
    document.getElementById('joy-level').textContent = state.joy.toFixed(2);
    document.getElementById('lambda-level').textContent = lambda_val.toFixed(2);
}

function displayMessage(sender, message) {
    const chatBox = document.getElementById('chat-box');
    const msgElement = document.createElement('p');
    msgElement.className = sender;
    msgElement.textContent = ${sender === 'user' ? 'أنت' : 'الروبوت'}: ${message};
    chatBox.appendChild(msgElement);
    chatBox.scrollTop = chatBox.scrollHeight;
}

async function sendMessage() {
    const userInput = document.getElementById('user-input');
    const prompt = userInput.value.trim();
    if (!prompt) return;

    displayMessage('user', prompt);
    userInput.value = '';

    try {
        const response = await fetch(API_ENDPOINT, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt: prompt })
        });

        if (!response.ok) {
            throw new Error(خطأ: ${response.status} - فشل الاتصال بخدمة EEAI);
        }

        const data = await response.json();
        
        displayMessage('ai', data.response_text);
        
        // تحديث لوحة المشاعر و Lambda بناءً على رد الخادم
        updateEmotionalDisplay(data.new_state, data.lambda_value);

    } catch (error) {
        // إذا ظهر هذا الخطأ مرة أخرى، فهذا يعني أن خادم Render لا يعمل بشكل صحيح
        displayMessage('error', حدث خطأ في الاتصال بالخدمة: ${error.message});
        console.error("Error communicating with API:", error);
    }
}

// تفعيل زر Enter للإرسال
document.getElementById('user-input').addEventListener('keypress', function (e) {
    if (e.key === 'Enter') {
        sendMessage();
    }
});
