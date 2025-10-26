# 2. توليد استجابة وهمية بناءً على محتوى المطالبة
        lambda_val = self._calculate_lambda()
        if lambda_val > 0.75:
            response = f"أنا سعيد جدًا بردك! (Lambda: {lambda_val:.2f}) - الرسالة: {user_prompt}"
        elif lambda_val < 0.25:
            response = f"أنا أشعر ببعض التوتر بشأن هذا. (Lambda: {lambda_val:.2f}) - الرسالة: {user_prompt}"
        else:
            response = f"حسناً، هذا مثير للاهتمام. (Lambda: {lambda_val:.2f}) - الرسالة: {user_prompt}"
        
        return response, new_emotions

    def _load_training_data(self):
        """تحميل بيانات تدريب وهمية للنموذج الداخلي."""
        self.X_train = np.array([[0.5, 0.5, 0.5], [0.1, 0.9, 0.1], [0.9, 0.1, 0.9]])
        self.y_train = np.array([0, 1, 2]) # 0: neutral, 1: positive, 2: negative
        self.emotions_features = ['joy', 'fear', 'calm']

    def _train_internal_model(self):
        """تدريب نموذج التعلم الآلي الداخلي."""
        try:
             self.internal_llm_model = RandomForestClassifier(n_estimators=10)
             self.internal_llm_model.fit(self.X_train, self.y_train)
        except Exception as e:
             # في حالة وجود خطأ في تهيئة النموذج (نقص المكتبات أو غيرها)
             print(f"Error training internal model: {e}")
             self.internal_llm_model = None

    def _predict_and_update_state(self, user_prompt: str) -> Dict[str, float]:
        """يتنبأ بالحالة العاطفية من المطالبة وتحديث الحالة."""
        
        # خطوة 1: استخراج الميزات العاطفية من المطالبة (محاكاة)
        # في تطبيق حقيقي، سيتم استخدام LLM أو NLP لتحليل النص
        
        # محاكاة تأثير المشاعر على الحالة:
        current_features = np.array([
            self.state.get('joy', 0.5), 
            self.state.get('fear', 0.5), 
            self.state.get('calm', 0.5)
        ]).reshape(1, -1)
        
        if self.internal_llm_model:
             prediction = self.internal_llm_model.predict(current_features)[0]
        else:
             # العودة إلى العشوائية إذا فشل النموذج
             prediction = random.choice([0, 1, 2])
        
        # خطوة 2: تطبيق التحديثات
        update_magnitude = 0.15 # حجم التغيير
        new_emotions = self.state.copy()
        
        if prediction == 1: # إيجابي
            new_emotions['joy'] = min(1.0, new_emotions.get('joy', 0) + update_magnitude)
            new_emotions['fear'] = max(0.0, new_emotions.get('fear', 0) - update_magnitude)
        elif prediction == 2: # سلبي
            new_emotions['fear'] = min(1.0, new_emotions.get('fear', 0) + update_magnitude)
            new_emotions['joy'] = max(0.0, new_emotions.get('joy', 0) - update_magnitude)
        
        # خطوة 3: تحديث الحالة وحفظها
        self.state.update(new_emotions)
        self.state_manager.save_state(new_emotions) # حفظ الحالة في SQLite
        
        return new_emotions

    def _generate_llm_response(self, user_prompt: str) -> Tuple[str, Dict[str, float]]:
        """يستخدم Gemini API لتوليد الاستجابة."""
        
        # 1. تحديث الحالة
        updated_state = self._predict_and_update_state(user_prompt)
        lambda_val = self._calculate_lambda()
        
        # 2. بناء المطالبة باستخدام الحالة الحالية
        system_prompt = PromptBuilder.build_system_prompt(self.state, lambda_val)
        
        # 3. استدعاء API
        try:
            client = self.llm_client
            
            # هنا يجب استخدام النموذج الذي تم تحديده في التهيئة
            model_name = self.internal_llm_model if self.internal_llm_model else 'gemini-2.5-flash'
            
            response = client.generate_content(
                 model=model_name,
                 contents=[user_prompt],
                 system_instruction=system_prompt
            )
            
            response_text = response.text
            
        except Exception as e:
            response_text = f"عذرًا، فشل الاتصال بخدمة Gemini API: {str(e)}"
            print(f"Gemini API Error: {e}")
            return response_text, updated_state
           def process_message(self, user_prompt: str) -> Tuple[str, Dict[str, float]]:
        """الواجهة العامة لمعالجة رسالة المستخدم."""
        
        if self.is_simulated:
             return self._generate_simulated_response(user_prompt)
        else:
             return self._generate_llm_response(user_prompt)

    def get_current_state(self) -> Dict[str, float]:
        """يعيد الحالة العاطفية الحالية."""
        return self.state
