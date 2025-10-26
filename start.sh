#!/usr/bin/env bash

# الأمر الصحيح لتشغيل خادم Uvicorn وتطبيق FastAPI
# سيتم استدعاء الكائن 'app' من الملف 'app.py'
uvicorn app:app --host 0.0.0.0 --port 10000
