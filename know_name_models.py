from google import genai

from app.config import GEMINI_API_KEY

client = genai.Client(api_key=GEMINI_API_KEY)

# ============== عرض النماذج المتاحة ==============
print("جاري البحث عن كل النماذج المتاحة...\n")

try:
    for m in client.models.list():
        if "gemini" in m.name.lower() and "embedding" not in m.name.lower():
            print(f"✅ اسم النموذج: {m.name}")
except Exception as e:
    print(f"حدث خطأ: {e}")
