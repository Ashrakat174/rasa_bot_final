# import os, logging, re, requests, chromadb, numpy as np
# import torch
# from rasa_sdk import Action, Tracker
# from rasa_sdk.executor import CollectingDispatcher
# from sentence_transformers import SentenceTransformer
# from dotenv import load_dotenv

# # إعداد اللوج
# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)

# # تحميل المتغيرات
# load_dotenv()
# API_KEY = os.getenv("OPENROUTER_API_KEY")

# # موديل التضمين
# embedder = SentenceTransformer("intfloat/multilingual-e5-large")

# def encode(text, is_query=True):
#     with torch.no_grad():
#         return embedder.encode(text, convert_to_numpy=True).astype("float32")

# # ChromaDB
# client = chromadb.PersistentClient(path="chroma_db")
# faq_collection = client.get_collection("faq_only")

# NON_ACADEMIC_PHRASES = ["اسمك", "اخبارك", "حبيبي", "بتحبني", "مساء الخير", "صباح الخير"]

# def is_academic_question(text):
#     return not any(p in text.lower() for p in NON_ACADEMIC_PHRASES)

# def shorten(text, max_sentences=2):
#     sentences = re.split(r'(?<=[.!؟])\s+', text)
#     return ' '.join(sentences[:max_sentences]).strip()

# def retrieve_context(collection, question, top_k=4, threshold=0.3):
#     query_emb = encode(question).tolist()
#     result = collection.query(
#         query_embeddings=[query_emb],
#         n_results=top_k,
#         include=["documents", "embeddings"]
#     )
#     docs = result.get("documents", [[]])[0]
#     embs = result.get("embeddings", [[]])[0]

#     if not docs:
#         return []

#     scored = []
#     for doc, emb in zip(docs, embs):
#         sim = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb))
#         if sim >= threshold:
#             scored.append((sim, doc))

#     return sorted(scored, key=lambda x: x[0], reverse=True)

# def rephrase_with_model(text, question):
#     prompt = f"""أنت مساعد جامعي ذكي ومتخصص في الرد على استفسارات الطلاب حول كلية الذكاء الاصطناعي.
#     مهمتك هي إعادة صياغة الإجابات بشكل أكاديمي واضح، وبأسلوب مبسط يسهل فهمه خاصة للطلاب الجدد أو المقبلين على الالتحاق بالكلية.
#     احرص على أن تكون الإجابة شاملة تمامًا، ولا تختصر أو تحذف أي معلومة، بل وضّحها بشكل أفضل مع الحفاظ على المعنى الأصلي.
#     إذا لاحظت أن الإجابة لا تحتوي على معلومات كافية، أضف في نهايتها تنويهاً مهذباً يشير إلى ضرورة الرجوع إلى إدارة الكلية أو الجهة المختصة.

#     السؤال: {question}
#     الإجابة الأولية: {text}

#     أعد صياغة الإجابة كاملة بشكل واضح وسلس وبدون أي اختصار، مع الالتزام بالتعليمات أعلاه."""


#     try:
#         r = requests.post(
#             "https://openrouter.ai/api/v1/chat/completions",
#             headers={"Authorization": f"Bearer {API_KEY}"},
#             json={
#                 "model": "openai/gpt-3.5-turbo",
#                 "messages": [
#                     {"role": "system", "content": "أعد صياغة إجابة الطالب بشكل أكاديمي واضح"},
#                     {"role": "user", "content": prompt}
#                 ]
#             }
#         )
#         if r.status_code == 200:
#             return shorten(r.json()["choices"][0]["message"]["content"].strip())
#         else:
#             logger.error(f"Rephrase Error: {r.status_code} - {r.text}")
#             return shorten(text)
#     except Exception as e:
#         logger.exception("خطأ في الاتصال بـ LLM.")
#         return shorten(text)


# class ActionRAG(Action):
#     def name(self):
#         return "action_rag_answer"

#     def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict):
#         question = tracker.latest_message.get("text", "").strip()
#         logger.info(f"📥 سؤال المستخدم: {question}")

#         if len(question.split()) < 2:
#             dispatcher.utter_message("من فضلك، اكتب سؤالك بشكل واضح.")
#             return []

#         if not is_academic_question(question):
#             dispatcher.utter_message("أنا هنا للرد على استفسارات الكلية فقط 😊")
#             return []

#         results = retrieve_context(faq_collection, question)

#         if not results:
#             dispatcher.utter_message("لم أجد معلومات كافية للإجابة على سؤالك.")
#             return []

#         best_score, best_doc = results[0]
#         logger.info(f"✅ أفضل نتيجة - Score: {best_score:.4f}")
#         answer = rephrase_with_model(best_doc, question)
#         dispatcher.utter_message(answer)
#         return []




import os, logging, re, requests, chromadb, numpy as np
import torch
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# إعداد اللوج
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# تحميل المتغيرات
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
API_KEY = os.getenv("OPENROUTER_API_KEY")

# موديل التضمين
embedder = SentenceTransformer("intfloat/multilingual-e5-large")

def encode(text):
    with torch.no_grad():
        return embedder.encode(text, convert_to_numpy=True).astype("float32")

# ChromaDB
client = chromadb.PersistentClient(path="chroma_db")
faq_collection = client.get_collection("faq_only")

# عبارات غير أكاديمية
NON_ACADEMIC_PHRASES = ["اسمك", "اخبارك", "حبيبي", "بتحبني", "مساء الخير", "صباح الخير"]

def is_academic_question(text):
    return not any(p in text.lower() for p in NON_ACADEMIC_PHRASES)

def shorten(text, max_sentences=5):
    sentences = re.split(r'(?<=[.!؟])\s+', text)
    return ' '.join(sentences[:max_sentences]).strip()

def retrieve_context(collection, question, top_k=4, threshold=0.35):
    query_emb = encode(question).tolist()
    result = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k,
        include=["documents", "embeddings"]
    )
    docs = result.get("documents", [[]])[0]
    embs = result.get("embeddings", [[]])[0]

    if not docs:
        return []

    scored = []
    for doc, emb in zip(docs, embs):
        sim = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb))
        if sim >= threshold:
            scored.append((sim, doc))

    return sorted(scored, key=lambda x: x[0], reverse=True)

def extract_memory_context(tracker: Tracker, max_turns=2):
    memory = []
    events = tracker.events[::-1]
    turns = 0
    last_user = ""
    last_bot = ""
    for event in events:
        if event.get("event") == "bot":
            last_bot = event.get("text", "")
        elif event.get("event") == "user":
            last_user = event.get("text", "")
            if last_user and last_bot:
                memory.append(f"سؤال سابق: {last_user}\nالإجابة: {last_bot}")
                turns += 1
                last_user = ""
                last_bot = ""
                if turns >= max_turns:
                    break
    return "\n\n".join(memory[::-1])  # الأقدم أولًا

def rephrase_with_model(text, question, memory=""):
    table_hint = """
مرفق أدناه جدول مواد لائحة الكلية وصفحاتها:

- تمهيد – صفحة 3
- أهداف الكلية – صفحة 4
- ملحق (1): نموذج الخطة الدراسية للمقررات – صفحة 42
- ملحق (2): وصف المقررات – صفحة 50
- مادة (1): قواعد القبول – صفحة 5
- مادة (2): أقسام الكلية – صفحة 5
- مادة (3): الدرجات العلمية – صفحة 10
- مادة (4): لغة التدريس – صفحة 11
- مادة (5): التعليم عن بعد – صفحة 11
- مادة (6): نظام الدراسة – صفحة 11
- مادة (7): الإرشاد الأكاديمي – صفحة 12
- مادة (8): التسجيل والحذف والإضافة – صفحة 12
- مادة (9): الانسحاب من المقرر – صفحة 13
- مادة (10): الغياب والمواظبة – صفحة 13
- مادة (11): الانقطاع عن الدراسة – صفحة 14
- مادة (12): الفصل من الكلية – صفحة 14
- مادة (13): الانتقال بين المستويات – صفحة 15
- مادة (14): التحويل من كليات أخرى – صفحة 15
- مادة (15): نظام الامتحانات – صفحة 15
- مادة (16): نظام التقويم – صفحة 16
- مادة (17): الرسوب والإعادة – صفحة 17
- مادة (18): مشروع التخرج – صفحة 18
- مادة (19): التدريب العملي والميداني – صفحة 18
- مادة (20): رسوم الدراسة – صفحة 18
- مادة (21): المنح الدراسية – صفحة 19
- مادة (22): دعم الطلاب – صفحة 19
- مادة (23): نظام الاستماع – صفحة 20
- مادة (25): مجلس إدارة البرامج المميزة – صفحة 21
- مادة (26): أحكام تنظيمية – صفحة 22
- مادة (27): متطلبات الدراسة – صفحة 22
- المتطلبات العامة – صفحة 24
- متطلبات الكلية – صفحة 25
- متطلبات التخصص – صفحة 29
"""

    prompt = f"""أنت مساعد جامعي ذكي ومتخصص في الرد على استفسارات الطلاب حول كلية الذكاء الاصطناعي.
مهمتك هي إعادة صياغة الإجابات بشكل أكاديمي واضح، وبدون اختصار أو حذف للمعلومات، مع توضيح أي نقاط غامضة لتكون مفهومة للطلاب الجدد.
إذا كانت هناك إجابة غير كافية، أضف ملاحظة مهذبة تحث الطالب على الرجوع لإدارة الكلية.
إذا كانت الإجابة تتعلق بموضوع مذكور في اللائحة، فأضف في النهاية (المصدر: مادة أو ملحق رقم X - صفحة Y من اللائحة الدراسية).

{table_hint}

السياق السابق:
{memory}

السؤال الحالي: {question}
الإجابة الأولية من قاعدة البيانات: {text}

أعد صياغة الإجابة كاملة وبدون اختصار وبأسلوب أكاديمي واضح.
"""

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://your-app.com",
                "X-Title": "TestBot"
            },
            json={
                "model": "google/gemini-2.0-flash-001",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 1000
            }
        )

        if response.status_code == 200:
            return shorten(response.json()["choices"][0]["message"]["content"].strip(), 7)
        else:
            logger.error(f"❌ Gemini Error {response.status_code}: {response.text}")
            return shorten(text)
    except Exception as e:
        logger.exception("⚠️ خطأ في الاتصال بـ Gemini.")
        return shorten(text)

class ActionRAGAnswer(Action):
    def name(self):
        return "action_rag_answer"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict):
        question = tracker.latest_message.get("text", "").strip()
        logger.info(f"📥 سؤال المستخدم: {question}")

        if len(question.split()) < 2:
            dispatcher.utter_message("من فضلك، اكتب سؤالك بشكل واضح.")
            return []

        if not is_academic_question(question):
            dispatcher.utter_message("أنا هنا للرد على استفسارات الكلية فقط 😊")
            return []

        results = retrieve_context(faq_collection, question)

        if not results:
            dispatcher.utter_message("لم أجد معلومات كافية للإجابة على سؤالك.")
            return []

        best_score, best_doc = results[0]
        logger.info(f"✅ أفضل نتيجة - Score: {best_score:.4f}")
        answer = rephrase_with_model(best_doc, question)
        dispatcher.utter_message(answer)
        return []
    
