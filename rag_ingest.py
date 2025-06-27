# import json, torch, chromadb
# from sentence_transformers import SentenceTransformer
# from tqdm import tqdm

# # تحميل موديل التضمين
# embedder = SentenceTransformer("intfloat/multilingual-e5-large")

# # إعداد قاعدة البيانات
# client = chromadb.PersistentClient(path="chroma_db")

# # إنشاء أو استرجاع التجميعة
# COLLECTION_NAME = "faq_only"
# if COLLECTION_NAME in [c.name for c in client.list_collections()]:
#     client.delete_collection(COLLECTION_NAME)
# collection = client.create_collection(COLLECTION_NAME)

# def encode(text):
#     text = "passage: " + text
#     with torch.no_grad():
#         return embedder.encode(text, convert_to_numpy=True).astype("float32")

# # إدخال البيانات من ملف الأسئلة
# with open("actions/faq_finetune.jsonl", "r", encoding="utf-8") as f:
#     for i, line in enumerate(f):
#         item = json.loads(line.strip())
#         q, a = item.get("instruction", ""), item.get("response", "")
#         if q and a:
#             emb = encode(a).tolist()
#             collection.add(
#                 documents=[a],
#                 embeddings=[emb],
#                 metadatas=[{"question": q}],
#                 ids=[f"faq_{i}"]
#             )

# print(f"✅ FAQ: تم إدخال {i+1} سؤال.")





# # import os, json, torch, chromadb
# # from tqdm import tqdm
# # from docx import Document
# # from sentence_transformers import SentenceTransformer

# # # تحميل موديل التضمين
# # embedder = SentenceTransformer("intfloat/multilingual-e5-large")

# # # إنشاء ChromaDB
# # client = chromadb.PersistentClient(path="chroma_db")

# # # حذف الكوليكشن لو موجودة
# # COLLECTION_NAME = "faq_only"
# # if COLLECTION_NAME in [c.name for c in client.list_collections()]:
# #     client.delete_collection(COLLECTION_NAME)

# # collection = client.create_collection(COLLECTION_NAME)

# # def encode(text):
# #     with torch.no_grad():
# #         return embedder.encode("passage: " + text, convert_to_numpy=True).astype("float32")

# # # ✅ إدخال الأسئلة
# # with open("actions/faq_finetune.jsonl", "r", encoding="utf-8") as f:
# #     for i, line in enumerate(f):
# #         item = json.loads(line.strip())
# #         q, a = item.get("instruction", ""), item.get("response", "")
# #         if q and a:
# #             emb = encode(a).tolist()
# #             collection.add(
# #                 documents=[a],
# #                 embeddings=[emb],
# #                 metadatas=[{"source": "faq", "question": q}],
# #                 ids=[f"faq_{i}"]
# #             )

# # # ✅ إدخال اللائحة
# # doc = Document("actions/AI_Bylaw.docx")
# # paras = [p.text.strip() for p in doc.paragraphs if len(p.text.strip()) > 30]

# # for j, para in enumerate(paras):
# #     emb = encode(para).tolist()
# #     collection.add(
# #         documents=[para],
# #         embeddings=[emb],
# #         metadatas=[{"source": "bylaw"}],
# #         ids=[f"bylaw_{j}"]
# #     )

# # print(f"✅ تم إدخال {i+1} من الأسئلة، و{len(paras)} من فقرات اللائحة.")



import json, torch, chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# تحميل موديل التضمين
embedder = SentenceTransformer("intfloat/multilingual-e5-large")

# إعداد قاعدة البيانات
client = chromadb.PersistentClient(path="chroma_db")

# إنشاء أو استرجاع التجميعة
COLLECTION_NAME = "faq_only"
if COLLECTION_NAME in [c.name for c in client.list_collections()]:
    client.delete_collection(COLLECTION_NAME)
collection = client.create_collection(COLLECTION_NAME)

def encode(text):
    text = "passage: " + text  # تنسيق E5
    with torch.no_grad():
        return embedder.encode(text, convert_to_numpy=True).astype("float32")

# إدخال البيانات من ملف الأسئلة
with open("actions/faq_finetune.jsonl", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        item = json.loads(line.strip())
        q, a = item.get("instruction", ""), item.get("response", "")
        if q and a:
            combined_text = f"السؤال: {q}\nالإجابة: {a}"
            emb = encode(combined_text).tolist()
            collection.add(
                documents=[combined_text],
                embeddings=[emb],
                metadatas=[{"question": q}],
                ids=[f"faq_{i}"]
            )

print(f"✅ تم إدخال {i+1} عنصر في قاعدة البيانات.")
