import os, chromadb
from docx import Document
from sentence_transformers import SentenceTransformer

# تحميل موديل التضمين
embedder = SentenceTransformer("intfloat/multilingual-e5-large")


# إعداد قاعدة البيانات
client = chromadb.PersistentClient(path="chroma_db")

# حذف وإنشاء التجميعة
COLLECTION_NAME = "bylaw_only"
if COLLECTION_NAME in [c.name for c in client.list_collections()]:
    client.delete_collection(COLLECTION_NAME)
collection = client.create_collection(COLLECTION_NAME)

# قراءة الفقرات من ملف الوورد
doc = Document("actions/AI_Bylaw.docx")
paras = [p.text.strip() for p in doc.paragraphs if len(p.text.strip()) > 40]

# إدخال الفقرات والـ embeddings
embeddings = embedder.encode(paras).tolist()
collection.add(
    documents=paras,
    embeddings=embeddings,
    metadatas=[{"source": "bylaw"}] * len(paras),
    ids=[f"bylaw_{i}" for i in range(len(paras))]
)

print(f"✅ Bylaw: {len(paras)} فقرة مدخلة.")
