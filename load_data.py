import faiss
import sqlite3
import numpy as np
import json
from sentence_transformers import SentenceTransformer

# Thread data (paste yours here or load from JSON)
# Path to your JSON file
file_path = "/Users/dan/calpoly/BusinessAnalytics/GSB570GENAI/studyDropshipping/qa_threads.json"

# Open and load the JSON data
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

model = SentenceTransformer('all-MiniLM-L6-v2')
dimension = 384
index = faiss.IndexFlatL2(dimension)
id_map = []

conn = sqlite3.connect("data/threads.db")
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS threads (
    id TEXT PRIMARY KEY,
    question TEXT,
    full_answer TEXT,
    category TEXT
)
""")

for item in data:
    emb = model.encode(item["full_thread_answer"])
    index.add(np.array([emb]))
    id_map.append(item["id"])

    cursor.execute("INSERT OR REPLACE INTO threads VALUES (?, ?, ?, ?)", (
        item["id"], item["question"], item["full_thread_answer"], item["category"]
    ))

conn.commit()
conn.close()

faiss.write_index(index, "models/faiss_index.index")
with open("models/id_map.txt", "w") as f:
    f.write("\n".join(id_map))
