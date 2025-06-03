import json
import os
import numpy as np
import sqlite3
#from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import boto3
from dotenv import load_dotenv
from TitanEmbeddings import TitanEmbeddings, generate_titan_vector_embedding

load_dotenv()
aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_DEFAULT_REGION")

aws_client = boto3.client(
    "bedrock-runtime",
    region_name=aws_region,
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key
)

# Load your threads
file_paths = [
    "studyDropshipping/general-chat.json",
    "studyDropshipping/scripts_json_format/script_autods.json",
    "studyDropshipping/scripts_json_format/script_ch4.json",
    "studyDropshipping/scripts_json_format/script_ch7.json",
    "studyDropshipping/scripts_json_format/script_ch8.json",
    "studyDropshipping/scripts_json_format/script_ch8_2.json",
    "studyDropshipping/scripts_json_format/script_ch9.json",
    "studyDropshipping/scripts_json_format/script_ch10.json",
    "studyDropshipping/scripts_json_format/script_final_build.json",
    "studyDropshipping/scripts_json_format/script_shopify_build.json"

]

os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

model = TitanEmbeddings(boto3_client=aws_client)
all_embeddings = []
id_map = []

conn = sqlite3.connect("data/threads.db", timeout=10.0)  # single connection
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS threads (
    id TEXT PRIMARY KEY,
    header TEXT,
    content TEXT,
    category TEXT,
    source TEXT
)
""")


try:
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            header = item.get("header") or item.get("question")
            content = item.get("content") or item.get("full_thread_answer")
            category = item.get("category", "Unknown")
            source = item.get("source", os.path.basename(file_path))
            thread_id = item["id"]

            if not (thread_id and content and header):
                print(f"Skipping incomplete item: {thread_id}")
                continue

            # 💡 Avoid locking the DB while calling Titan
            embedding = generate_titan_vector_embedding(content)

            all_embeddings.append(embedding)
            id_map.append(thread_id)

            # 💾 Write to DB after embedding is ready
            cursor.execute("""
                INSERT OR REPLACE INTO threads (id, header, content, category, source)
                VALUES (?, ?, ?, ?, ?)
            """, (thread_id, header, content, category, source))

    conn.commit()
finally:
    conn.close()


# Normalize and save embeddings
all_embeddings = normalize(np.array(all_embeddings))
np.save("models/embeddings.npy", all_embeddings)

with open("models/id_map.txt", "w") as f:
    f.write("\n".join(map(str, id_map)))

print(f"Done. Saved {len(id_map)} threads.")
