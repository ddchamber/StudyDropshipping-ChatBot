import re
import json

# File path
file_path = "/Users/williamkapner/Downloads/CHAPTER 9 - TikTok Ads.txt"
output_path = "script_ch9.json"

def preprocess_script(text):
    text = re.sub(r'\r\n?', '\n', text)  # Normalize line endings
    text = re.sub(r'\n{3,}', '\n\n', text.strip())  # Collapse excessive blank lines
    return text

def chunk_script(text):
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    current_header = "General"  # Fallback if no header
    chunk_id = 1

    for para in paragraphs:
        # Heuristic: header = short and contains no punctuation (not a sentence)
        if len(para.split()) <= 10 and not re.search(r'[.!?]', para):
            current_header = para
        else:
            chunks.append({
                "id": chunk_id,
                "header": current_header,
                "content": para,
                "category": 'TikTok Ads',
                "source": 'course'
            })
            chunk_id += 1
    return chunks


# Step 3: Format chunks into the required structure
def format_chunks(chunked, category="TikTok Ads", source="course"):
    formatted_chunks = []
    for i, chunk in enumerate(chunked, start=1):
        formatted_chunk = {
            "id": i,
            "header": chunk["header"],
            "content": chunk["content"],
            "category": category,
            "source": source
        }
        formatted_chunks.append(formatted_chunk)
    return formatted_chunks

# Step 4: Save to JSON
def save_to_json(chunks, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved {len(chunks)} chunks to {path}")

# Step 5: Preview
def print_sample_chunks(chunks, n=5):
    for chunk in chunks[:n]:
        print(json.dumps(chunk, indent=2))
        print("-" * 50)

# -------- Main Execution --------
with open(file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

processed_text = preprocess_script(raw_text)
chunked = chunk_script(processed_text)
formatted_chunks = format_chunks(chunked)
save_to_json(formatted_chunks, output_path)
print_sample_chunks(formatted_chunks)
