import re
import json

# File path
file_path = "/Users/williamkapner/Downloads/CHAPTER 9 - TikTok Ads.txt"
output_path = "script_ch9.json"

# Step 1: Clean line endings and normalize spacing
def preprocess_script(text):
    text = re.sub(r'\r\n?', '\n', text)  # Normalize line endings
    text = re.sub(r'\n{3,}', '\n\n', text.strip())  # Collapse excessive blank lines
    return text

# Step 2: Chunk based on headers and paragraphs
def chunk_script(text):
    lines = text.splitlines()
    chunks = []
    current_header = None
    current_body = []
    chunk_id = 1

    def is_header(line):
        return line.strip() != "" and line.strip().istitle()

    for line in lines:
        stripped = line.strip()

        if is_header(stripped):
            if current_header and current_body:
                chunks.append({
                    "chunk_id": chunk_id,
                    "header": current_header,
                    "content": "\n".join(current_body).strip(),
                    "category": 'TikTok Ads',
                    "source": 'course'
                })
                chunk_id += 1
                current_body = []
            current_header = stripped
        elif current_header:
            current_body.append(line)

    # Final chunk
    if current_header and current_body:
        chunks.append({
            "chunk_id": chunk_id,
            "header": current_header,
            "body": "\n".join(current_body).strip()
        })

    return chunks

# Step 3: Format chunks into the required structure
def format_chunks(chunked, category="TikTok Ads", source="course"):
    formatted_chunks = []
    for i, chunk in enumerate(chunked, start=1):
        formatted_chunk = {
            "id": i,
            "header": chunk["header"],
            "content": chunk["body"],
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
