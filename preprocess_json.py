import os
import json
import time
import pandas as pd
import joblib
import google.generativeai as genai

GEMINI_API_KEY = ""
genai.configure(api_key=GEMINI_API_KEY)

def create_embedding(text_list):
    embeddings = []
    for text in text_list:
        result = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document"
        )
        embeddings.append(result["embedding"])
        time.sleep(0.1)
    return embeddings

jsons = os.listdir("newjsons")
my_dicts = []
chunk_id = 0

for json_file in jsons:
    with open(f"newjsons/{json_file}", encoding="utf-8") as f:
        content = json.load(f)

    print(f"Creating embeddings for: {json_file}")
    texts = [c["text"] for c in content["chunks"]]
    embeddings = create_embedding(texts)

    for i, chunk in enumerate(content["chunks"]):
        chunk["chunk_id"] = chunk_id
        chunk["embedding"] = embeddings[i]
        chunk_id += 1
        my_dicts.append(chunk)

df = pd.DataFrame.from_records(my_dicts)
joblib.dump(df, "embeddings.joblib")
print(f"Done. Total chunks embedded: {chunk_id}")
print("Saved: embeddings.joblib")