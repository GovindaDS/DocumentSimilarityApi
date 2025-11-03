# main.py

from fastapi import FastAPI, UploadFile, File
import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from utils import extract_text_from_file, get_document_embedding

app = FastAPI()

# Load model and data
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = np.load("embeddings.npy")
with open("ids.json", "r") as f:
    ids = json.load(f)

model1 = SentenceTransformer('all-MiniLM-L6-v2')
embeddings1 = np.load("embeddings.npy")
with open("ids.json", "r") as f:
    ids1 = json.load(f)

@app.post("/check-similar-documents")
async def check_similar_documents(file: UploadFile = File(...), threshold: float = 0.85):
    ext = os.path.splitext(file.filename)[1]
    temp_file = f"temp{ext}"
    with open(temp_file, "wb") as f:
        f.write(await file.read())

    try:
        # Extract and embed the uploaded document
        text = extract_text_from_file(temp_file)
        doc_vector = get_document_embedding(text, model)

        # Compute similarity
        sims = cosine_similarity([doc_vector], embeddings)[0]

        # Find all matches above threshold
        results = []
        for idx, score in enumerate(sims):
            if score >= threshold:
                results.append({
                    "matched_document_id": ids[idx],
                    "similarity_score": float(score)
                })

        return {
            "total_matches": len(results),
            "similar_documents": results
        }

    except Exception as e:
        return {"error": str(e)}
    finally:
        os.remove(temp_file)
