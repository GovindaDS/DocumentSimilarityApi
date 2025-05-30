# Save this in train_model.py
from sentence_transformers import SentenceTransformer
import numpy as np
import json
from utils import get_document_embedding

documents = [
    {"id": "doc1", "content": "This is a document about car, automobile, and motor car all being similar."},
    {"id": "doc2", "content": "A comprehensive guide to NLP techniques for long documents."},
    {"id": "doc3", "content": "The client fully paid the invoice for the services provided."}
]

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = []
ids = []

for doc in documents:
    emb = get_document_embedding(doc["content"], model)
    embeddings.append(emb)
    ids.append(doc["id"])

np.save("embeddings.npy", np.array(embeddings))
with open("ids.json", "w") as f:
    json.dump(ids, f)
