import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import csv
from tqdm import tqdm

def normalize(embeddings):
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

EPOCH = 150
model = SentenceTransformer(f"checkpoints/model_{EPOCH}")

with open("law.json", "r") as f:
    datas = json.load(f)

ids = []
context = []
for data in datas:
    ids.append(data["id"])
    context.append(data["context"])

context_emb = model.encode(context)
context_emb = normalize(context_emb)
# context_np = context_emb.cpu().numpy()

# dimension = context_np.shape[1]
# index = faiss.IndexFlatL2(dimension)
# index.add(context_np)

# faiss.write_index(index, "law_index.faiss")

K = 3
with open("test_data_sen.json", "r") as f:
    datas = json.load(f)

results = [["id", "TARGET"]]
score = []
for data in tqdm(datas):
    query = data["query"]
    query_emb = model.encode(query)
    query_emb = normalize(query_emb.reshape(1, -1))
    # query_np = query_emb.cpu().numpy().reshape(1, -1)

    sim = np.dot(query_emb, context_emb.T).flatten()
    topk = np.argsort(sim)[::-1][:K]

    # distances, indices = index.search(query_np, K)
    retrieved = []
    for i in topk:
        if sim[i] < 0.8:
            break
        retrieved.append(ids[i])
        score.append(sim[i])

    results.append([data["id"], ",".join(retrieved)])

with open("results/sentecetrans.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(results)

print(len(score))
print(max(score))
print(min(score))
