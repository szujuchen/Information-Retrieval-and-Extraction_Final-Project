import json
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from torch.nn.functional import cosine_similarity
from tqdm import tqdm
import csv
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

with open("data/test_data.json", "r") as f:
    queries = json.load(f)

with open("data/law.json", "r") as f:
    laws = json.load(f)

MAX_L = 512
MODEL_PATH = 'hfl/chinese-roberta-wwm-ext'
DEVICE = "cuda"
TOP_N = 3

class BiEncoder(nn.Module):
    def __init__(self, model_name):
        super(BiEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.fc = nn.Linear(self.bert.config.hidden_size, 768)  # Project to same size of embedding space

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # CLS token output
        embedding = self.fc(pooled_output)  # Get embeddings
        return embedding

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BiEncoder(MODEL_PATH).to(DEVICE)

law_emb = []
with torch.no_grad():
    for article in tqdm(laws):
        article_encoding = tokenizer(article["context"], truncation=True, padding='max_length', max_length=MAX_L, return_tensors="pt").to(DEVICE)
        article_input_ids = article_encoding["input_ids"]
        article_attention_mask = article_encoding["attention_mask"]
        
        article_embedding = model(article_input_ids, article_attention_mask)
        law_emb.append(article_embedding)

    def inference(q, max_length=512):
        query_encoding = tokenizer(q, truncation=True, padding='max_length', max_length=max_length, return_tensors="pt").to(DEVICE)
        query_input_ids = query_encoding["input_ids"]
        query_attention_mask = query_encoding["attention_mask"]
        query_embedding = model(query_input_ids, query_attention_mask)

        similarities = []
        for law, emb in zip(laws, law_emb):
            sim_score = cosine_similarity(query_embedding, emb)
            similarities.append((law["id"], sim_score.item()))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return [cand[0] for cand in similarities[:TOP_N]], similarities[0][1], similarities[TOP_N-1][1]

    model.eval()
    results = [["id", "TARGET"]]

    for query in tqdm(queries):
        relevant, max_sim, min_sim = inference(query["query"])
        results.append([query["id"], ",".join(relevant), max_sim, min_sim])

    with open("results/roberta.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(results)
