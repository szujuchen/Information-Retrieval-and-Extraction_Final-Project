import json
import random

with open("law.json", "r") as f:
    laws = json.load(f)

with open("train_data.json", "r") as f:
    traindata = json.load(f)

train_dataset = []
for data in traindata:
    if data["question"]:
       query = data["title"] + " " + data["question"]
    else:
        query = data["title"]

    relevant_article_ids = data["label"].split(',')
    relevant_contexts = [article["context"] for article in laws if article["id"] in relevant_article_ids]
    pos = []
    neg = []
    for id, context in zip(relevant_article_ids, relevant_contexts):
        # Select negative articles (randomly choose an article that is not relevant)
        negative_articles = [article for article in laws if article["id"] not in relevant_article_ids]
        negative_article = random.choice(negative_articles)

        pos.append({
            "id": id,
            "context": context
        })

        neg.append({
            "id": negative_article["id"],
            "context": negative_article["context"]
        })
        
    # Append the query with its relevant and negative contexts
    for p, n in zip(pos, neg):
        train_dataset.append({
            "query": query,
            "title": data["title"],
            "question": data["question"],
            "positive": p,
        })

with open("train_data_sen.json", "w") as f:
    json.dump(train_dataset, f, indent=2, ensure_ascii=False)

    