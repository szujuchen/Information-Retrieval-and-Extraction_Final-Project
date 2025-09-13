import json
import random

with open("test_data.json", "r") as f:
    traindata = json.load(f)

train_dataset = []
for data in traindata:
    if data["question"]:
       query = data["title"] + " " + data["question"]
    else:
        query = data["title"]
        
    # Append the query with its relevant and negative contexts
    train_dataset.append({
        "id": data["id"],
        "query": query,
        "title": data["title"],
        "question": data["question"],
    })

with open("test_data_sen.json", "w") as f:
    json.dump(train_dataset, f, indent=2, ensure_ascii=False)

    