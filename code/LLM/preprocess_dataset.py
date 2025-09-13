import json
import random

"""
1. Load Data
"""
train_file = 'data/train_data.jsonl'
dev_file = 'data/dev_data.jsonl'

with open(train_file, 'r', encoding='utf-8') as f:
    train_data = [json.loads(line) for line in f]

if len(train_data) < 175:
    raise ValueError("Train dataset has less than 175 samples.")

"""
2. Split Data and Save
"""
random.shuffle(train_data)
moved_data = train_data[:175]
remaining_data = train_data[175:]

with open(dev_file, 'w', encoding='utf-8') as f:
    for item in moved_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

with open(train_file, 'w', encoding='utf-8') as f:
    for item in remaining_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

"""
3. Turn train.jsonl to train.json (for train.py)
"""
new = []
for i in remaining_data:
    new.append({"id": i["id"], "instruction": str(i["title"]) +'\n'+ str(i["question"]), "output": i["label"]})
    print(new[-1])

with open("data/train_data.json", "w", encoding="utf-8") as f:
    json.dump(new, f, ensure_ascii=False)