import json
import random
# train
with open("../data/train_raw_gpt.json") as f:
    data = json.load(f)
    
new = []
for i in data:
    char = i["title"][-1]
    ext = i["extend"].replace("\n\n", "\n")
    if i["title"] in ext:
        query = ext
    elif '\u3040' <= char <= '\u30FF' or '\u4E00' <= char <= '\u9FFF':
        query = f'{i["title"]}。 {ext}'
    else:
        query = f'{i["title"]} {ext}'

    new.append({"id": i["id"], "instruction": query, "output": i["label"]})
    # print(new[-1])

random.shuffle(new)
train = new[:1000]
valid = new[1000:]
with open("../data/train_data_gpt.json", "w") as f:
    json.dump(train, f, ensure_ascii=False, indent=2)

with open("../data/dev_data_gpt.json", "w") as f:
    json.dump(valid, f, ensure_ascii=False, indent=2)

# test
with open("../data/test_raw_gpt.json") as f:
    data = json.load(f)
    
new = []
for i in data:
    char = i["title"][-1]
    ext = i["extend"].replace("\n\n", "\n")
    if i["title"] in ext:
        query = ext
    elif '\u3040' <= char <= '\u30FF' or '\u4E00' <= char <= '\u9FFF':
        query = f'{i["title"]}。 {ext}'
    else:
        query = f'{i["title"]} {ext}'

    new.append({"id": i["id"], "instruction": query})
    # print(new[-1])

with open("test_data_gpt.json", "w") as f:
    json.dump(new, f, ensure_ascii=False, indent=2)