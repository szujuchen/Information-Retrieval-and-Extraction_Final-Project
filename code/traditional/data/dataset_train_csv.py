import csv
import json
import pandas as pd

#original
with open("train_data.json", "r") as f:
    trains = json.load(f)

laws = pd.read_csv("law_dataset.csv", dtype={"id": str, "article": str})
laws_id = laws["id"].values.tolist()

with open("train_data.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "question", "article_ids"])

    for train in trains:
        query = train["title"]
        if train["question"]:
            query += f" {train['question']}"

        ans = ""
        for a in train["label"].split(","):
            if "條之" in a:
                chopped = a.split("條之")
                name = chopped[0] + "-" + chopped[1] + "條"
                
            else:
                name = a

            if name not in laws_id:
                continue
            ans += name
            ans += ","
        ans = ans[:-1]

        if ans == "":
            continue
        writer.writerow([train["id"], str(query), ans])

#gpt
with open("train_data_gpt.json", "r") as f:
    trains = json.load(f)

laws = pd.read_csv("law_dataset.csv", dtype={"id": str, "article": str})
laws_id = laws["id"].values.tolist()

with open("train_data_gpt.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "question", "article_ids"])

    for train in trains:
        char = train["title"][-1]
        ext = train["extend"].replace("\n\n", "\n")
        if train["title"] in ext:
            query = ext
        elif '\u3040' <= char <= '\u30FF' or '\u4E00' <= char <= '\u9FFF':
            query = f'{train["title"]}。 {ext}'
        else:
            query = f'{train["title"]} {ext}'

        ans = ""
        for a in train["label"].split(","):
            if "條之" in a:
                chopped = a.split("條之")
                name = chopped[0] + "-" + chopped[1] + "條"
                
            else:
                name = a

            if name not in laws_id:
                continue
            ans += name
            ans += ","
        ans = ans[:-1]

        if ans == "":
            continue
        writer.writerow([train["id"], str(query), ans])



