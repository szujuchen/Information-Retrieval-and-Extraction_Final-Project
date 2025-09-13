import csv
import json
import pandas as pd

#origianl
with open("test_data.json", "r") as f:
    trains = json.load(f)

with open("test_data.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "question", "article_ids"])

    for train in trains:
        if train["question"]:
            query = train["title"]+" "+train["question"]
        else:
            query = train["title"]

        ans = ""
        writer.writerow([train["id"], str(query), ans])

#gpt
with open("test_data_gpt.json", "r") as f:
    trains = json.load(f)

with open("test_data_gpt.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "question", "article_ids"])

    for train in trains:
        if train["question"]:
            query = train["title"]+" "+train["question"]
        else:
            query = train["title"]

        ans = ""
        writer.writerow([train["id"], str(query), ans])
