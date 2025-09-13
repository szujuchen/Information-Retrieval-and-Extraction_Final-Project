import csv
import json
# original
with open("law.json", "r") as f:
    laws = json.load(f)

with open("law.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "article"])

    for law in laws:
        writer.writerow([law["id"], f'{law["id"]}：{law["context"]}'])
#gpt
with open("law_gpt.json", "r") as f:
    laws = json.load(f)

with open("law_gpt.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "article"])

    for law in laws:
        char = law["title"][-1]
        ext = law["extend"].replace("\n\n", "\n")
        if law["title"] in ext:
            query = ext
        elif '\u3040' <= char <= '\u30FF' or '\u4E00' <= char <= '\u9FFF':
            query = f'{law["title"]}。 {ext}'
        else:
            query = f'{law["title"]} {ext}'
        writer.writerow([law["id"], str(query)])