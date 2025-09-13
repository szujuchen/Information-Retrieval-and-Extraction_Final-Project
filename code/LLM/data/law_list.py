import json
import re

with open("law.json", "r", encoding="utf-8") as f:
    law_list = json.load(f)

converted_list = []
for item in law_list:
    # Use regex to find and replace patterns like '第906-4條' with '第906條之4'
    converted_item = re.sub(r'第(\d+)-(\d+)條', r'第\1條之\2', item)
    converted_list.append(converted_item)

with open("law_list.json", "w", encoding="utf-8") as f:
    json.dump(converted_list, f, ensure_ascii=False)