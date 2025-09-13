import json
# convert jsonl to json
data = []
with open('train_data.jsonl', 'r') as jsonl_file:
    for line_number, line in enumerate(jsonl_file, start=1):
        line = line.strip()
        if not line:  # Skip empty lines
            continue
        try:
            data.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON on line {line_number}: {e}")
            continue

with open('train_data.json', 'w') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)

data = []
with open('test_data.jsonl', 'r') as jsonl_file:
    for line_number, line in enumerate(jsonl_file, start=1):
        line = line.strip()
        if not line:  # Skip empty lines
            continue
        try:
            data.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON on line {line_number}: {e}")
            continue

with open('test_data.json', 'w') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)