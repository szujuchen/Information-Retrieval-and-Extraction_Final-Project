import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import HfFolder
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_compute_dtype = torch.bfloat16
    )

def calculate_f1_score(true_labels, pred_labels):
    precision, recall, f1 = 0, 0, 0
    for true, pred in zip(true_labels, pred_labels):
        true_set, pred_set = set(true), set(pred)
        common = true_set & pred_set
        if pred_set:
            precision += len(common) / len(pred_set)
        if true_set:
            recall += len(common) / len(true_set)
    precision /= len(true_labels)
    recall /= len(true_labels)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

dataset_path = 'data/dev_data_gpt.json'
with open(dataset_path, 'r', encoding='utf-8') as f:
    dataset = json.load(f)

HfFolder.save_token("")
MODEL_num = 20
MODEL_checkpoint = 15
MODEL_dir = "model/checkpoint-"

for i in range(1, MODEL_num+1):
    MODEL_path = MODEL_dir + str(MODEL_checkpoint * i)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_path)
    model = AutoModelForCausalLM.from_pretrained(MODEL_path,
                                                torch_dtype=torch.bfloat16,
                                                quantization_config=bnb_config
                                                )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    all_labels = []
    all_predictions = []

    for item in dataset:
        label = item['label'].split(",")  
        all_labels.append(label)

        input_text = f"作為法律達人，你的任務是告訴我可以從臺灣的哪個法條得到相關問題的答案。\n\n### 問題：\n{item['instruction']}\n\n### 回應：你可以參考"
        inputs = tokenizer(input_text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        output = model.generate(inputs["input_ids"], max_new_tokens=100, attention_mask=inputs["attention_mask"],  num_return_sequences=1, temperature=0.7, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(output[0], skip_special_tokens=True)

        try:
            answer = response[len(input_text):]
            answer = answer.split("\n")[0].strip()
            prediction = answer.split(",")  
        except IndexError:
            prediction = []

        all_predictions.append(prediction)
        
    precision, recall, f1 = calculate_f1_score(all_labels ,all_predictions)
    print(f"path: {MODEL_path}, precision: {precision}, recall: {recall}, f1: {f1}")