import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import HfFolder
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from transformers import BitsAndBytesConfig
import pandas as pd
import csv

bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_compute_dtype = torch.bfloat16
)

"""
1. Load Data
"""
dataset_path = 'data/test_data_gpt.json'
with open(dataset_path, 'r', encoding='utf-8') as f:
    dataset = json.load(f)
HfFolder.save_token("") # fill your hugging face user token if needed

"""
2. Parameters
"""
MODEL_num = 10 
CURRENT_checkpoint = 105 
MODEL_checkpoint = 0 # 要 infer 的 model 之間的 steps 差
MODEL_dir = "model/checkpoint-" # directory path containing model

"""
3. Inference Loop
"""
for i in range(MODEL_num):
    # 3-1. Load Model
    MODEL_path = MODEL_dir + str(CURRENT_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_path)
    model = AutoModelForCausalLM.from_pretrained(MODEL_path,
                                                torch_dtype=torch.bfloat16,
                                                quantization_config=bnb_config
                                                )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # 3-2. Ask Model
    all_predictions = []
    for item in dataset:

        input_text = f"作為法律達人，你的任務是告訴我可以從臺灣的哪個法條得到相關問題的答案。\n\n### 問題：\n{item['instruction']}\n\n### 回應：你可以參考"
        inputs = tokenizer(input_text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        output = model.generate(inputs["input_ids"], max_new_tokens=110, attention_mask=inputs["attention_mask"],  num_return_sequences=1, temperature=0.7, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(output[0], skip_special_tokens=True)

        try:
            answer = response[len(input_text):]
            answer = answer.split("\n")[0].strip()
            prediction = answer.split(",")  
        except IndexError:
            prediction = []

        all_predictions.append(answer)
    
    # 3-3. Save result
    ids = [item.get('id', None) for item in dataset]

    output_data = {'id': ids, 'TARGET': all_predictions}
    df = pd.DataFrame(output_data)

    output_path = 'result/checkpoint_' + str(CURRENT_checkpoint) + '_' + str(i) + '.csv'
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    CURRENT_checkpoint += MODEL_checkpoint