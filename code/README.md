# 2024-Fall IRIE final project-Team 17
## OS / GPU 
CSIE Workstation Linux \
NVIDIA GeForce RTX 4090 / 3090

## Python Version
python=3.9

## Requirements
```
torch
transformers
pandas
tqdm
huggingface_hub
scikit-learn
bitsandbytes
peft
```
(if any package missing, please pip install !)

## Traditional Method (`traditional/`)
### Step 1. **Data Preprocess**
  - Law: \
  Put the `law.zip` under `law/` and run `preprocess.py` to get a parsed law json file that matches each law id and law context.
  - Data:
    1. Put the `train_data.jsonl` and `test_data.jsonl` under `data/` and run `jsonltojson.py` to convert files to json format.
    2. Run the `dataset_law_csv.py`, `dataset_train_csv.py` and `dataset_test_csv.py` to convert the json to csv format for later training and inference.

### Step 2. **Train Tradition Model**
We train the traditional model by using BiEncoder architecture. 
Modify the training arguments at `train_biencoder.py #278` and run `python train_encoder.py` to start the training.

### Step 3. **Inference**
Modify the checkpoint path at `test_biencoder.py #16` and run `python test_biencoder.py`.
The prediciton file will be save at the checkpoint directory.

### Step 4. **Postprocess**
Modify the output and input file path at `postprocess.py #4 #7`.
The postprocess will modify the law id if it is shown as "XXX法MM-N條" to "XXX法MM條之N" 

## LLM (`LLM/`)

### Step 1. Data Preprocess - `Preprocess_data.py`

1. **Load Data**: The training dataset (`train_data.jsonl`) is loaded, ensuring it contains at least 175 samples.
2. **Data Splitting**: Randomly select 175 samples from the training data and move them to the development dataset (`dev_data.jsonl`). The remaining 1000 data stays in `train_data.jsonl`.
3. **Format Conversion**: Convert the remaining training data into `train_data.json` format, including fields for ID, instruction (title + question), and output (label), for compatibility with `train.py`.

### Step 2. Fine-Tune LLM - Use `train.sh` to run    `train.py`

We fine-tuned the LLM using [QLoRA](https://github.com/artidoro/qlora) with the following modifications:

- **Model**: `yentinglin/Llama-3-Taiwan-8B-Instruct` was used as the base model.
- **Dataset**: Fine-tuning was performed on `train_data.json` in the Alpaca format.
- **Prompt**: 
>     作為法律達人，你的任務是告訴我可以從臺灣的哪個法條得到相關問題的答案。
>     ### 問題：{instruction}
>     ### 回應：你可以參考
- **Training Parameters**: Adjusted key hyperparameters for efficient training:
  - **Learning Rate**: As default, `2e-4` 
  - **Precision**: 4-bit quantization (`bits=4`) with BF16 support.
  - **Batching**: Gradient accumulation with a per-device batch size of 4.
  - **Steps**: 200 max steps, saving checkpoints every 20 steps.
  - **Scheduler**: Constant learning rate scheduler.

Run the script using the `train.sh` command.

### Setp3. Evaluate Saved Model - `validation.py`

- **Dataset**: Evaluation was conducted on `dev_data.jsonl` by **`Step 1`**.
- **Model Checkpoints**: Iteratively loaded checkpoints (`checkpoint-15`, `checkpoint-30`, ..., `checkpoint-300`) for evaluation.
- **Evaluation Metrics**: Precision, Recall, and F1 score were computed using the model's responses compared to ground truth labels.

The script provides metrics for each checkpoint to identify the best-performing model.

### Step 4. Generate Predictions - `infer.py`


- **Dataset**: Inference is performed on `test_data.jsonl`.
- **Model Checkpoints**: Specify parameters to infer multiple models (`MODEL_num`) or the same model multiple times by adjusting the `CURRENT_checkpoint` and `MODEL_checkpoint` values.
- **Process**:
  1. Load the specified model checkpoint.
  2. Format the input prompt with the title and question from the dataset.
  3. Generate model predictions using the `generate` function.
  4. Save results (`id` and `TARGET`) as CSV files for each checkpoint.

The output CSV files are saved in the `result` directory, named based on the corresponding checkpoint and iteration.

### Step 5. Voting and Filtering - `postprocess_predict.py`

The final step involves aggregating predictions from multiple inference runs to create a robust submission:

1. **Voting**:
   - Combine results from multiple checkpoints or runs (`checkpoint_*.csv`) using a majority voting mechanism.
   - Default: Predictions made in more than 11 out of 20 runs are retained.

2. **Validation Against Law List**:
   - Filter the voted results against a predefined list of valid law IDs (`law_list.json`).
   - Remove invalid predictions and retain only valid law references.

The final submission is stored as a CSV file containing the `id` and the filtered `TARGET` for each test sample.