import os
import json
from os.path import abspath, join

import torch
import pandas as pd

from utils.data import BSARDataset
from utils.eval import BiEncoderEvaluator
from models.trainable_dense_models import BiEncoder



if __name__ == '__main__':
    # 1. Load an already-trained BiEncoder.
    checkpoint_path = abspath(join(__file__, "training/Dec16-21-38-03/best")) 
    model = BiEncoder.load(checkpoint_path)

    # 2. Load the test set.
    test_queries_df = pd.read_csv(abspath(join(__file__, "data/train_data.csv"))) # "data/train_data_gpt.csv"
    documents_df = pd.read_csv(abspath(join(__file__, "data/law_data.csv")))  # "data/law_gpt.csv"
    test_dataset = BSARDataset(test_queries_df, documents_df)

    # 3. Initialize the Evaluator.
    evaluator = BiEncoderEvaluator(queries=test_dataset.queries, 
                                   documents=test_dataset.documents, 
                                   relevant_pairs=test_dataset.one_to_many_pairs, 
                                   score_fn=model.score_fn)

    # 4. Run trained model and compute scores.
    scores = evaluator(model=model,
                       device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
                       batch_size=512)

    # 5. Save results.
    os.makedirs(checkpoint_path, exist_ok=True)
    with open(join(checkpoint_path, 'test_scores_fixed.json'), 'w') as fOut:
        json.dump(scores, fOut, indent=2)
