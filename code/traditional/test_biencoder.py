import os
import json
from os.path import abspath, join

import torch
import pandas as pd

from utils.data import Test_BSARDataset
from utils.eval import BiEncoderTester
from models.trainable_dense_models import BiEncoder

import csv

if __name__ == '__main__':
    # 1. Load an already-trained BiEncoder.
    checkpoint_path = abspath(join(__file__, "training/Dec16-21-38-03/best")) #checkpoint path
    model = BiEncoder.load(checkpoint_path)

    # 2. Load the test set.
    test_queries_df = pd.read_csv(abspath(join(__file__, "data/test_data.csv")))  #data/test_data_gpt.csv
    documents_df = pd.read_csv(abspath(join(__file__, "data/law.csv")))  #data/law_gpt.csv
    test_dataset = Test_BSARDataset(test_queries_df, documents_df)

    # 3. Initialize the Evaluator.
    tester = BiEncoderTester(queries=test_dataset.queries, 
                                documents=test_dataset.documents, 
                                score_fn=model.score_fn)

    # 4. Run trained model and get results
    predictions = tester(model=model,
                       device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
                       batch_size=512)

    # 5. Save results.
    os.makedirs(checkpoint_path, exist_ok=True)
    with open(join(checkpoint_path, 'predictions.csv'), 'w') as fOut:
        writer = csv.writer(fOut)
        writer.writerow(["id", "TARGET"])
        writer.writerows(predictions)
