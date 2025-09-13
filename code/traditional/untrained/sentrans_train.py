import json
import os
from sentence_transformers import SentenceTransformer, InputExample, losses, models, util
from torch.utils.data import DataLoader

EPOCH = 150
BATCH = 32

train = []
with open("data/train_data_sen.json", "r") as f:
    datas = json.load(f)

for data in datas:
    train.append(
        InputExample(
            texts=[data["query"], data["positive"]["context"]],  # Positive pair
            label=1.0
        )
    )
    train.append(
        InputExample(
            texts=[data["query"], data["negative"]["context"]],  # Negative pair
            label=0.0
        )
    )

train_dataloader = DataLoader(train, shuffle=True, batch_size=BATCH)
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

train_loss = losses.CosineSimilarityLoss(model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=EPOCH,
    show_progress_bar=True
)

model.save(f"checkpoints/model_{EPOCH}")
