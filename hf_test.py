from sklearn.model_selection import train_test_split
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

test = pd.read_csv("data/test.csv")
test = test[["id", "Review"]]

test = test.fillna(" ")


# Load the tokenizer and the model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("./results/checkpoint-46000")


class AmazonReviewDataset(Dataset):
    def __init__(self, reviews, tokenizer, max_length):
        self.reviews = reviews
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = self.reviews[idx]
        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "review_text": review,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }


# Set parameters
MAX_LEN = 128
BATCH_SIZE = 32

reviews = test["Review"].tolist()

data = AmazonReviewDataset(reviews, tokenizer, MAX_LEN)
dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model.eval()
all_preds = []
with torch.no_grad():
    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids, attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())

test["overall"] = all_preds
test["overall"] = test["overall"] + 1
test = test.drop(columns=["Review"])
test.to_csv("submissions/test_predictions_hf.csv", index=False)
