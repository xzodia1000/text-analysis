from sklearn.model_selection import train_test_split
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

# Load your dataset
df = pd.read_csv("data/train.csv")
df = df[["Review", "overall"]]
df = df.dropna(subset=["Review"])
df["overall"] = df["overall"] - 1

# Load the tokenizer and the model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("./results/checkpoint-46000")


class AmazonReviewDataset(Dataset):
    def __init__(self, reviews, labels, tokenizer, max_length):
        self.reviews = reviews
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = self.reviews[idx]
        label = self.labels[idx]
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
            "labels": torch.tensor(label, dtype=torch.long),
        }


# Set parameters
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 5

# Split the data
X_train, X_val, y_train, y_val = train_test_split(
    df["Review"], df["overall"], test_size=0.2, random_state=42
)

# Load the validation dataset
val_dataset = AmazonReviewDataset(
    X_val.to_numpy(), y_val.to_numpy(), tokenizer, MAX_LEN
)

# Create a DataLoader for the validation set
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Move the model to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Evaluate the model
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for batch in tqdm(val_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate the accuracy
accuracy = accuracy_score(all_labels, all_preds)
print(f"Accuracy: {accuracy:.4f}")
