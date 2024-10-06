import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from torch.utils.data import Dataset

# Load your dataset
df = pd.read_csv("data/train.csv")
df = df[["Review", "overall"]]
df = df.dropna(subset=["Review"])
df["overall"] = df["overall"] - 1


class SaveModelCallback(TrainerCallback):
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def on_epoch_end(self, args, state, control, **kwargs):
        model.save_pretrained(f"{self.output_dir}/checkpoint-{state.epoch}")


# Preprocess the data
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

# Load the tokenizer and the dataset
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_dataset = AmazonReviewDataset(
    X_train.to_numpy(), y_train.to_numpy(), tokenizer, MAX_LEN
)
val_dataset = AmazonReviewDataset(
    X_val.to_numpy(), y_val.to_numpy(), tokenizer, MAX_LEN
)

save_model_callback = SaveModelCallback(output_dir="./hf/")

# Define the model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[save_model_callback],
)

# Train the model
trainer.train()
