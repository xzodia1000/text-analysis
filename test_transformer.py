import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score

test = pd.read_csv("data/test.csv")
test = test[["id", "Review"]]

test = test.fillna(" ")

# Load the tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using {device} device")

model.to(device)
state_dict = torch.load("transformer_epoch_6.pth")
model.load_state_dict(state_dict)


# Preprocess the text data
def preprocess_texts(texts, tokenizer, max_length=512):
    input_ids = []
    attention_masks = []

    for text in tqdm(texts):
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids.append(encoded["input_ids"])
        attention_masks.append(encoded["attention_mask"])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return input_ids, attention_masks


# Prepare the dataset
texts = test["Review"].tolist()  # Assuming 'review_text' is the column with text data

input_ids, attention_masks = preprocess_texts(texts, tokenizer)

# Create DataLoader
test_data = TensorDataset(input_ids, attention_masks)
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False)

print("Validation loop")

# Validation loop
model.eval()
all_preds = []
with torch.no_grad():
    for batch in tqdm(test_dataloader):
        batch_inputs, batch_masks = batch
        batch_inputs = batch_inputs.to(device)
        batch_masks = batch_masks.to(device)

        outputs = model(batch_inputs, attention_mask=batch_masks)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())

test["overall"] = all_preds
test["overall"] = test["overall"] + 1
test = test.drop(columns=["Review"])
test.to_csv("data/test_predictions.csv", index=False)
