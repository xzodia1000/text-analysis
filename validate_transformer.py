import argparse
import pandas as pd
import numpy as np
from sklearn.base import accuracy_score
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, TensorDataset

from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    BertTokenizer,
    BertForSequenceClassification,
)


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


def get_data_loaders(tokenizer):
    train = pd.read_csv("data/train.csv")
    train = train[["Review", "overall"]]
    train = train.dropna(subset=["Review"])
    texts = train["Review"].tolist()
    labels = train["overall"].values - 1

    input_ids, attention_masks = preprocess_texts(texts, tokenizer)
    labels = torch.tensor(labels)

    train_inputs, val_inputs, train_labels, val_labels = train_test_split(
        input_ids, labels, random_state=42, test_size=0.2
    )
    train_masks, val_masks, _, _ = train_test_split(
        attention_masks, labels, random_state=42, test_size=0.2
    )

    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)

    val_data = TensorDataset(val_inputs, val_masks, val_labels)
    val_dataloader = DataLoader(val_data, batch_size=32, shuffle=False)

    class_weights = compute_class_weight(
        "balanced", classes=np.unique(labels.numpy()), y=labels.numpy()
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    return train_dataloader, val_dataloader, class_weights


def validate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            batch_inputs, batch_masks, batch_labels = batch
            batch_inputs = batch_inputs.to(device)
            batch_masks = batch_masks.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_inputs, attention_mask=batch_masks)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)

    return accuracy


def prepare_model(model_name):
    if model_name == "roberta":
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        model = RobertaForSequenceClassification.from_pretrained(
            "roberta-base", num_labels=5
        )
    elif model_name == "distilbert":
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=5
        )
    elif model_name == "bert":
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=5
        )
    else:
        raise ValueError("Invalid model name")

    return model, tokenizer


def __main__():
    models = ["roberta_cw", "distilbert_cw", "distilbert"]

    for model_path in models:
        model_name = model_path.split("_")[0]
        model, tokenizer = prepare_model(model_name)
        _, val_dataloader, _ = get_data_loaders(tokenizer)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        state_dict = torch.load(f"./models/{model_path}.pth")
        model.load_state_dict(state_dict)

        accuracy = validate_model(model, val_dataloader, device)
        tqdm.write(
            f"Model: {model_name}, Accuracy: {accuracy:.4f}, Class Weights: {"Yes" if 'cw' in model_path else 'No'}"
        )
