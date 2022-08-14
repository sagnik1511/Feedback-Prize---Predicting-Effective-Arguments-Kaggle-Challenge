from torch import load
from sklearn.metrics import (accuracy_score, precision_score, f1_score, recall_score)
from src.dataset import BERTDatasetTraining
from src.bert_model import FeedbackBERT
from torch.utils.data import random_split, DataLoader


def calculate_scores(y_true, y_pred, dataset="training"):
    acc = accuracy_score(y_true, y_pred)
    prc = precision_score(y_true, y_pred, average="weighted")
    rcs = recall_score(y_true, y_pred, average="weighted")
    f1s = f1_score(y_true, y_pred, average="weighted")


    return {
        f"{dataset}_accuracy": acc,
        f"{dataset}_precision": prc,
        f"{dataset}_recall": rcs,
        f"{dataset}_f1_score": f1s
    }

def create_loaders(params):
    dataset = BERTDatasetTraining(params["train_data_path"], params["seq_len"])
    split_ratio = params["split_ratio"]
    tr_size = int(len(dataset) * (1-split_ratio))
    ts_size = len(dataset) - tr_size

    train_dataset, val_dataset = random_split(dataset, [tr_size, ts_size])
    print(f"Train Dataset populated with {len(train_dataset)} records.")
    print(f"Validation Dataset populated with {len(val_dataset)} records.")
    train_dl = DataLoader(train_dataset, batch_size = params["batch_size"], shuffle=True, drop_last=True)
    val_dl = DataLoader(val_dataset, batch_size = params["batch_size"], shuffle=True, drop_last=True)

    return train_dl, val_dl

def load_checkpoint(params):
    chkp = load(params["training"]["chkp_path"], map_location="cpu")
    model = FeedbackBERT(**params["model"])
    model.load_state_dict(chkp["model_state"])

    return model

