import yaml
import torch
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.utils import load_checkpoint
from src.dataset import BERTDatasetTesting
from torch.utils.data import DataLoader


def predict(params):
    dataset = BERTDatasetTesting(params["dataset"]["test_data_path"],
                                 params["dataset"]["seq_len"])
    
    loader = DataLoader(dataset, batch_size=params["dataset"]["batch_size"], shuffle=False)
    print("Data Loder Generated...")
    model = load_checkpoint(params)
    print("Model Loaded...")
    print(model)
    softmax = torch.nn.Softmax(dim=1)
    predictions = torch.ones(1,3)
    with torch.no_grad():
        for batch in tqdm(loader):
            input_ids = batch["input_ids"]
            mask = batch["attention_mask"]
            token_type_ids = batch["token_type_ids"]
            pred = softmax(model(input_ids, mask, token_type_ids))
            predictions = torch.cat([predictions, pred], axis=0)
    predictions = predictions[1:,:]
    encoder = pickle.load(open("lbc.pkl", "rb")) 
    submission_dataframe = pd.DataFrame(predictions.detach().numpy(), columns = encoder.classes_)
    submission_dataframe["discourse_id"] = pd.read_csv(params["dataset"]["test_data_path"])["discourse_id"]
    submission_dataframe = submission_dataframe[["discourse_id","Ineffective","Adequate","Effective"]]
    submission_dataframe.to_csv("submission.csv")
    print("Prediction file saved...")
    

if __name__ == '__main__':
    config_path = "config/__base__.yaml"
    with open(config_path, "r") as f:
        params = yaml.safe_load(f)
        f.close()
    print(params)
    predict(params)