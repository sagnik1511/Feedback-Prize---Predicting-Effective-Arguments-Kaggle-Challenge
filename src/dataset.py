import pickle
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertTokenizer


# tokenizer for the task
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokenizer.model_max_length = 512

class BERTDatasetTraining:

    def __init__(self, meta_df, seq_len = 512):

        self.meta_df = pd.read_csv(meta_df)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self._label_encode()

    def __len__(self):
        return len(self.meta_df)

    # performing target feature encoding
    def _label_encode(self):
        with open("lbc.pkl", "rb") as f:
            self.lbc = pickle.load(f)
        self.meta_df["discourse_effectiveness"] = self.lbc.transform(
            self.meta_df["discourse_effectiveness"])

    def __getitem__(self, index):
        sentence = self.meta_df.loc[index,"discourse_type"] + " " + \
         self.meta_df.loc[index,"discourse_text"]
        label = self.meta_df.loc[index, "discourse_effectiveness"]

        # encoding texts
        tokens = tokenizer.encode_plus(sentence, padding="max_length", truncation=True)

        # adding padding to the right of dataset
        pad_len = max(0, self.seq_len - len(tokens["input_ids"]))
        tokens["input_ids"] += [0 for _ in range(pad_len)]
        tokens["attention_mask"] += [0 for _ in range(pad_len)]
        tokens["token_type_ids"] += [0 for _ in range(pad_len)]

        return {
            "input_ids" : torch.tensor(tokens["input_ids"], dtype=torch.long),
            "attention_mask" : torch.tensor(tokens["attention_mask"], dtype=torch.long),
            "token_type_ids" : torch.tensor(tokens["token_type_ids"], dtype=torch.long),
            "target" : torch.tensor(label, dtype=torch.long)
        }



class BERTDatasetTesting:

    def __init__(self, meta_df, seq_len = 512):

        self.meta_df = pd.read_csv(meta_df)
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __len__(self):
        return len(self.meta_df)


    def __getitem__(self, index):
        sentence = self.meta_df.loc[index,"discourse_type"] + " " + \
         self.meta_df.loc[index,"discourse_text"]

        # encoding texts
        tokens = tokenizer.encode_plus(sentence, padding="max_length", truncation=True)

        # adding padding to the right of dataset
        pad_len = max(0, self.seq_len - len(tokens["input_ids"]))
        tokens["input_ids"] += [0 for _ in range(pad_len)]
        tokens["attention_mask"] += [0 for _ in range(pad_len)]
        tokens["token_type_ids"] += [0 for _ in range(pad_len)]

        return {
            "input_ids" : torch.tensor(tokens["input_ids"], dtype=torch.long),
            "attention_mask" : torch.tensor(tokens["attention_mask"], dtype=torch.long),
            "token_type_ids" : torch.tensor(tokens["token_type_ids"], dtype=torch.long)
        }


if __name__ == '__main__':

    ds = BERTDatasetTraining("data/train.csv")
    dl = DataLoader(ds, batch_size=32, shuffle=True)

    batch = next(iter(dl))

    print(batch["input_ids"].shape, batch["attention_mask"].shape, batch["token_type_ids"].shape, batch["target"].shape)