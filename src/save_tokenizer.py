import pickle
from transformers import BertTokenizer

# tokenizer for the task
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokenizer.model_max_length = 512

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
    f.close()