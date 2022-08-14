import yaml
import pickle
from src.utils import load_checkpoint

config_path = "config/__base__.yaml"
params = yaml.safe_load(open(config_path, "r"))

model = load_checkpoint(params)
with open("bert_model.pkl", "wb") as f:
    pickle.dump(model, f)
    f.close()