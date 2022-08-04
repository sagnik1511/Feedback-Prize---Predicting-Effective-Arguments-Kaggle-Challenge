import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# filepath
filepath = "data/train.csv"

# reading file
df = pd.read_csv(filepath)

#loading unique target features
target_features = df["discourse_effectiveness"].to_numpy()


# fitting encoder
ohc = LabelEncoder()
ohc.fit(target_features)

# saving encoder
with open("lbc.pkl", "wb") as f:
    pickle.dump(ohc, f)
    f.close()
