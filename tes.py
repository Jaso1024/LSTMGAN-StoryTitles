import pandas as pd

data = pd.read_pickle("nosleep_data.pkl")
data.to_csv("nosleep_data.csv")