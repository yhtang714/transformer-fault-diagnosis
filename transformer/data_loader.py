import pandas as pd
from config import gas_columns

def load_data(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=gas_columns, how='all')
    return df