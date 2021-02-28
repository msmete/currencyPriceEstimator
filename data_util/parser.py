import pandas as pd

def read_csv(path):
    dataset = pd.read_csv(path, decimal=',')
    return dataset

