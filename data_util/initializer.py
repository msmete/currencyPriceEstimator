from data_util import parser
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import datetime as dt


def initialize():
    data_bnb = parser.read_csv(os.path.join("data", "BNB_USD.csv"))
    data_btc = parser.read_csv(os.path.join("data", "BTC_USD.csv"))
    data_eth = parser.read_csv(os.path.join("data", "ETH_USD.csv"))
    # data_eur = parser.read_csv(os.path.join("data", "EUR_USD.csv"))
    # data_xau = parser.read_csv(os.path.join("data", "XAU_USD.csv"))
    # datasets = [data_bnb, data_btc, data_eth, data_eur, data_xau]
    datasets = [data_bnb, data_btc, data_eth]
    dataset_train = concat(datasets)
    dataset_train = dataset_train[:: -1]
    # dataset_train = np.invert(dataset_train)
    cols = [i for i in range(len(dataset_train[0, :]))]

    for i in range(0, dataset_train.shape[0]):
        for j in cols :
            dataset_train[i][j] = float_format(dataset_train[i][j].replace(',', ''))

    dataset_train = dataset_train.astype(float)
    training_set = pd.DataFrame(data=dataset_train, columns = cols)  # 1st row as the column names

    # Feature Scaling
    sc = StandardScaler()
    training_set_scaled = sc.fit_transform(training_set)

    sc_predict = StandardScaler()
    sc_predict.fit_transform(np.array(training_set[0]).reshape((len(training_set[0]), 1)))
    return training_set_scaled, sc_predict


def concat(items):
    new_items = [clean(item) for item in items]
    data_frame = new_items[0]
    for i in range(len(new_items)-1):
        data_frame = np.hstack((data_frame, new_items[i+1]))
    return data_frame


def clean(item):
    return item.drop(["Tarih", "Fark %"], axis=1)


def float_format(value):
    k = value.find("K")
    if k > 0:
        return float(value[:-1]) * 10**3
    k = value.find("M")
    if k > 0:
        return float(value[:-1]) * 10**6
    k = value.find("B")
    if k > 0:
        return float(value[:-1]) * 10**9
    k = value.find("T")
    if k > 0:
        return float(value[:-1]) * 10**12
    return float(value)


def create_datelist():
    data_bnb = parser.read_csv(os.path.join("data", "BNB_USD.csv"))
    datelist =  data_bnb["Tarih"]
    datelist = list(datelist)[::-1]
    datelist = [dt.datetime.strptime(date, '%d.%m.%Y').date() for date in datelist]
    return datelist



