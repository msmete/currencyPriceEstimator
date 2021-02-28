import numpy as np


def train_test_split(training_set_scaled, c_index=0):
    # Creating a data structure with 90 timestamps and 1 output
    X_train = []
    y_train = []

    n_future = 60   # Number of days we want top predict into the future
    n_past = 90     # Number of past days we want to use to predict the future

    for i in range(n_past, len(training_set_scaled) - n_future +1):
        X_train.append(training_set_scaled[i - n_past:i, 0:training_set_scaled.shape[1] - 1])
        y_train.append(training_set_scaled[i + n_future - 1:i + n_future, c_index])

    X_train, y_train = np.array(X_train), np.array(y_train)

    print('X_train shape == {}.'.format(X_train.shape))
    print('y_train shape == {}.'.format(y_train.shape))
    return X_train, y_train