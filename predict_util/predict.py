import pandas as pd


def predict(model, X_train, datelist_train, n_future=90, n_past=60):
    # Generate list of sequence of days for predictions
    datelist_future = pd.date_range(datelist_train[-1], periods=n_future, freq='1d').tolist()

    '''
    Remeber, we have datelist_train from begining.
    '''

    # Convert Pandas Timestamp to Datetime object (for transformation) --> FUTURE
    datelist_future_ = []
    for this_timestamp in datelist_future:
        datelist_future_.append(this_timestamp.date())

    # Perform predictions
    predictions_future = model.predict(X_train[-n_future:])

    predictions_train = model.predict(X_train[n_past:])
    return predictions_future, predictions_train, datelist_future_