from data_util import initializer, preprocess
from train_util import model, train
from predict_util import predict
from general_util import util
import pandas as pd

n_past = 60
n_future = 90
datelist = initializer.create_datelist()
training_set_scaled, sc_predict = initializer.initialize()
X_train, y_train = preprocess.train_test_split(training_set_scaled)
predictor = model.create(90, training_set_scaled.shape[1]-1)
predictor, history = train.train(predictor, X_train, y_train )
predictions_future, predictions_train, datelist_future = predict.predict(predictor, X_train, datelist, n_future=90, n_past=60)
# Inverse the predictions to original measurements


y_pred_future = sc_predict.inverse_transform(predictions_future)
y_pred_train = sc_predict.inverse_transform(predictions_train)


PREDICTIONS_FUTURE = pd.DataFrame(y_pred_future, columns=[0]).set_index(pd.Series(datelist_future))
PREDICTION_TRAIN = pd.DataFrame(y_pred_train, columns=[0]).set_index(pd.Series(datelist[2 * n_past + n_future -1:]))

# Convert <datetime.date> to <Timestamp> for PREDCITION_TRAIN
PREDICTION_TRAIN.index = PREDICTION_TRAIN.index.to_series().apply(util.datetime_to_timestamp)

print(PREDICTION_TRAIN)