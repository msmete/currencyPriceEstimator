from data_util import initializer, preprocess
from train_util import model, train
from predict_util import predict
from general_util import util
import pandas as pd


n_past = 90
n_future = 60
c_index = 0
epochs = 5
# datelist = initializer.create_datelist()
training_set_scaled, sc_predict, training_set, datelist = initializer.initialize(c_index)
X_train, y_train = preprocess.train_test_split(training_set_scaled, c_index, n_past, n_future)
predictor = model.create(n_past, training_set_scaled.shape[1]-1)
predictor, history = train.train(predictor, X_train, y_train, epochs)
predictions_future, predictions_train, datelist_future = predict.predict(predictor, X_train, datelist, n_future=n_future, n_past=n_past)
# Inverse the predictions to original measurements


y_pred_future = sc_predict.inverse_transform(predictions_future)
y_pred_train = sc_predict.inverse_transform(predictions_train)


PREDICTIONS_FUTURE = pd.DataFrame(y_pred_future, columns=[0]).set_index(pd.Series(datelist_future))
# PREDICTION_TRAIN = pd.DataFrame(y_pred_train, columns=[0]).set_index(pd.Series(datelist[2 * n_past + n_future -1:]))
PREDICTION_TRAIN = pd.DataFrame(y_pred_train, columns=[0]).set_index(pd.Series(datelist[n_past:]))

# Convert <datetime.date> to <Timestamp> for PREDCITION_TRAIN
PREDICTION_TRAIN.index = PREDICTION_TRAIN.index.to_series().apply(util.datetime_to_timestamp)

print(PREDICTION_TRAIN)
# util.visualize()




import matplotlib.pyplot as plt
from pylab import rcParams

rcParams['figure.figsize'] = 14, 5

# Plot parameters
START_DATE_FOR_PLOTTING = '2020-01-01'

plt.plot(PREDICTIONS_FUTURE.index, PREDICTIONS_FUTURE[0], color='r', label='Predicted Stock Price')
plt.plot(PREDICTION_TRAIN.loc[START_DATE_FOR_PLOTTING:].index, PREDICTION_TRAIN.loc[START_DATE_FOR_PLOTTING:][0], color='orange', label='Training predictions')
# plt.plot(training_set.loc[START_DATE_FOR_PLOTTING:].index, training_set.loc[START_DATE_FOR_PLOTTING:][1], color='b', label='Actual Stock Price')
plt.plot(training_set[training_set.columns[0]], training_set[training_set.columns[c_index+1]],  color='b', label='Actual Stock Price')

plt.axvline(x = min(PREDICTIONS_FUTURE.index), color='green', linewidth=2, linestyle='--')

plt.grid(which='major', color='#cccccc', alpha=0.5)

plt.legend(shadow=True)
plt.title('Predcitions and Acutal Stock Prices', family='Arial', fontsize=12)
plt.xlabel('Timeline', family='Arial', fontsize=10)
plt.ylabel('Stock Price Value', family='Arial', fontsize=10)
plt.xticks(rotation=45, fontsize=8)
plt.show()

