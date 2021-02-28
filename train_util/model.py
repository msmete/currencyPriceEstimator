# Import Libraries and packages from Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.optimizers import Adam


def create(h, w):
    # Initializing the Neural Network based on LSTM
    model = Sequential()

    # Adding 1st LSTM layer
    model.add(LSTM(units=128, return_sequences=True, input_shape=(h, w)))

    # Adding 2nd LSTM layer
    model.add(LSTM(units=100, return_sequences=False))

    # Adding Dropout
    model.add(Dropout(0.1))

    # Output layer
    model.add(Dense(units=1, activation='linear'))
    # model.add(Dense(units=1, activation='relu'))

    # Compiling the Neural Network
    model.compile(optimizer = Adam(learning_rate=0.01), loss='mean_squared_error')
    return model