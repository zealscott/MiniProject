import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM, Bidirectional
from keras.layers.convolutional import MaxPooling1D, Conv1D
from keras.callbacks import EarlyStopping
from keras.models import load_model
import pickle
import numpy as np
import pandas as pd
# import my file
import Doc2Vec


def MyLSTM():
    # model
    model = Sequential()
    # add conv1D layer
    model.add(Conv1D(filters=32, kernel_size=3, padding='same',
                     activation='relu', input_shape=(1000, 100)))
    model.add(Dropout(0.2))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Dropout(0.2))
    # add conv1D layer
    model.add(Conv1D(filters=32, kernel_size=3,
                     padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Dropout(0.2))
    # add conv1D layer
    model.add(Conv1D(filters=32, kernel_size=3,
                     padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Dropout(0.2))
    # add conv1D layer
    model.add(Conv1D(filters=32, kernel_size=3,
                     padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Dropout(0.2))
    # add LSTM layer
    model.add(LSTM(300, dropout=0.2, recurrent_dropout=0.2))
    # model.add(Dropout(0.2))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    # model complete
    print(model.summary())
    return model


if __name__ == '__main__':
    # get model
    model = MyLSTM()
    # fit train data
    trainData, trainLabel = Doc2Vec.LoadDataTrain()
    # model.fit(trainData, trainLabel, validation_split=0.20,
    #           epochs=10, batch_size=64, callbacks=[EarlyStopping(patience=2)])
    model.fit(trainData, trainLabel, validation_split=0.2, epochs=10, batch_size=64)
    # save model
    model.save('./Persistence/model.h5')
    # model = load_model('./RNN/model.h5')
    X_test = Doc2Vec.LoadDataTest()
    Y_test = model.predict(X_test)
    print(Y_test)
    print(Y_test.ndim)
    # save result
    Doc2Vec.SaveResult(Y_test)
