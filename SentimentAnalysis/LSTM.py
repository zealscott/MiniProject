import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM, Bidirectional
from keras.layers.convolutional import MaxPooling1D, Conv1D
from keras.callbacks import EarlyStopping
from keras.models import load_model
import numpy as np
import pandas as pd
# import for ROC
import functools
from keras import backend as K
import tensorflow as tf
# import my file
import Doc2Vec

# warp tensorflow metrics to keras
# i.e. ROC
def as_keras_metric(method):
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper


def MyLSTM():
    # how many words count?
    feature = 700
    # size of each word
    vec_size = 300
    # use ROC
    auc_roc = as_keras_metric(tf.metrics.auc)
    # model
    model = Sequential()
    # add conv1D layer
    model.add(Conv1D(filters=32, kernel_size=3, padding='same',
                     activation='relu', input_shape=(feature, vec_size)))
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
                  optimizer='adam', metrics=[auc_roc])
    # model complete
    print(model.summary())
    return model


if __name__ == '__main__':
    # get model
    model = MyLSTM()
    # fit train data
    trainData, trainLabel = Doc2Vec.LoadDataTrain()
    # model.fit(trainData, trainLabel, validation_split=0.2,
    #           epochs=20, batch_size=64, callbacks=[EarlyStopping(patience=2)])
    model.fit(trainData, trainLabel, validation_split=0.2, epochs=10, batch_size=64)
    # save model
    model.save('./Persistence/Model/LSTM.h5')
    # auc_roc = as_keras_metric(tf.metrics.auc)
    # model = load_model('./Persistence/Model/LSTM.h5',custom_objects={'auc':auc_roc})
    X_test = Doc2Vec.LoadDataTest()
    Y_test = model.predict(X_test)
    # save result
    Doc2Vec.SaveResult(Y_test)
