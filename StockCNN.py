
import numpy
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
import random
from imblearn.over_sampling import SMOTE

class StockCNN:
    def __init__(self):
        self.length_of_sequences = 100
        self.rise_rate = 1.02 # 2%以上

    def load_data(self, date, data, n_prev=100, use_smote=False):
        label = []
        X, Y = [], []
        num_Y0, num_Y1 = 0, 0
        for i in range(len(data) - n_prev - 1):
            label.append(date.iloc[i+n_prev].as_matrix())                      # label = iloc[100]
            X.append(data['open'].iloc[i:(i+n_prev)].as_matrix())              # X     = iloc[0:100]
            array = data.iloc[i:(i+n_prev)].as_matrix()                        # array = iloc[0:100]
            #if (float(array[-1]) > float(data.iloc[i+n_prev].as_matrix())):    # Y     = iloc[99] 比較 iloc[100]
            #if (float(data.iloc[i + n_prev].as_matrix()) > float(data.iloc[i + n_prev + 1].as_matrix())):  # Y     = iloc[99] 比較 iloc[100]
            if (float(data.iloc[i + n_prev + 1].as_matrix()) / float(data.iloc[i + n_prev].as_matrix()) < self.rise_rate ):  # Y     = iloc[99] 比較 iloc[100]
                Y.append([0])
                num_Y0 += 1
            else:
                Y.append([1])
                num_Y1 += 1

        # SMOTEを使って、データのアンバランスを調整
        if use_smote == True:
            sm = SMOTE()
            X_res, Y_res = sm.fit_sample(X, Y)

            ret_label = numpy.array(label)
            retX = numpy.array(X_res)
            retY = numpy.array(Y_res)
        else:
            ret_label = numpy.array(label)
            retX = numpy.array(X)
            retY = numpy.array(Y)
        return ret_label, retX, retY

    def create_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.length_of_sequences, activation='sigmoid'))
        model.add(Dense(128, activation='sigmoid'))
        model.add(Dense(64, activation='sigmoid'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

        return model
