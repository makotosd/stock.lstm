#
#
#
import numpy
import pandas
from sklearn import preprocessing

from keras.models import Sequential
from keras.models import model_from_json

class StockCNN:
    def __init__(self):
        self.length_of_sequences = 100

    def load_data(self, date, data, n_prev=100):
        label = []
        X, Y = [], []
        for i in range(len(data) - n_prev):
            label.append(date.iloc[i+n_prev].as_matrix())
            X.append(data['close'].iloc[i:(i+n_prev)].as_matrix())
            array = data.iloc[i:(i+n_prev)].as_matrix()
            if (float(array[-1]) > float(data.iloc[i+n_prev].as_matrix())):
                Y.append([0])
            else:
                Y.append([1])

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

if __name__ == "__main__":

    stock = StockCNN()
    data = None
    for year in range(2007, 2018):
        data_ = pandas.read_csv('stocks_6702-T_1d_' + str(year) +  '.csv', encoding="shift-jis")
        data = data_ if (data is None) else pandas.concat([data, data_])

    data.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'value']
    data['date'] = pandas.to_datetime(data['date'], format='%Y-%m-%d')
    data['close'] = preprocessing.scale(data['close'])
    data = data.sort_values(by='date')
    data = data.reset_index(drop=True)
    data = data.loc[:, ['date', 'close']]

    # データ準備
    split_pos = int(len(data) * 0.9)
    x_label, x_train, y_train = stock.load_data(data[['date']].iloc[0:split_pos],
                                                data[['close']].iloc[0:split_pos], stock.length_of_sequences)
    x_tlabel, x_test,  y_test = stock.load_data(data[['date']].iloc[split_pos:],
                                                data[['close']].iloc[split_pos:], stock.length_of_sequences)

    model = None
    # モデルのロード
    with open('model.json') as f:
        model = model_from_json(f.read())

    # モデルのコンパイル
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    # 重みのロード
    model.load_weights('weights.best.hdf5')
    layer_num = 0
    W = model.layers[layer_num].get_weights()[0]
    print(W)

    # 評価
    good = 0
    index = 0
    for values in x_test:
        y = y_test[index][0]
        predict = model.predict(numpy.array([values]))[0][0]
        print(x_tlabel[index][0], ',', y, ',', predict)
        #  print(y)
        #  print(predict)
        if predict < 0.5:
            if y == 0:
                good += 1
        else:
            if y == 1:
                good += 1
        index += 1

    print("accuracy = {0:.2f}".format(float(good) / len(x_test)))



