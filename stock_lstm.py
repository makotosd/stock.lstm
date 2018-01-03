# -*- coding: utf-8 -*-
#
#  <https://qiita.com/tsunaki/items/05248eea2220fb9efd71#_reference-7567a2ba4f277b965e28>より
#

import sys
import os
#  import numpy
import pandas
from sklearn import preprocessing

import keras
import StockCNN

if __name__ == "__main__":

    stock = StockCNN.StockCNN()
    data = None
    for year in range(2007, 2018):
        data_ = pandas.read_csv('stocks_6702-T_1d_' + str(year) +  '.csv', encoding="shift-jis")
        data = data_ if (data is None) else pandas.concat([data, data_])

    data.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'value']
    data['date'] = pandas.to_datetime(data['date'], format='%Y-%m-%d')
    data['open'] = preprocessing.scale(data['open'])
    data = data.sort_values(by='date')
    data = data.reset_index(drop=True)
    data = data.loc[:, ['date', 'open']]

    # データ準備
    split_pos = int(len(data) * 0.9)
    x_label, x_train, y_train = stock.load_data(data[['date']].iloc[0:split_pos],
                                                data[['open']].iloc[0:split_pos], stock.length_of_sequences)
    x_tlabel, x_test,  y_test = stock.load_data(data[['date']].iloc[split_pos:],
                                                data[['open']].iloc[split_pos:], stock.length_of_sequences)

    model = stock.create_model()

    #  callback
    #  学習が収束した際に途中で学習を打ち切る用のコールバックと，
    #  TensorFlowのTensorBoardに書き出す用のコールバックを生成
    # es_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=1000, verbose=0, mode='auto')
    tb_cb = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1)
    mc_cb = keras.callbacks.ModelCheckpoint('./weights.best.hdf5', monitor='val_acc', verbose=1, save_best_only=True,
                                            mode='max')

    # model.fit(x_train, y_train, nb_epoch=5000, batch_size=500, verbose=1)
    history = model.fit(x_train, y_train, nb_epoch=1000, batch_size=500, verbose=0,
                        validation_data=(x_test, y_test), callbacks=[tb_cb, mc_cb])
    # model.fit(x_train, y_train, nb_epoch=1000, batch_size=500, verbose=1,
    #          validation_data=(x_test, y_test), callbacks=[es_cb, tb_cb])

    # モデルの保存
    from keras.utils import plot_model
    model_json = model.to_json()
    with open("model.json", mode='w') as f:
        f.write(model_json)

    '''''
    #  学習済みの重みの保存
    model.save_weights("weight.hdf5")

    #  学習履歴の保存
    import pickle
    with open("history.pickle", mode='wb') as f:
        pickle.dump(history.history, f)

    # 評価
    good = 0
    index = 0
    for values in x_test:
        y = y_test[index][0]
        predict = model.predict(numpy.array([values]))[0][0]
        print(x_tlabel[index][0])
        print(y)
        print(predict)
        if predict < 0.5:
            if y == 0:
                good += 1
        else:
            if y == 1:
                good += 1
        index += 1
    print("accuracy = {0:.2f}".format(float(good) / len(x_test)))
    '''''