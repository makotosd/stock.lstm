# coding: Shift_JIS

import os  # osモジュールのインポート
import fnmatch
import re
import pandas

regex = r'^20[0-9]+$'

#
#
#
data = None

# os.listdir('パス')
# 指定したパス内の全てのファイルとディレクトリを要素とするリストを返す
for year in range(2000, 2009):
    items = []
    ccs = []
    dir = "./csv_day/" + str(year) + "/"
    print(dir)
    for file in os.listdir(dir):
        if fnmatch.fnmatch(file, '*.txt'):
            print("\t" + file)
            for line in open(dir + file, 'r', encoding='cp932'):         # 1行分
                if re.match(regex, line):
                    date = line.rstrip()                # 改行の削除
                else:
                    itemList = line[:-1].split('\t')    # タブ区切りを読む
                    itemList.insert(1, date)
                    itemList.pop(2)
                    #  print(itemList)
                    items.append(itemList)

                    ccs.append(itemList[0])   # カンパニーコードを記録

    data = pandas.DataFrame(items)
    if(len(data.columns) == 7):
        data.columns = ['cc', 'date', 'open', 'high', 'low', 'close', 'volume']
    elif(len(data.columns) == 8):
        data.columns = ['cc', 'date', 'open', 'high', 'low', 'close', 'volume', 'value']
    # print(data.index)
    # print(data.columns)
    # print(data)

    data['date'] = pandas.to_datetime(data['date'], format='%Y%m%d')   # 日付型に。

    ccs_uniq = list(set(ccs))  # 記録しておいたカンパニーコードをUniqに
    ccs_uniq.sort()
    for cc in ccs_uniq:
        print("CC: ", cc, ", Year: ", year)
        filename = 'csv_cc/stocks_%s_1d_%s.csv' % (cc, year)
        data_cc = data[data.cc == cc]
        data_cc.to_csv(filename, index=False,
                       columns=['date', 'open', 'high', 'low', 'close', 'volume'],
                       header=['date', 'open', 'high', 'low', 'close', 'volume'])


