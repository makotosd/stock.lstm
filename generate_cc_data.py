# coding: Shift_JIS

import os  # os���W���[���̃C���|�[�g
import fnmatch
import re
import pandas

regex = r'^20[0-9]+$'

#
#
#
data = None

# os.listdir('�p�X')
# �w�肵���p�X���̑S�Ẵt�@�C���ƃf�B���N�g����v�f�Ƃ��郊�X�g��Ԃ�
for year in range(2000, 2009):
    items = []
    ccs = []
    dir = "./csv_day/" + str(year) + "/"
    print(dir)
    for file in os.listdir(dir):
        if fnmatch.fnmatch(file, '*.txt'):
            print("\t" + file)
            for line in open(dir + file, 'r', encoding='cp932'):         # 1�s��
                if re.match(regex, line):
                    date = line.rstrip()                # ���s�̍폜
                else:
                    itemList = line[:-1].split('\t')    # �^�u��؂��ǂ�
                    itemList.insert(1, date)
                    itemList.pop(2)
                    #  print(itemList)
                    items.append(itemList)

                    ccs.append(itemList[0])   # �J���p�j�[�R�[�h���L�^

    data = pandas.DataFrame(items)
    if(len(data.columns) == 7):
        data.columns = ['cc', 'date', 'open', 'high', 'low', 'close', 'volume']
    elif(len(data.columns) == 8):
        data.columns = ['cc', 'date', 'open', 'high', 'low', 'close', 'volume', 'value']
    # print(data.index)
    # print(data.columns)
    # print(data)

    data['date'] = pandas.to_datetime(data['date'], format='%Y%m%d')   # ���t�^�ɁB

    ccs_uniq = list(set(ccs))  # �L�^���Ă������J���p�j�[�R�[�h��Uniq��
    ccs_uniq.sort()
    for cc in ccs_uniq:
        print("CC: ", cc, ", Year: ", year)
        filename = 'csv_cc/stocks_%s_1d_%s.csv' % (cc, year)
        data_cc = data[data.cc == cc]
        data_cc.to_csv(filename, index=False,
                       columns=['date', 'open', 'high', 'low', 'close', 'volume'],
                       header=['date', 'open', 'high', 'low', 'close', 'volume'])


