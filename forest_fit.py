

from sklearn import tree
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
import csv
import random
import math

TRAINING_DATA_LENGTH = 60
DATA_FOR_TRAINING = 0.9

# 明日と明後日の株価からラベル(-1, 0, 1, 2)を返す。
#    0: 1%以上下がった
#    1: ±1%以下の変動
#    2: 1-2%上がった
#    3: 2%以上上がった
def upordown(tomorrow, thedayaftertomorrow):
    rate = float(thedayaftertomorrow) / float(tomorrow)
    ret = 0
    if rate < 0.99:
        ret = 0
    elif rate < 1.01:
        ret = 1
    elif rate < 1.02:
        ret = 2
    else:
        ret = 3

    return ret
#
# 株価データ(csv)の読み込み
#
dates = []
prices = []
with open('stocks_6702-T_1d_2017-2007.csv', newline='') as csvfile:
#with open('stocks_6702-T_1d_2017.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    header = next(spamreader)  # 最初の一行をヘッダーとして取得

    for row in reversed(list(spamreader)):
        dates.append(row[0])
        prices.append(row[1])

#
#　トレーニングデータ変換
#
training_range = (len(prices) - TRAINING_DATA_LENGTH - 2)   # 後ろのほうはトレーニングデータの始点にならない
                                                            # 2は、ラベル生成に翌日と翌々日を使うから
number_of_training = int(training_range * DATA_FOR_TRAINING)   # DATA_FOR_TRAINING率を使う。
trained_flag = [0 for i in range(training_range)]   # フラグの初期化
number_of_trained = 0    # トレーニング済みの回数
train_X = []
train_Y = []
i = 0
while number_of_trained < number_of_training:
    # i = random.randint(0, training_range-1)
    if trained_flag[i] == 0:
        features = prices[i:i+TRAINING_DATA_LENGTH]  # 1トレーニングデータセット
        label = upordown(prices[i+TRAINING_DATA_LENGTH], prices[i+TRAINING_DATA_LENGTH+1])
        train_X.append(features)
        train_Y.append(label)
        trained_flag[i] = 1     # フラグを立てて、
        number_of_trained += 1  # トレーニング済みの個数を増やす
        i += 1

# DecisionTree
#clf = tree.DecisionTreeClassifier()

# The main parameters to adjust when using these methods is n_estimators and max_features. The former is the number of trees in the forest. The larger the better, but also the longer it will take to compute. In addition, note that results will stop getting significantly better beyond a critical number of trees. The latter is the size of the random subsets of features to consider when splitting a node. The lower the greater the reduction of variance, but also the greater the increase in bias. Empirical good default values are max_features=n_features for regression problems, and max_features=sqrt(n_features) for classification tasks (where n_features is the number of features in the data).
clf = RandomForestClassifier(n_estimators=400, max_features=int(math.sqrt(TRAINING_DATA_LENGTH)))

#
# トレーニング
#
print ("training...", number_of_training, "training data. A number of feature is", TRAINING_DATA_LENGTH, ".")
clf = clf.fit(train_X, train_Y)

# training結果のダンプ
joblib.dump(clf, 'decisiontree.pkl', compress=True)

#
# 予測 ＆ 正解率計算
#
test_X = []
test_Y = []
answer_Y = []
hit_or_miss = [0 for i in range(training_range)]

hit = 0
hit_up = 0
hit_down = 0
miss = 0
miss_up = 0
miss_down = 0
print("testing...", trained_flag.count(0), "test data")
summary = [{'0':0,'1':0, '2':0, '3':0},
{'0':0,'1':0, '2':0, '3':0},
{'0':0,'1':0, '2':0, '3':0},
{'0':0,'1':0, '2':0, '3':0}]
for i in range(training_range):
    if trained_flag[i] == 0:    # 0:トレーニングに使っていない
        test_X = prices[i:i+TRAINING_DATA_LENGTH]  #
        result = clf.predict([test_X])
        test_Y.append(result[0])

        label = upordown(prices[i+TRAINING_DATA_LENGTH], prices[i+TRAINING_DATA_LENGTH+1])
        answer_Y.append(label)

        if(test_Y[-1] == answer_Y[-1]):
            hit += 1
        else:
            miss += 1

        # summary[test_Y[-1]][answer_Y[-1]] += 1

        ''''''
        if test_Y[-1] >= 1 and test_Y[-1] == answer_Y[-1]:  # 上がると予想し、当たった
            hit_up += 1
            hit_or_miss[i] = 1
        elif test_Y[-1] < 1 and test_Y[-1] == answer_Y[-1]:  # 下がると予想し、当たった
            hit_down += 1
            hit_or_miss[i] = 1
        elif test_Y[-1] >= 1 and test_Y[-1] != answer_Y[-1]:  # 上がると予想し、外れた
            miss_up += 1
            hit_or_miss[i] = 0
        else:                                                 # 下がると予想し、外れた
            miss_down += 1
            hit_or_miss[i] = 0
        ''''''

#  print (test_Y)
#  print (answer_Y)
#  print("予測up: 予測down = ", test_Y.count(1), ":", test_Y.count(-1))
#  print("結果up: 結果down = ", answer_Y.count(1), ":", answer_Y.count(-1))
#  print("UP予想で、正解   | 不正解 = ", hit_up, "|", miss_up, "正解率:", hit_up/(hit_up+miss_up))
#  print("DOWN予想で、正解 | 不正解 = ", hit_down, "|", miss_down)
#  print("正解率 = ", (hit_up+hit_down)/len(answer_Y))
print(trained_flag)
print(hit_or_miss)
print("hit:", hit)
print("mis:", miss)
