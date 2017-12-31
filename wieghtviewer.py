#
#
#
from keras.models import model_from_json

model = None
# モデルのロード
with open('model.json') as f:
    model = model_from_json(f.read())

# モデルのコンパイル
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# 重みのロード
model.load_weights('weight.hdf5')

layer_num = 0

W = model.layers[layer_num].get_weights()[0]

print(W)


