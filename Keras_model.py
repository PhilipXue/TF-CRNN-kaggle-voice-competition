

import keras

from keras import Model
from keras.layers import Conv2D, MaxPool2D, BatchNormalization, Input, AveragePooling1D
from keras.layers import SeparableConv2D, MaxPool1D, GlobalAveragePooling2D,Dropout,GlobalAveragePooling1D,GlobalMaxPooling1D
from keras.layers import CuDNNGRU as GRU
from keras.layers import CuDNNLSTM as LSTM
from keras.layers import Reshape, Bidirectional, Flatten, Dense, Permute, Multiply, Average, Concatenate,Add
from keras.preprocessing import image
from keras.backend import separable_conv2d
import tqdm
import os
import time
import numpy as np


data_root = '/home/philip/data/Keyword_spot/'
batch_size = 512
training_data_gen = image.ImageDataGenerator(width_shift_range=0.1)
training_gen = training_data_gen.flow_from_directory(data_root + "train/original_all", class_mode="categorical",
                                                     target_size=(128, 63),
                                                     batch_size=batch_size,
                                                     color_mode="grayscale")


conv_size = (3, 3)
# conv_filter = [64, 128, 256, 256]
conv_filter = [8, 16, 32, 64, 128, 256]

dilation = [1, 1, 2, 2]
RNN_Layer = [GRU, LSTM][0]
RNN_repeat = 2
class_num = training_gen.num_classes
attention = False
incept_conv_size = [(1, 1), (5, 5), (3, 7), (7, 3)]


def incept_block(inputs, filters):
    x = Conv2D(int(filters * 0.75), (1, 1),
               padding='same', activation='relu')(inputs)
    a = [SeparableConv2D(filters // 4, size, padding='same',
                         activation='relu')(x) for size in incept_conv_size]
    b = Concatenate()(a)
    return b


def SE_block(inputs, filters):
    x = GlobalAveragePooling2D()(inputs)
    x = Dense(filters // 8, activation='relu')(x)
    x = Dense(filters, activation='sigmoid')(x)
    x = Multiply()([inputs, x])
    return x

def Res_block(inputs,filters):
    y = Conv2D(filters,(1,1),padding='same')(inputs)
    x = SeparableConv2D(filters,(3,3),padding='same',dilation_rate=(1,1))(inputs)
    x = SeparableConv2D(filters,(3,3),padding='same',dilation_rate=(1,1))(x)
#     x = SE_block(x,filters)
    x = Add()([y,x])
    x = SeparableConv2D(filters,(7,7),padding='same',dilation_rate=(1,1))(x)
    x = SeparableConv2D(filters,(7,7),padding='same',dilation_rate=(1,4))(x)
#     x = SE_block(x,filters)
    o = Add()([y,x])
    return o


def get_model():
    inputs = Input((128, 63, 1))
    x = Res_block(inputs,conv_filter[0])
    x = BatchNormalization()(x)
    x = MaxPool2D((4, 1))(x)
    x = Res_block(x,conv_filter[1])
    x = BatchNormalization()(x)
    x = MaxPool2D((2, 1))(x)
    x = Res_block(x,conv_filter[2])
    x = BatchNormalization()(x)
    x = MaxPool2D((2, 1))(x)
    x = Res_block(x,conv_filter[3])
    x = BatchNormalization()(x)
    x = MaxPool2D((2, 1))(x)
    x = Res_block(x,conv_filter[4])
    x = BatchNormalization()(x)
    x = MaxPool2D((2,1))(x)
    x = Res_block(x,conv_filter[5])
    x = BatchNormalization()(x)
    x = MaxPool2D((2,1))(x)
    x = Reshape((63, conv_filter[-1]))(x)
    if attention:
        y = Permute((2,1))(x)
        x = Dense(63, activation='softmax')(y)
        x = Multiply()([y, x])
        x = Permute((2,1))(x)
    for i in range(RNN_repeat):
        x = Bidirectional(
            RNN_Layer(conv_filter[-1], return_sequences=True), merge_mode='ave')(x)
    x = GlobalMaxPooling1D()(x)
    output = Dense(class_num, activation='softmax')(x)
    return Model(inputs, output)


# def get_model():
#     inputs = Input((128, 63, 1))
#     # x = incept_block(inputs, conv_filter[0])
#     x = Conv2D()
#     x = SE_block(x, conv_filter[0])
#     x = BatchNormalization()(x)
#     x = MaxPool2D((4, 1))(x)
#     x = incept_block(x, conv_filter[1])
#     x = SE_block(x, conv_filter[1])
#     x = BatchNormalization()(x)
#     x = MaxPool2D((4, 1))(x)
#     x = incept_block(x, conv_filter[2])
#     x = SE_block(x, conv_filter[2])
#     x = BatchNormalization()(x)
#     x = MaxPool2D((4, 1))(x)
#     x = SeparableConv2D(conv_filter[3], (1, 1),
#                         padding='same', activation='relu')(x)
#     x = BatchNormalization()(x)
#     x = MaxPool2D((2, 1))(x)
#     x = Reshape((63, conv_filter[-1]))(x)
#     for i in range(RNN_repeat):
#         x = Bidirectional(
#             RNN_Layer(conv_filter[-1], return_sequences=True), merge_mode='ave')(x)
#     x = GlobalAveragePooling1D()(x)
#     output = Dense(class_num, activation='softmax')(x)
#     return Model(inputs, output)

model = get_model()
sgd = keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.03)
adadelta = keras.optimizers.Adadelta(
    lr=1, rho=0.95, epsilon=1e-08, decay=0.0)
adam = keras.optimizers.Adam()
optimizer = [sgd, adadelta, adam][1]

model.compile(optimizer=optimizer, loss=keras.losses.categorical_crossentropy,
              metrics=[keras.metrics.categorical_accuracy])
model.summary()

model_name = "Res_expand_GRU"
model_folder = os.path.join(
    "/home/philip/data/Keyword_spot/saved_model/", model_name)
if not os.path.isdir(model_folder):
    os.mkdir(model_folder)

with open(os.path.join(model_folder, 'SUMMARY.txt'), 'w') as f:
    f.write(str(training_gen.class_indices))

circle = 5
for i in range(circle):
    print("circle %d out of %d" % ((i + 1), circle))
    history = model.fit_generator(
        training_gen, steps_per_epoch=training_gen.n // training_gen.batch_size + 1, epochs=5)
    model.save(os.path.join(model_folder, "%s_%d.h5" % (model_name, (i + 1))))

# import predict_test
# import importlib
# importlib.reload(predict_test)
# predict_test.predict_test(model,training_gen.class_indices,0.3)
# def incept_block(inputs,filters):
#     a = [Conv2D(int(filters*i),(1,1),padding='same',activation='relu')(inputs) for i in [1,0.5,0.75,0.75]]
#     a = [SeparableConv2D(filters)]
#     b = Concatenate()(a)
#     return b
