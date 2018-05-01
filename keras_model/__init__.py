import keras
from keras import Model
from keras.layers import (
    Conv2D, MaxPool2D, BatchNormalization, Input, AveragePooling1D)
from keras.layers import (SeparableConv2D, MaxPool1D, GlobalAveragePooling2D,
                          Dropout, GlobalAveragePooling1D, GlobalMaxPooling1D)
from keras.layers import CuDNNGRU as GRU
from keras.layers import CuDNNLSTM as LSTM
from keras.layers import (Reshape, Bidirectional, Flatten,
                          Dense, Permute, Multiply, Average, Concatenate, Add)
import numpy as np

conv_size = (3, 3)
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


def Res_block(inputs, filters, se_block=False):
    y = Conv2D(filters, (1, 1), padding='same')(inputs)
    x = SeparableConv2D(filters, (3, 3), padding='same',
                        dilation_rate=(1, 1))(inputs)
    x = SeparableConv2D(filters, (3, 3), padding='same',
                        dilation_rate=(1, 1))(x)
    if se_block:
        x = SE_block(x, filters)
    x = Add()([y, x])
    x = SeparableConv2D(filters, (7, 7), padding='same',
                        dilation_rate=(1, 1))(x)
    x = SeparableConv2D(filters, (7, 7), padding='same',
                        dilation_rate=(1, 4))(x)
    if se_block:
        x = SE_block(x, filters)
    o = Add()([y, x])
    return o
