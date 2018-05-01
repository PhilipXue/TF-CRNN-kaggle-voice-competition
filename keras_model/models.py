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
from keras_model import SE_block, Res_block, incept_block
conv_filter = [8, 16, 32, 64, 128, 256]


class Model(object):
    def __init__(self, class_num, RNN_repeat=2, RNN_layer=GRU, attention=False):
        self.class_num = class_num
        self.RNN_layer = RNN_layer
        self.RNN_repeat = RNN_repeat
        self.attention = attention

    def build(self, input_layer):
        pass


class ResNetModel(Model):
    def build(self, input_layer):
        x = Res_block(input_layer, conv_filter[0])
        x = BatchNormalization()(x)
        x = MaxPool2D((4, 1))(x)
        x = Res_block(x, conv_filter[1])
        x = BatchNormalization()(x)
        x = MaxPool2D((2, 1))(x)
        x = Res_block(x, conv_filter[2])
        x = BatchNormalization()(x)
        x = MaxPool2D((2, 1))(x)
        x = Res_block(x, conv_filter[3])
        x = BatchNormalization()(x)
        x = MaxPool2D((2, 1))(x)
        x = Res_block(x, conv_filter[4])
        x = BatchNormalization()(x)
        x = MaxPool2D((2, 1))(x)
        x = Res_block(x, conv_filter[5])
        x = BatchNormalization()(x)
        x = MaxPool2D((2, 1))(x)
        x = Reshape((63, conv_filter[-1]))(x)
        if self.attention:
            y = Permute((2, 1))(x)
            x = Dense(63, activation='softmax')(y)
            x = Multiply()([y, x])
            x = Permute((2, 1))(x)
        for i in range(self.RNN_repeat):
            x = Bidirectional(
                self.RNN_layer(conv_filter[-1], return_sequences=True), merge_mode='ave')(x)
        x = GlobalMaxPooling1D()(x)
        output_layer = Dense(self.class_num, activation='softmax')(x)
        return output_layer


class InceptNetModel(Model):
    def build(self, input_layer):
        x = incept_block(input_layer, conv_filter[0])
        x = SE_block(x, conv_filter[0])
        x = BatchNormalization()(x)
        x = MaxPool2D((4, 1))(x)
        x = incept_block(x, conv_filter[1])
        x = SE_block(x, conv_filter[1])
        x = BatchNormalization()(x)
        x = MaxPool2D((4, 1))(x)
        x = incept_block(x, conv_filter[2])
        x = SE_block(x, conv_filter[2])
        x = BatchNormalization()(x)
        x = MaxPool2D((4, 1))(x)
        x = SeparableConv2D(conv_filter[3], (1, 1),
                            padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D((2, 1))(x)
        x = Reshape((63, conv_filter[-1]))(x)
        for i in range(RNN_repeat):
            x = Bidirectional(
                self.RNN_layer(conv_filter[-1], return_sequences=True), merge_mode='ave')(x)
        x = GlobalAveragePooling1D()(x)
        output_layer = Dense(self.class_num, activation='softmax')(x)
        return output_layer
