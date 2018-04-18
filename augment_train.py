import glob

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
from keras.applications import MobileNet


data_root = '/home/philip/data/Keyword_spot/'
batch_size = 128
training_data_gen = image.ImageDataGenerator(width_shift_range=0.1)
training_gen = training_data_gen.flow_from_directory(data_root + "train/augmented_1", class_mode="categorical",
                                                     target_size=(128, 63),
                                                     batch_size=batch_size,
                                                     color_mode="grayscale")
model_name = "augmented_train"
model_folder = os.path.join(
    "/home/philip/data/Keyword_spot/saved_model/", model_name)
if not os.path.isdir(model_folder):
    os.mkdir(model_folder)
# models_path = glob.glob(data_root+"saved_model/ensemble/*.h5")
models_path = glob.glob(data_root+"saved_model/augmented_train/*.h5")

for path in models_path:
    model_name = str(os.path.basename(path))[:-3]
    print(model_name)
    model = keras.models.load_model(path)
    # config = model.to_json()
    # model = keras.models.model_from_json(config)
    sgd = keras.optimizers.SGD(lr=0.01, momentum=0.9)
    adadelta = keras.optimizers.Adadelta(
        lr=1, rho=0.95, epsilon=1e-08, decay=0.0)
    adam = keras.optimizers.Adam()
    optimizer = [sgd, adadelta, adam][1]
    model.compile(optimizer=optimizer, loss=keras.losses.categorical_crossentropy,
                  metrics=[keras.metrics.categorical_accuracy])
    circle = 1
    for i in range(circle):
        print("circle %d out of %d" % ((i + 1), circle))
        history = model.fit_generator(
            training_gen, steps_per_epoch=training_gen.n // training_gen.batch_size + 1, epochs=3)
        # model.save(os.path.join(model_folder, "%s_%d.h5" % (model_name, (i + 1))))
        model.save(path)
    del model
# import predict_test
# import importlib
# importlib.reload(predict_test)
# predict_test.predict_test(model,training_gen.class_indices,0.2)

