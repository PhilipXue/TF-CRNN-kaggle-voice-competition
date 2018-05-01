import keras
import argparse
from pathlib import Path
from keras_model.models import ResNetModel, InceptNetModel
from keras.preprocessing import image
from keras.layers import Input
from config import training_img_data_folder
# parse argument
parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', type=int, default=64)
parser.add_argument('-l', '--learning_rate', type=float, default=0.01)
parser.add_argument('-t', '--training_circles', type=int, default=5)
parser.add_argument('-o', '--optimizer', type=str, default='adadelta')
args = parser.parse_args()
# get configuration for training from argument parser
batch_size = args.batch_size
opti = args.optimizer
learning_rate = args.learning_rate
circle = args.training_circles
# specific training set location
training_data_gen = image.ImageDataGenerator(width_shift_range=0.1)
training_gen = training_data_gen.flow_from_directory(training_img_data_folder, class_mode="categorical",
                                                     target_size=(128, 63),
                                                     batch_size=batch_size,
                                                     color_mode="grayscale")
# construct model
input_layer = Input((128, 63, 1))
resnet = ResNetModel(class_num=training_gen.num_classes)
output_layer = resnet.build(input_layer)
model = keras.models.Model(input_layer, output_layer)
# config model on optimizer, metirc and loss function
sgd = lambda learning_rate: keras.optimizers.SGD(
    lr=learning_rate, momentum=0.9, decay=0.03)
adadelta = lambda learning_rate: keras.optimizers.Adadelta(
    lr=learning_rate, rho=0.95, epsilon=1e-08, decay=0.0)
adam = keras.optimizers.Adam()
optimizer_list = {'sgd': sgd, 'adadelta': adadelta, 'adam': adam}
optimizer = optimizer_list[opti]
model.compile(optimizer=optimizer(learning_rate), loss=keras.losses.categorical_crossentropy,
              metrics=[keras.metrics.categorical_accuracy])
# determine where to save the model
model_name = "Res_expand_GRU"
model_save_root = Path("/home/philip/data/Keyword_spot/saved_model/")
model_folder = model_save_root / model_name
model_folder.mkdir(parents=True, exist_ok=True)
# save for every 5 epochs, and train for total of configed circles
for i in range(circle):
    print("circle %d out of %d" % ((i + 1), circle))
    history = model.fit_generator(
        training_gen, steps_per_epoch=training_gen.n // training_gen.batch_size + 1, epochs=5)
    model.save(os.path.join(model_folder, "%s_%d.h5" % (model_name, (i + 1))))
