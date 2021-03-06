from sklearn.model_selection import train_test_split
import tensorflow as tf
from config import test_image_data_folder, training_img_data_folder, class_indices, preprocess_size, image_size
from functools import reduce
import os


def preporcess_image(image):
    process_it = lambda content, function: function(content)
    process_chain = [
        lambda x:tf.image.resize_images(x, preprocess_size),
        lambda x:tf.random_crop(x, image_size),
        tf.image.random_flip_left_right,
        tf.image.random_flip_up_down,
        tf.to_float
    ]
    return reduce(process_it, process_chain, image)


def decode_img(file_name, label):
    file_content = tf.read_file(file_name)
    # channel have to be implicitly specific to 3 to let the training process know about the input shape
    # content = tf.image.decode_jpeg(file_content, channels=3)
    content = tf.image.decode_png(file_content, channels=1)
    image = preporcess_image(content)
    image *= (1 / 255)
    return image, label


def decode_img_test(file_name, label=None):
    file_content = tf.read_file(file_name)
    content = tf.image.decode_jpeg(file_content, channels=1)
    # image = tf.image.resize_images(content, (image_size, image_size))
    image = tf.to_float(content)
    image *= (1 / 255)
    if label == None:
        return image
    else:
        return image, label


def get_label_from_filename(full_filename):
    dirname = full_filename.parent.name
    label = class_indices[dirname]
    return label


def create_train_dataset(img, label, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((img, label))
    dataset = dataset.map(decode_img, num_parallel_calls=os.cpu_count()).prefetch(tf.contrib.data.AUTOTUNE).batch(
        batch_size=batch_size)
    return dataset


def create_valid_dataset(img, label, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((img, label))
    dataset = dataset.map(decode_img_test, num_parallel_calls=os.cpu_count()).prefetch(tf.contrib.data.AUTOTUNE).batch(
        batch_size=batch_size)
    return dataset


class training_valid_dataset(object):
    def __init__(self, batch_size, image_size, valid_split=0.1, random_seed=42):
        self.batch_size = batch_size
        self.image_size = image_size
        training_img = list(training_img_data_folder.glob('*/*.png'))
        training_label = list(map(get_label_from_filename, training_img))
        training_img = list(map(str, training_img))
        self.training_img, self.test_img, self.training_label, self.test_label = train_test_split(
            training_img, training_label, test_size=valid_split, random_state=random_seed)

    def training_input_fn(self):
        return create_train_dataset(self.training_img, self.training_label, self.batch_size).repeat()

    def test_input_fn(self):
        return create_valid_dataset(self.test_img, self.test_label, self.batch_size)


class test_dataset(object):
    def __init__(self, batch_size, image_size):
        self.self.batch_size = batch_size
        self.image_size = image_size
        self.test_img = test_image_data_folder.glob('*.jpg')

    def test_input_fn(self):
        dataset = tf.data.Dataset.from_tensor_slices(
            self.test_img)
        dataset = dataset.map(decode_img_test, num_parallel_calls=num_parallel_calls).prefetch(
            int(1.2 * batch_size)).batch(self.batch_size)
        return dataset
