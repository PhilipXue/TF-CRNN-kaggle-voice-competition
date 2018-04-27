import keras
import os
import glob
import numpy as np
import pickle

class_indices = {'bed': 0, 'bird': 1, 'cat': 2, 'dog': 3, 'down': 4, 'eight': 5, 'five': 6, 'four': 7, 'go': 8, 'happy': 9,
                 'house': 10, 'left': 11, 'marvin': 12, 'nine': 13, 'no': 14, 'off': 15, 'on': 16, 'one': 17, 'right': 18,
                 'seven': 19, 'sheila': 20, 'silence': 21, 'six': 22, 'stop': 23, 'three': 24, 'tree': 25, 'two': 26,
                 'up': 27, 'wow': 28, 'yes': 29, 'zero': 30}

data_root = "/home/philip/data/Keyword_spot/"


def get_label_from_filename(full_filename):
    dirname = os.path.dirname(full_filename)
    label = dirname.split("/")[-1]
    label_indice = class_indices[label]
    one_hot_encode = keras.utils.to_categorical(
        label_indice, len(class_indices))
    return one_hot_encode


record_file = glob.glob(data_root + "record_training/*.pickle")


def read_record(file_name):
    with open(file_name, 'rb') as f:
        record = pickle.load(f)
        return record


records = [read_record(_) for _ in record_file]


def gather_data(file_name):
    #     file_name = os.path.basename(file_name)
    gather_record = np.array([record[file_name] for record in records])
    label = get_label_from_filename(file_name)
    return gather_record, label


test_files = glob.glob(data_root+"train/original_all/***/*.png")

training_data = [gather_data(file_name) for file_name in test_files]
