import keras
import tqdm
import os
import cv2
import glob
import csv
import numpy as np
from config import test_image_data_folder
valid_command = ['yes', 'no', 'up', 'down', 'left', 'right',
                 'on', 'off', 'stop', 'go', 'silence', 'unknown']

batch_size = 128


def predict_test(model, class_indices, thresh_hold):
    csvfile = open("new_submission.csv", 'w')
    fieldnames = ['fname', 'label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    def replace_and_save_result(file_name, predict, possibility):
        if predict not in valid_command or (predict != 'silence' and possibility < thresh_hold):
            predict = "unknown"
        filen = file_name.name.replace("jpg", "wav")
        writer.writerow({"fname": filen, "label": predict})

    test_files = test_image_data_folder.glob("*.jpg")
    class_indice = list(class_indices.keys())
    for start in tqdm.tqdm(range(0, total_file, batch_size)):
        end = min(start + batch_size, total_file)
        batch_files = test_files[start:end]
        im = np.array([cv2.imread(i, 0) for i in batch_files])
        im = np.expand_dims(im, -1)
        result = model.predict_on_batch(im)
        poss = np.max(result, 1)
        predicts = [class_indice[np.argmax(a)] for a in result]
        [replace_and_save_result(_[0], _[1], _[2])
         for _ in zip(batch_files, predicts, poss)]
    csvfile.close()
