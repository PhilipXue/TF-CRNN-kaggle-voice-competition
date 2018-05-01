import librosa
import numpy as np
import tqdm
import os
import functools
from pathlib import Path
import cv2
import tqdm
from config import (data_root, training_audio_data_folder,
                    training_img_data_folder, test_audio_data)


def audio_file_to_mel_spectrum(audio_file_name, hop_length=512, sr=22050):
    '''
    Get an audio fiel its file name and convert it into two dimensions np array
    '''
    y, ori_sr = librosa.load(audio_file_name, sr=None)
    y = librosa.core.resample(cut_y, orig_sr=ori_sr, target_sr=sr)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, hop_length=hop_length)
    db = librosa.power_to_db(mel, ref=np.min)
    db -= db.min()
    db *= 255 / (db.max()**(db.max() != 0))
    if db.shape[1] != 63:
        db = cv2.resize(db, (db.shape[0], 63))
    return db


def process_data_and_save(audio_files, training):
    for audio_file_name in tqdm.tqdm(audio_files):
        db_img = audio_file_to_mel_spectrum(audio_file_name)
        # convert audio file name to the corresponding image file name
        img_file = audio_file_name.name.replace('.wav', '.jpg')
        if training:
            category = audio_file_name.parent.name
            img_category_folder = training_img_data_folder / category
            if not img_folder.exists():
                os.mkdir(img_category_folder)
        else:
            img_category_folder = test_image_data_folder
        img_filename = img_category_folder / img_file
        cv2.imwrite(img_filename, db_img)


if __name__ == '__main__':
    process_data_and_save(audio_files=training_audio_data, training=True)
    process_data_and_save(audio_files=test_audio_data, training=False)
