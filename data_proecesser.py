import librosa
import numpy as np
import tqdm
import os
import functools
from pathlib import Path
import cv2
import tqdm
from config import data_root, audio_data_folder, img_data_folder, all_audio_data


@functools.lru_cache()
def _load_audio(audio_file_name):
    y, sr = librosa.load(audio_file_name, sr=None)
    return y, sr


def audio_file_to_mel_spectrum(audio_file_name, hop_length=512, sr=22050):
    y, ori_sr = _load_audio(audio_file_name)
    y = librosa.core.resample(cut_y, orig_sr=ori_sr, target_sr=sr)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, hop_length=hop_length)
    return mel


def mel_to_db(mel):
    db = librosa.power_to_db(mel, ref=np.min)
    db -= db.min()
    db *= 255 / (db.max()**(db.max() != 0))
    if db.shape[1] != 63:
        db = cv2.resize(db, (db.shape[0], 63))
    return db


def audio_filename_to_img_filename(audio_file_name):
    category = audio_file_name.parent.name
    img_file = audio_file_name.name.replace('.wav', '.jpg')
    img_category_folder = img_data_folder / category
    if not img_folder.exists():
        os.mkdir(img_category_folder)
    img_filename = img_category_folder / img_file
    return img_filename


def process_and_save(all_audio_files):
    for audio_file in tqdm.tqdm(all_audio_files):
        mel = audio_file_to_mel_spectrum(audio_file)
        db_img = mel_to_db(mel)
        cv2.imwrite(audio_filename_to_img_filename(audio_file), db_img)


if __name__ == '__main__':
    process_and_save(all_audio_files=all_audio_data)
