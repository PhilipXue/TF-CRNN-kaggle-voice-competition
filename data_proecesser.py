
import os
import glob
import random
import shutil
import re

import numpy as np

try:
    import cv2
except ImportError:
    pass

from data_process.audio import (
    audio_length, melspectrogram, mel_to_db, melspectrogram_width,
    signal_extract_mask
)

from config import mel_hop_length as get_mel_hop_length

# TODO(wakisaka): Delete all of config value.
from config import (
    RAW_TRAIN_DATA_FOLDER,
    RAW_TEST_DATA_FOLDER,

    DATA_TRAIN_FOLDER,
    DATA_VAL_FOLDER,
    DATA_TEST_FOLDER,
    STEP_LENGTH,
    OVERALL_SET_SIZE,
    VALIDATION_SIZE,
)


def compute_flag(annotations, audio_start, audio_end):
    """If annotations include audio_start and audio_end return 1."""
    flag = 0
    # check flag.
    for annotation in annotations:
        annotation_start, annotation_end = annotation

        if audio_end < annotation_start or audio_start > annotation_end:
            continue

        # only count if the overlap duration is longer tha  10ms
        duration = 10
        if annotation_start + duration < audio_end and annotation_end - duration > audio_start:
            flag = 1
            break

    return flag

def compute_category(annotations, audio_start, audio_end):
    """Returns the category according to the overlapped duration
        not include: 0
        partially include: < 50%: lt50
        partially include: < 100%: lt100
        entirely include: 100%: 100
    """

    if audio_end - audio_start <= 0:
        return '0'

    # check category.
    overlapped = 0
    for annotation in annotations:
        annotation_start, annotation_end = annotation

        if audio_end < annotation_start or audio_start > annotation_end:
            continue

        overlap_start = max(annotation_start, audio_start)
        overlap_end = min(annotation_end, audio_end)

        # entirely overlapped
        if overlap_start == audio_start and overlap_end == audio_end:
            return '100'

        overlapped += overlap_end - overlap_start

    overlapped_rate = (overlapped*100)/(audio_end - audio_start)

    # print("overlapped:{} overlapped-rate:{}".format(overlapped, overlapped_rate))
    if overlapped <= 10:
        return '0'
    elif overlapped_rate < 50:
        return 'lt50'
    else:
        return 'lt100'

def _output_image_path(audio_file_path, start_time, end_time, output_dir_base, flag):
    """Return output image path."""
    filename = "{}_{}_{}_.png".format(start_time, end_time, os.path.basename(audio_file_path))
    output_path = os.path.join(
        output_dir_base, str(flag), filename
    )

    return output_path

def specie_from_raw_audio_path(audio_path):
    dirname = os.path.basename(os.path.dirname(audio_path))

    if dirname == "audio":
        return "crow"

    return dirname


def annotations_from_raw_audio_path(audio_path):
    annotation_file = os.path.basename(audio_path).split(".")[0] + ".Table"
    annotation_dir = os.path.dirname(audio_path).replace("audio", "annotation")
    annotation = open(os.path.join(annotation_dir, annotation_file), "r")
    annotations = []
    for line in annotation:
        sp = line.split(",")
        if sp[1] == "crow":
            start = int(float(sp[0]) * 1000)
            end = int(float(sp[-1]) * 1000)
            annotations.append((start, end))

    return annotations


def _slice_audio_and_save_mel(audio_file_name, annotation_folder,
                              output_data_folder, slice_duration, step_length):
    '''
    This function is to slice the original audio data into fixed length slices
    and convert the annotation to binary label meawhile this function stores
    the original annotaion and audio slice to corresponding folder
    argument :
        audio_fileinput: path of audio file
        filnum: iterable number to assign number to every single file
        slice_duration: length for every slice in millisecond
    '''
    print("Slice: {}".format(audio_file_name))
    mel_hop_length = get_mel_hop_length(slice_duration)
    audiolen = audio_length(audio_file_name)

    # TODO(wakisaka): use annotations_from_raw_audio_path()
    annotation_file = os.path.basename(audio_file_name).split(".")[0] + ".Table"
    annotation = open(os.path.join(annotation_folder, annotation_file), "r")
    annotations = []
    for line in annotation:
        sp = line.split(",")
        if sp[1] == "crow":
            start = int(float(sp[0]) * 1000)
            end = int(float(sp[-1]) * 1000)
            annotations.append((start, end))

    for audio_start in range(0, audiolen, step_length):
        audio_end = audio_start + slice_duration
        # ignore last.
        if audiolen < audio_end:
            break

        flag = compute_flag(annotations, audio_start, audio_end)

        mel = melspectrogram(audio_file_name, mel_hop_length, audio_start/1000, slice_duration/1000)
        db = mel_to_db(mel)

        output_path = _output_image_path(audio_file_name, audio_start, audio_end, output_data_folder, flag)
        cv2.imwrite(output_path, db)
    return


def create_data_folder(target_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        os.mkdir(os.path.join(target_folder, '0'))
        os.mkdir(os.path.join(target_folder, '1'))


def bg_audio_slice_and_save_mel(audio_file, output_data_folder,
                                slice_duration, step_length):
    '''
    Deal with the data in background folder for training process
    '''
    print("Slice: {}".format(audio_file))
    mel_hop_length = get_mel_hop_length(slice_duration)
    audiolen = audio_length(audio_file)
    flag = 0

    for audio_start in range(0, audiolen, step_length):
        audio_end = audio_start + slice_duration
        # ignore last.
        if audiolen < audio_end:
            break
        mel = melspectrogram(audio_file, mel_hop_length, audio_start/1000, slice_duration/1000)
        db = mel_to_db(mel)

        output_path = _output_image_path(audio_file, audio_start, audio_end, output_data_folder, flag)
        cv2.imwrite(output_path, db)
    return


def test_bg_slice_and_save_mel(audio_file, output_data_folder,
                               slice_duration, step_length):
    '''
    Deal with the back ground audio for test, sigal/noise separation conducted
    Please refer the the github issue #3
    '''
    print("Slice: {}".format(audio_file))
    mel_hop_length = get_mel_hop_length(slice_duration)
    audiolen = audio_length(audio_file)

    mask = signal_extract_mask(audio_file, mel_hop_length)

    # TODO(wakisaka): change slice_duration to step_length.
    for start in range(0, audiolen, slice_duration):
        end = start + slice_duration
        # ignore last.
        if audiolen < end:
            break

        start_index = melspectrogram_width(start, mel_hop_length)
        end_index = melspectrogram_width(end, mel_hop_length)

        flag = 0

        if np.any(mask[start_index:end_index] > 0):

            mel = melspectrogram(audio_file, mel_hop_length, start/1000, slice_duration/1000)
            db = mel_to_db(mel)

            output_path = _output_image_path(audio_file, start, end, output_data_folder, flag)

            cv2.imwrite(output_path, db)

    return


def create_test_overall_set(test_folder, size=100):
    '''
    Create the overall folder sampled with the size defined
    for each species
    '''
    overall = os.path.join(test_folder, "overall")
    if os.path.exists(overall):
        shutil.rmtree(overall)

    create_data_folder(overall)

    for species in os.listdir(test_folder):
        if not os.path.isdir(os.path.join(test_folder, species)):
            continue
        if species == 'overall':
            continue

        flag = str(int(species == 'crow'))
        species_folder = os.path.join(test_folder, species)
        files = glob.glob(os.path.join(species_folder, flag, "*.png"))

        rand_list = random.sample(files, min(size, len(files)))
        for src_file in rand_list:
            dist_dir = os.path.join(overall, flag)
            shutil.copy(src_file, dist_dir)


def process_test_data(slice_duration):
    test_image_dir = os.path.join(DATA_TEST_FOLDER, str(slice_duration))
    if os.path.exists(test_image_dir):
        shutil.rmtree(test_image_dir)

    os.makedirs(test_image_dir, exist_ok=True)

    crow_image_dir = os.path.join(test_image_dir, "crow")
    create_data_folder(crow_image_dir)

    # TODO: mp3
    crow_audios = glob.glob(os.path.join(RAW_TEST_DATA_FOLDER, "crow", "audio", "*.wav"))
    annotation_dir = (os.path.join(RAW_TEST_DATA_FOLDER, "crow", "annotation"))

    for crow_audio in crow_audios:
        _slice_audio_and_save_mel(
            crow_audio,
            annotation_dir,
            crow_image_dir,
            slice_duration,
            STEP_LENGTH
        )

    other_audio_dir = os.path.join(RAW_TEST_DATA_FOLDER, "other")
    for species in os.listdir(other_audio_dir):
        species_image_folder = os.path.join(test_image_dir, species)
        species_audio_folder = os.path.join(other_audio_dir, species)

        if not os.path.isdir(species_audio_folder):
            continue
        create_data_folder(species_image_folder)

        bg_audios = glob.glob(os.path.join(species_audio_folder, "*.mp3"))

        for bg_audio in bg_audios:
            test_bg_slice_and_save_mel(
                bg_audio,
                species_image_folder,
                slice_duration,
                STEP_LENGTH)

    create_test_overall_set(test_image_dir, OVERALL_SET_SIZE)


def process_train_data(slice_duration):
    train_dir = os.path.join(DATA_TRAIN_FOLDER, str(slice_duration))
    val_dir = os.path.join(DATA_VAL_FOLDER, str(slice_duration))

    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)

    if os.path.exists(val_dir):
        shutil.rmtree(val_dir)

    create_data_folder(train_dir)
    create_data_folder(val_dir)

    # TODO: mp3
    crow_audios = glob.glob(os.path.join(RAW_TRAIN_DATA_FOLDER, "crow", "audio", "*.wav"))
    annotation_dir = (os.path.join(RAW_TRAIN_DATA_FOLDER, "crow", "annotation"))

    for crow_audio in crow_audios:
        _slice_audio_and_save_mel(
            crow_audio,
            annotation_dir,
            train_dir,
            slice_duration,
            STEP_LENGTH)

    bg_audios = glob.glob(os.path.join(RAW_TRAIN_DATA_FOLDER, "other", "*.mp3"))
    for bg_audio in bg_audios:
        bg_audio_slice_and_save_mel(bg_audio,
                                    train_dir,
                                    slice_duration,
                                    STEP_LENGTH)



def separate_validation_set(slice_duration, validation_size=VALIDATION_SIZE):
    '''
    Create the validation set from training set with the size defined
    for each class
    '''

    data_train_folder = os.path.join(DATA_TRAIN_FOLDER, str(slice_duration))
    data_val_folder = os.path.join(DATA_VAL_FOLDER, str(slice_duration))

    # fix random seed
    random.seed(1)
    classes = [0, 1]

    for class_id in classes:
        train_files = glob.glob(os.path.join(data_train_folder, str(class_id), "*.png"))
        train_files = random.sample(train_files, k=validation_size // 2)

        for train_file in train_files:
            val_file = os.path.join(data_val_folder, str(class_id), os.path.basename(train_file))
            shutil.move(train_file, val_file)