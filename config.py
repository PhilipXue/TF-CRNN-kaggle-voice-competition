import platform
from pathlib import Path
if platform.system() == 'Linux':
    data_root = Path('/home/philip/data/Keyword_spot')
else:
    data_root = Path('D:/Keyword_spot')
training_audio_data_folder = data_root / 'train/audio'
training_audio_data = training_audio_data_folder.glob("*/*.wav")
training_img_data_folder = data_root / 'train/train_image'
training_img_data_folder.mkdir(parents=True, exist_ok=True)
test_audio_data_folder = data_root / 'test/audio'
test_audio_data = test_audio_data_folder.glob("*.wav")
test_image_data_folder = data_root / 'test/test_image'
test_image_data_folder.mkdir(parents=True, exist_ok=True)

class_indices = {'bed': 0, 'bird': 1, 'cat': 2, 'dog': 3, 'down': 4,
                 'eight': 5, 'five': 6, 'four': 7, 'go': 8, 'happy': 9,
                 'house': 10, 'left': 11, 'marvin': 12, 'nine': 13, 'no': 14,
                 'off': 15, 'on': 16, 'one': 17, 'right': 18, 'seven': 19,
                 'sheila': 20, 'silence': 21, 'six': 22, 'stop': 23, 'three': 24,
                 'tree': 25, 'two': 26, 'up': 27, 'wow': 28, 'yes': 29, 'zero': 30}

preprocess_size = (128, 72)
image_size = (128, 63, 1)
