import platform
import os
if platform.system() == 'Linux':
    data_root = Path('/home/philip/data/Keyword_spot')
else:
    data_root = Path('D:/Keyword_spot')
training_audio_data_folder = data_root / 'train/audio'
training_audio_data = training_audio_data_folder.glob("*/*.wav")
training_img_data_folder = data_root / 'train/trian_image'
if not training_img_data_folder.exists():
    os.mkdir(training_img_data_folder)
test_audio_data_folder = data_root / 'test/audio'
test_audio_data = test_audio_data_folder.glob("*.wav")
test_image_data_folder = data_root / 'test/test_image'
if not test_image_data_folder.exists():
    os.mkdir(test_image_data_folder)
