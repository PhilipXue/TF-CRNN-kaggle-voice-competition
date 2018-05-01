import platform
from pathlib import Path
if platform.system() == 'Linux':
    data_root = Path('/home/philip/data/Keyword_spot')
else:
    data_root = Path('D:/Keyword_spot')
training_audio_data_folder = data_root / 'train/audio'
training_audio_data = training_audio_data_folder.glob("*/*.wav")
training_img_data_folder = data_root / 'train/trian_image'
training_img_data_folder.mkdir(parents=True, exist_ok=True)
test_audio_data_folder = data_root / 'test/audio'
test_audio_data = test_audio_data_folder.glob("*.wav")
test_image_data_folder = data_root / 'test/test_image'
test_image_data_folder.mkdir(parents=True, exist_ok=True)
