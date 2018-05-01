import platform
import os
if platform.system() == 'Linux':
    data_root = Path('/home/philip/data/Keyword_spot')
else:
    data_root = Path('D:/Keyword_spot')
audio_data_folder = data_root / 'train/audio'
all_audio_data = audio_data_folder.glob("*/*.wav")
audio_file_num = len(all_audio_data)
img_data_folder = data_root / 'train/trian_image'
if not img_data_folder.exists():
    os.mkdir(img_data_folder)
