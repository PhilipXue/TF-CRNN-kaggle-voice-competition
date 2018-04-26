import librosa
import numpy as np
# import cv2
from glob import glob
import tqdm
import os
from pathlib import Path
data_root = Path('D:/Keyword_spot')
audio_data_folder = data_root / 'train/audio'

print(audio_data_folder.exists())