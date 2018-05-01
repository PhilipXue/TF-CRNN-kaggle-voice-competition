import os
import pickle
import numpy as np
import glob
import tqdm
data_root = '/home/philip/data/Keyword_spot/'
valid_command = ['yes', 'no', 'up', 'down', 'left', 'right',
                 'on', 'off', 'stop', 'go', 'silence', 'unknown']

class_indices = {'bed': 0, 'bird': 1, 'cat': 2, 'dog': 3, 'down': 4, 'eight': 5, 'five': 6, 'four': 7, 'go': 8, 'happy': 9, 'house': 10, 'left': 11, 'marvin': 12, 'nine': 13, 'no': 14,
                 'off': 15, 'on': 16, 'one': 17, 'right': 18, 'seven': 19, 'sheila': 20, 'silence': 21, 'six': 22, 'stop': 23, 'three': 24, 'tree': 25, 'two': 26, 'up': 27, 'wow': 28, 'yes': 29, 'zero': 30}
class_indice = list(class_indices.keys())
ensemble_record_path = data_root+"resemble_record/ensemble_10_89.pickle"
with open(ensemble_record_path,'rb') as f:
    record = pickle.load(f)

test_files = glob.glob(data_root + 'test/test_image/*.png')
prob = 0
for file in tqdm.tqdm(test_files):
    key = os.path.basename(file).replace('png', 'wav')
    ensumbled = record[key]
    predict = class_indice[np.argmax(ensumbled)]
    possibility = np.max(ensumbled)
    if (predict in valid_command and possibility > 0.95) or (predict not in valid_command and possibility > 0.9):
        ori = file
        dst = data_root + 'train/domain_adapt/' + \
            predict+'/' + os.path.basename(file)
        dst_dir = os.path.dirname(dst)
        if not os.path.isdir(dst_dir):
            os.mkdir(dst_dir)
        cmd = "cp %s %s"%(ori, dst)
        prob += 1
        os.system(cmd)
# print(prob)