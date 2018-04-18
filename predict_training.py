import os
import cv2
import glob
import tqdm
import csv
import numpy as np
import keras
import pickle


# In[3]:


data_root= '/home/philip/data/Keyword_spot/'
valid_command = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence', 'unknown']

model_folder = os.path.join(data_root,'saved_model','ensemble')

class_indices = {'bed': 0, 'bird': 1, 'cat': 2, 'dog': 3, 'down': 4, 'eight': 5, 'five': 6, 'four': 7, 'go': 8, 'happy': 9, 'house': 10, 'left': 11, 'marvin': 12, 'nine': 13, 'no': 14, 'off': 15, 'on': 16, 'one': 17, 'right': 18, 'seven': 19, 'sheila': 20, 'silence': 21, 'six': 22, 'stop': 23, 'three': 24, 'tree': 25, 'two': 26, 'up': 27, 'wow': 28, 'yes': 29, 'zero': 30}
class_indice = list(class_indices.keys())


# In[6]:


def predict_and_save_result(load_mode,model_name,model=None,json_file=None,weight=None,batch_size = 256):
    if load_mode == 'model' and model!=None:
        pass
    elif load_mode == 'h5':
        model = keras.models.load_model(model)
    elif load_mode == 'json':
        with open(json_file,'r') as jsonfile:
            recover = json.load(jsonfile)
        model = keras.models.model_from_json(recover)
        model.load_weights(weight)
    test_files = glob.glob(data_root+"train/original_all/***/*.png")
    total_file = len(test_files)
    record_dic = {}
    record_path = os.path.join(data_root,'record_training/',"%s_predict.pickle"%(model_name))
    if os.path.isfile(record_path):
        return
    def record_prediction(file_name,prediction):
#         file_name = os.path.basename(file_name)
        record_dic[file_name] = prediction
    for start in tqdm.tqdm(range(0,total_file,batch_size)):
        end = min(start+batch_size,total_file)
        batch_files =test_files[start:end]
        im = np.array([cv2.imread(i,0) for i in batch_files])
        im = np.expand_dims(im,-1)
        result = model.predict_on_batch(im)
        [record_prediction(*_) for _ in zip(batch_files,result)]
    with open(record_path,'wb') as f:
        pickle.dump(record_dic,f,protocol=pickle.HIGHEST_PROTOCOL)
    del model




for path in glob.glob(data_root+'saved_model/ensemble/*.h5'):
    model_name = str(os.path.basename(path))[:-3]
    print(model_name)
    predict_and_save_result('h5',model_name,path)