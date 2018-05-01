
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import os
import cv2
import glob
import tqdm
import csv
import numpy as np
import keras
import pickle


# # In[3]:


data_root= '/home/philip/data/Keyword_spot/'
valid_command = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence', 'unknown']

model_folder = os.path.join(data_root,'saved_model','ensemble')

class_indices = {'bed': 0, 'bird': 1, 'cat': 2, 'dog': 3, 'down': 4, 'eight': 5, 'five': 6, 'four': 7, 'go': 8, 'happy': 9, 'house': 10, 'left': 11, 'marvin': 12, 'nine': 13, 'no': 14, 'off': 15, 'on': 16, 'one': 17, 'right': 18, 'seven': 19, 'sheila': 20, 'silence': 21, 'six': 22, 'stop': 23, 'three': 24, 'tree': 25, 'two': 26, 'up': 27, 'wow': 28, 'yes': 29, 'zero': 30}
class_indice = list(class_indices.keys())




def predict_and_save_result(load_mode,model_name,model=None,json_file=None,weight=None,batch_size = 256):
    record_path = os.path.join(data_root,'record_prediction/',"%s_predict.pickle"%(model_name))
    if os.path.isfile(record_path):
        return
    if load_mode == 'model' and model!=None:
        pass
    elif load_mode == 'h5':
        model = keras.models.load_model(model)
    elif load_mode == 'json':
        with open(json_file,'r') as jsonfile:
            recover = json.load(jsonfile)
        model = keras.models.model_from_json(recover)
        model.load_weights(weight)
    test_files = glob.glob(data_root+"test/test_image/*.png")
    total_file = len(test_files)
    record_dic = {}
    def record_prediction(file_name,prediction):
        file_name = (os.path.basename(file_name)).replace('png','wav')
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



import os
import glob
import csv
import numpy as np
import pickle


# record_file = glob.glob(data_root+"record_prediction/*.pickle")
record_file = glob.glob(data_root+"record_prediction/*.pickle")
record_dict = {}
def ensemble(records,weight):
    csvfile = open("new_submission.csv", 'w')
    fieldnames = ['fname', 'label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    test_files = glob.glob(data_root+"test/audio/*.wav")
    order = 0.5
    thresh_hold = 0.1
    for file_name in tqdm.tqdm(test_files):
        file_name = os.path.basename(file_name)
        gather_record = np.array([record[file_name] for record in records])
        gather_record = gather_record*weight
        if len(gather_record) != 1:
            ensumbled = np.mean(np.power(gather_record,order),axis=0)
        else:
            ensumbled = gather_record
        ensumbled = ensumbled/np.sum(ensumbled)
        record_dict[file_name] = ensumbled
        predict = class_indice[np.argmax(ensumbled)]
        possibility = np.max(ensumbled)
        if predict not in valid_command or (predict != 'silence' and possibility < thresh_hold):
            predict = "unknown"
        writer.writerow({"fname": file_name, "label": predict})
    csvfile.close()


# import keras
# model = keras.models.load_model('ensemble.h5')

# def ensemble(records):
#     csvfile = open("stack_submission.csv", 'w')
#     fieldnames = ['fname', 'label']
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#     writer.writeheader()
#     test_files = glob.glob(data_root+"test/audio/*.wav")
#     order = 0.5
#     thresh_hold = 0.1
#     for file_name in tqdm.tqdm(test_files):
#         file_name = os.path.basename(file_name)
#         gather_record = np.array([record[file_name] for record in records])
#         gather_record = np.expand_dims(gather_record,0)
#         gather_record = np.expand_dims(gather_record,-1)
#         ensumbled = model.predict(gather_record)
#         ensumbled = ensumbled/np.sum(ensumbled)
#         record_dict[file_name] = ensumbled
#         predict = class_indice[np.argmax(ensumbled)]
#         possibility = np.max(ensumbled)
#         if predict not in valid_command:
#             predict = "unknown"
#         writer.writerow({"fname": file_name, "label": predict})
#     csvfile.close()

# record_file = glob.glob(data_root+"record_prediction/*.pickle")
weight = np.ones_like(record_file)
def determin_weight(file_name):
    file_name = os.path.basename(file_name)
    score = int(file_name.split('_')[-2])
    if score>86:
        return 1.5
    elif score ==86:
        return 1.
    else:
        return 0.7
weight =  np.array([list(map(determin_weight,record_file)) for _ in range(31)]).T
print(weight[:,0])
record_dict = {}
def read_record(file_name):
    with open(file_name,'rb') as f:
        record = pickle.load(f)
        return record
records = [read_record(_) for _ in record_file]
ensemble(records,weight)

# ensemble_record_path = data_root+"resemble_record/ensemble_9_88.pickle"
# with open(ensemble_record_path,'wb') as f:
#     pickle.dump(record_dict,f,protocol=pickle.HIGHEST_PROTOCOL)

