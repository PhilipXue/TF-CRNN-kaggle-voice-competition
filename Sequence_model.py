
# coding: utf-8

# In[1]:

import librosa
import keras
from keras.models import Model
from keras.layers import Conv2D,MaxPool2D,BatchNormalization,Input,Activation,MaxPool1D,Conv1D
from keras.layers import CuDNNGRU as GRU
from keras.layers import CuDNNLSTM as LSTM
from keras.layers import Reshape, Bidirectional,Flatten,Dense,InputLayer,Permute,Multiply


# In[2]:

import tensorflow as tf
import numpy as np
import tqdm
import glob
import os


# In[3]:

data_root = "/home/philip/data/Keyword_spot/"


# In[4]:

class_indices = {'bed': 0, 'bird': 1, 'cat': 2, 'dog': 3, 'down': 4, 'eight': 5, 'five': 6, 'four': 7, 'go': 8, 'happy': 9, 
                 'house': 10, 'left': 11, 'marvin': 12, 'nine': 13, 'no': 14, 'off': 15, 'on': 16, 'one': 17, 'right': 18,
                 'seven': 19, 'sheila': 20, 'silence': 21, 'six': 22, 'stop': 23, 'three': 24, 'tree': 25, 'two': 26,
                 'up': 27, 'wow': 28, 'yes': 29, 'zero': 30}


# In[5]:

load_audio = True
def get_label_from_filename(full_filename):
    dirname = os.path.dirname(full_filename)
    label = dirname.split("/")[-1]
    if label == 'silence_bg':
        label ='silence'
    label_indice = class_indices[label]
    one_hot_encode = keras.utils.to_categorical(label_indice,len(class_indices))
    return one_hot_encode
def get_sequence_from_file(filename):
    y,sr = librosa.load(filename,sr=16000)
    if load_audio:
        if len(y)<16000:
            pad_with = (16000-len(y))//2+1
            y = np.pad(y,pad_with,'constant')[:16000]
        return y
    else:
        mel = librosa.feature.melspectrogram(y,sr,hop_length=256)
        db = librosa.power_to_db(mel, ref=np.min)
        return db


# In[6]:

batch_size = 256


# In[7]:

all_files = glob.glob(data_root+"train/audio/***/*.wav") + glob.glob(data_root+"train/silence/*.wav") + glob.glob(data_root+"train/silence_bg/*.wav")
file_number = len(all_files)
step = file_number//batch_size+1

def data_gen(batch_size):
    while True:
        for i in range(step):
            start = i*batch_size
            end = min((i+1)*batch_size,file_number)
            file_batch = all_files[start:end]
            label = np.array([get_label_from_filename(file) for file in file_batch])
            seq = np.array([get_sequence_from_file(file) for file in file_batch])
            seq = np.expand_dims(seq,2)
            yield seq, label


# In[10]:

attention = False

def conv_1d_model(inputs):
    x = Conv1D(8,3,padding='same',activation='relu', strides=1)(inputs)
    x = BatchNormalization()(x)
    x = MaxPool1D(pool_size=(4))(x)
    x = Conv1D(16,3,padding='same', activation='relu',dilation_rate=2)(x)
    x = BatchNormalization()(x)
    x = MaxPool1D(pool_size=(4))(x)
    x = Conv1D(32,3,padding='same', activation='relu',dilation_rate=2)(x)
    x = BatchNormalization()(x)
    x = MaxPool1D(pool_size=(4))(x)
    x = Conv1D(64,3,padding='same', activation='relu',dilation_rate=2)(x)
    x = BatchNormalization()(x)
    x = MaxPool1D(pool_size=(4))(x)
    x = Conv1D(128,3,padding='same', activation='relu',dilation_rate=4)(x)
    x = BatchNormalization()(x)
    x = Conv1D(256,3,padding='same', activation='relu',dilation_rate=4)(x)
    x = BatchNormalization()(x)
    x = Conv1D(512,3,padding='same', activation='relu',dilation_rate=4)(x)
    x = BatchNormalization()(x)
    x = Bidirectional(GRU(128,return_sequences=True),merge_mode='concat')(x)
    x = Bidirectional(GRU(128,return_sequences=True),merge_mode='concat')(x)        
    if attention:
        a = Permute((2,1))(x)
        a = Dense(158, activation='softmax',input_dim=(158,))(a)
        a = Permute((2,1))(a)
        x = Multiply()([x,a])
    x = keras.layers.GlobalMaxPool1D()(x)
    # x = Flatten()(x)
    x = Dense(31,activation='softmax')(x)
    return x


# In[11]:

input_layer = Input((16000,1))
output = conv_1d_model(input_layer)
model = Model(input_layer,output)


# In[12]:

model.summary()


# In[13]:

rmsprop = keras.optimizers.RMSprop()
model.compile(optimizer=rmsprop, loss=keras.losses.categorical_crossentropy,
              metrics=[keras.metrics.categorical_accuracy])


# In[14]:

data = data_gen(batch_size=batch_size)


# In[19]:

model_name = '1DCRNN_with_dialation'
model_folder = os.path.join(
    "/home/philip/data/Keyword_spot/saved_model/", model_name)
if not os.path.isdir(model_folder):
    os.mkdir(model_folder)


# In[20]:

for i in range(5):
    model_save_path = os.path.join(model_folder,"%s_%d.h5"%(model_name,i))
    print(model_save_path)
    model.fit_generator(data,steps_per_epoch=step,epochs=10,verbose=1,use_multiprocessing=True)
    model.save(model_save_path)


# In[ ]:



