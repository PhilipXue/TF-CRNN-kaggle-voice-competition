from keras.layers import Dense, Flatten, Dropout, Activation
from keras.layers import MaxPool1D, Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Reshape
from keras.layers import CuDNNGRU as GRU
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.layers import Bidirectional

def CNNnetwork(input_layer):
    x = BatchNormalization()(input_layer)
    x = Convolution2D(16, kernel_size=(3, 3), padding='same')(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D(pool_size=(3, 3))(x)

    x = Convolution2D(16, kernel_size=(3, 3), padding='same')(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D(pool_size=(3, 3))(x)

    x = Convolution2D(16, kernel_size=(3, 1), padding='same')(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D(pool_size=(3, 1))(x)

    x = Convolution2D(16, kernel_size=(3, 1), padding='same')(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D(pool_size=(3, 1))(x)

    x = Flatten()(x)

    x = Dense(256)(x)
    x = LeakyReLU()(x)
    x = Dropout(0.5)(x)

    x = Dense(32)(x)
    x = LeakyReLU()(x)
    x = Dropout(0.5)(x)

    x = Dense(30, activation='sigmoid')(x)
    model = Model(input_layer, x)

    return model


def CNNnetwork_BN(input_layer):
    x = Convolution2D(16, kernel_size=(3, 3), padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D(pool_size=(3, 3))(x)

    x = Convolution2D(16, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D(pool_size=(3, 3))(x)

    x = Convolution2D(16, kernel_size=(3, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D(pool_size=(3, 1))(x)

    x = Convolution2D(16, kernel_size=(3, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D(pool_size=(3, 1))(x)

    x = Flatten()(x)

    x = Dense(256)(x)
    x = LeakyReLU()(x)
    x = Dropout(0.5)(x)

    x = Dense(32)(x)
    x = LeakyReLU()(x)
    x = Dropout(0.5)(x)

    x = Dense(1, activation='sigmoid')(x)
    model = Model(input_layer, x)

    return model
# In[8]:


# Network end at second place for the same challenge as above
# 4 conv layers and 2 GRU layers, About 810K parameters
# did not use the same input spetrogram as defined with original one
# Change some parameters to fit with our input.
# Name after the Tempere University


# In[9]:


def CRNNnetwork(input_layer):
    x = Convolution2D(filters=96, kernel_size=(
        5, 5), padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    x = MaxPooling2D(pool_size=(4, 1))(x)

    x = Convolution2D(filters=96, kernel_size=(
        5, 5), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    x = MaxPooling2D(pool_size=(4, 1))(x)

    x = Convolution2D(filters=96, kernel_size=(
        5, 5), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    x = MaxPooling2D(pool_size=(4, 1))(x)

    x = Convolution2D(filters=96, kernel_size=(
        5, 5), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 1))(x)
    image_width = int(x.get_shape()[2])
    x = Reshape((image_width, 96))(x)
    x = Bidirectional(GRU(96, return_sequences=True)(x))
    x = Bidirectional(GRU(96, return_sequences=True)(x))

    x = MaxPool1D(pool_size=image_width)(x)
    x = Flatten()(x)
    x = Dense(30, activation='softmax')(x)
    model = Model(input_layer, x)
    return model


# In[10]:


# Network model from http://ceur-ws.org/Vol-1609/16090547.pdf
# But the  goal of this network is a little different from ours
# and require preprocessing.


# In[11]:


def ETHnetwork(input_layer):
    x = Dropout(0.2)(input_layer)

    x = BatchNormalization()(x)
    x = Convolution2D(64, activation='relu', padding='same', strides=(1, 2),
                      input_shape=(128, 256, 3), kernel_size=(5, 5))(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = BatchNormalization()(x)
    x = Convolution2D(64, kernel_size=(5, 5), activation='relu',
                      strides=(1, 1), padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = BatchNormalization()(x)
    x = Convolution2D(128, kernel_size=(5, 5), activation='relu',
                      strides=(1, 1), padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = BatchNormalization()(x)
    x = Convolution2D(256, kernel_size=(5, 5), activation='relu',
                      strides=(1, 1), padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = BatchNormalization()(x)
    x = Convolution2D(256, kernel_size=(3, 3), activation='relu',
                      strides=(1, 1), padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Flatten()(x)

    x = BatchNormalization()(x)
    x = Dense(1024, kernel_initializer='glorot_uniform', activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(2, kernel_initializer='glorot_uniform',
              activation='softmax')(x)
    model = Model(input_layer, x)
    return model