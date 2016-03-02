
# coding: utf-8

# In[1]:

import os
from scipy.misc import imread, imresize, imsave
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

import pandas as pd
import numpy as np
from keras.utils import np_utils


# In[3]:

image_dir = 'mass_50_padding_dataset'
df = pd.read_csv('mass_case_description.csv')
keys = {'MALIGNANT': 1, 'BENIGN': 0, 'BENIGN_WITHOUT_CALLBACK': 0}

X_train = {}
y_train = {}

X_train['CC'] = []
y_train['CC'] = []

X_train['MLO'] = []
y_train['MLO'] = []

for fn in os.listdir(image_dir):
    s = imresize(imread(image_dir+'/'+fn), (64,64))
    s = s[None]
    
    arr = fn.split('_')
    patient_id = arr[1]
    side = arr[2]
    view = arr[3]
    
    label = df[df['patient_id'] == 'P_'+str(patient_id)][df['side'] == side][df['view'] == view]['pathology'].values[0]
    
    X_train[view].append(s)
    y_train[view].append(keys[label])
    
X_train_CC, y_train_CC = np.array(X_train['CC']), np.array(y_train['CC'])
X_train_MLO, y_train_MLO = np.array(X_train['MLO']), np.array(y_train['MLO'])

Y_train_CC = np_utils.to_categorical(y_train_CC, 2)
Y_train_MLO = np_utils.to_categorical(y_train_CC, 2)


# In[4]:

print X_train_CC.shape, X_train_MLO.shape


# In[5]:

X_train_CC = X_train_CC.astype('float32')
X_train_CC -= X_train_CC.mean()
X_train_CC /= 255


# In[23]:

import keras.callbacks

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.acc = []
        self.val_acc = []

    def on_batch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.acc.append(logs.get('val_acc'))

class TrainHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, batch, logs={}):
        print logs
        self.losses.append(logs.get('acc'))

class ValHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.acc = []
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('val_acc'))


# In[32]:

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD, Adam
from keras.regularizers import l2, activity_l2
from keras.preprocessing.image import ImageDataGenerator

model = Sequential()
model.add(Flatten(input_shape=(1, 64, 64)))
model.add(Dense(5, init='he_normal', W_regularizer=l2(0.5)))
model.add(Activation('relu'))
model.add(Dense(2, init='he_normal', W_regularizer=l2(0.5)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')
loss = LossHistory()

history = model.fit(X_train_CC, Y_train_CC, validation_split=0.2, show_accuracy=True,
                    shuffle=True, nb_epoch=20, batch_size=1, callbacks=[loss])


# In[33]:

plt.plot(history.history['val_acc'])
plt.plot(history.history['acc'])
plt.show()

plt.plot(loss.val_acc)
plt.plot(loss.acc)


# In[12]:

y_pred = model.predict(X_v)


# In[ ]:

y_pred = np.argmax(y_pred, axis=1)


# In[ ]:

accuracy = 0.0
for i in range(y_pred.shape[0]):
    accuracy += (1 == y_v[i][y_pred[i]])

accuracy /= y_pred.shape[0]


# In[ ]:

print accuracy


# In[ ]:

import pydot


# In[ ]:



