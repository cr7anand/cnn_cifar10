# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 22:55:40 2017

@author: anand
"""
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.optimizers import Adam, SGD
from keras import regularizers
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
#from keras.utils.visualize_util import model_to_dot
K.set_image_dim_ordering('th')
import matplotlib.pyplot as plt
import numpy as np
import cPickle as pickle


# random seed for reproducibility 
seed = 7
np.random.seed(seed)

# load mnist dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# reshape to be [samples][pixels][width][height]
X_train = X_train.astype('float32')
X_test = X_test.astype('float32') 


# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# create model
def baseline_model():
    model = Sequential()
    model.add(Convolution2D(64, 3, 3, border_mode='same', kernel_regularizer=regularizers.l2(l=0.0001), input_shape=X_train.shape[1:]))
    model.add(BatchNormalization())    
    model.add(Activation('relu'))    
    model.add(Convolution2D(64, 3, 3, border_mode='same', kernel_regularizer=regularizers.l2(l=0.0001)))
    model.add(BatchNormalization())    
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    
    model.add(Convolution2D(128, 3, 3, border_mode='same', kernel_regularizer=regularizers.l2(l=0.0001), input_shape=X_train.shape[1:]))
    model.add(BatchNormalization())    
    model.add(Activation('relu'))    
    model.add(Convolution2D(128, 3, 3, border_mode='same', kernel_regularizer=regularizers.l2(l=0.0001)))
    model.add(BatchNormalization())    
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Convolution2D(256, 3, 3, border_mode='same', kernel_regularizer=regularizers.l2(l=0.0001)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    
    model.add(Flatten())
    model.add(Dense(512,kernel_regularizer=regularizers.l2(l=0.0001)))
    model.add(BatchNormalization())    
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(256, kernel_regularizer=regularizers.l2(l=0.0001)))
    model.add(BatchNormalization())    
    model.add(Activation('relu'))   
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    # compile model
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
    return model

# build the model
model = baseline_model()
# define grid search parameters
#batch_size = [10,20,50,100,200,500]
#learning_rate = [0.00001,0.0001,0.001,0.01]
#dropout_rate = [0.1,0.2,0.3,0.4,0.5]
#param_grid = dict(batch_size=batch_size, lr=learning_rate,d_rate=dropout_rate)
#grid = GridSearchCV(estimater=model, param_grid=param_grid,n_jobs=-1)
#grid_result = grid.fit(X_train,y_train)

# define early-stopping criteria
callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=0)]

# fit the model
history = model.fit(X_train,y_train,validation_data=(X_test,y_test),nb_epoch=30,batch_size=150,shuffle=True)

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for acc
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# final evaluation of the model
#scores=model.evaluate(X_test,y_test,verbose=0)
#print("Baseline Error: %.2f%%" % (100-scores[1]*100))

# save training history to file
with open('cifar10_cnn_earlystop_trainhistory.pkl', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

# serialize model to YAML
model_yaml = model.to_yaml()
with open("cifar10_cnn_earlystop.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)

# serialize weights to HDF5
model.save_weights("cifar10_cnn_earlystop.h5")
