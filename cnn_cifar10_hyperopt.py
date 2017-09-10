# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 15:36:57 2017

@author: anand
"""

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D
from keras import regularizers
from keras.optimizers import Adam, Nadam, SGD
from hyperopt import Trials, STATUS_OK, tpe
from hyperas.distributions import uniform, choice, loguniform
from hyperas import optim
from keras.utils import np_utils
from keras import backend as K
from keras.wrappers.scikit_learn import KerasClassifier
#from keras.utils.visualize_util import plot
K.set_image_dim_ordering('th')
#import matplotlib.pyplot as plt
import numpy as np
import cPickle as pickle

# random seed for reproducibility
seed = 7
np.random.seed(seed)

def data():
    """
    Data providing function:

    This function is separated from model() so that hyperopt
    won't reload data for each run.
    """
    #load data
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

    return X_train, y_train, X_test, y_test, num_classes

# create model
def model(X_train, y_train, X_test, y_test,num_classes):
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=X_train.shape[1:]))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0.2, 0.8)}}))

    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0.2, 0.8)}}))

    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0.2, 0.8)}}))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0.2, 0.8)}}))
    model.add(Dense(num_classes, activation='softmax'))

    # compile model
    adam = Adam(lr={{loguniform(-10,-4)}})
    model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size={{choice([50, 100, 200, 300, 400])}}, verbose=2)

    scores, acc = model.evaluate(X_test, y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

#running hyperopt
if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=model, data=data, algo=tpe.suggest, max_evals=5, trials=Trials())
    X_train, y_train, X_test, y_test, num_classes = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)

# serialize best_model to YAML
#model_yaml = best_model.to_yaml()
#with open("cnn_cifar10_hyperopt.yaml", "w") as yaml_file:
#    yaml_file.write(model_yaml)

# serialize best model weights to HDF5
#best_model.save_weights("cnn_cifar10_hyperopt.h5")
