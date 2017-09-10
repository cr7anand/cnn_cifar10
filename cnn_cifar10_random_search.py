# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 11:54:30 2017

@author: anand
"""
import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.optimizers import Adam
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
#from numpy.random import uniform
from scipy.stats import uniform
K.set_image_dim_ordering('th')
import matplotlib.pyplot as plt



# random seed for reproducibility 
seed = 7
np.random.seed(seed)

# load mnist dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# reshape to be [samples][pixels][width][height]
X_train = X_train.astype('float32')
X_test = X_test.astype('float32') 


# normalize inputs from 0-255 to 0-1
X_train = X_train/255
X_test = X_test/255

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# create model
def baseline_model(dropout_rate,learn_rate):
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=X_train.shape[1:], init='uniform',activation='relu'))
    model.add(Convolution2D(32, 3, 3,init='uniform',activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(128,init='uniform',activation='relu'))
    model.add(Dense(60,init='uniform',activation='relu'))
    model.add(Dense(num_classes,init='uniform',activation='softmax'))
    # compile model
    optimizer = Adam(lr=learn_rate)
    model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    return model

# build the model
my_model = KerasClassifier(baseline_model, batch_size=200)
# define grid search parameters
#batch_size = [10, 20, 50, 100, 200, 500]
learn_rate = uniform(10**-5.0,10**-2.0)
#decay_rate = uniform(0,1)
#epochs= [10, 20, 30, 40]
dropout_rate = uniform(0.1,0.9)
param_grid = {'dropout_rate':dropout_rate, 'learn_rate':learn_rate}
validator = RandomizedSearchCV(estimator=my_model, param_distributions=param_grid, n_iter = 10,scoring='log_loss',n_jobs=1)
validator.fit(X_train, y_train)

# summarize results
print("Best: %f using %s" % (validator.best_score_, validator.best_params_))
means = validator.cv_results_['mean_test_score']
stds = validator.cv_results_['std_test_score']
params = validator.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
