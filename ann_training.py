#!/usr/bin/env python3

"""
@author: Ayobami Ogunmolasuyi
Thayer School of Engineering at Dartmouth College
ayobami.o.ogunmolasuyi.th@dartmouth.edu

FirnLearn: A Neural Network based approach to Firn Densification Modeling for Antarctica

Creating the training, validation, and test datasets

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from keras.models import Sequential
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from keras import backend as K
from keras.layers import Dropout
from keras.layers import LeakyReLU
from keras.callbacks import ModelCheckpoint


from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
#from keras.layers import Activation
from keras.layers import GaussianNoise
from tensorflow.keras import optimizers

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_validation_scaled = scaler.transform(X_validation)
X_test_scaled = scaler.transform(X_test)

output_scaler = MinMaxScaler(feature_range=(0, 1))
Y_train_scaled = output_scaler.fit_transform(Y_train.reshape(-1, 1)).ravel()
Y_test_scaled = output_scaler.transform(Y_test.reshape(-1, 1)).ravel()
Y_validation_scaled = output_scaler.transform(Y_validation.reshape(-1, 1)).ravel()


def train_model(activation1,learning_rate,n_epochs, X_train, Y_train):
    number_of_features = 3
    # Start neural network
    model = Sequential()

    # Add fully connected layer with a ReLU activation function
    model.add(layers.Dense(3, activation=activation1, input_shape=(number_of_features,)))

    
    #for k in range (n_layers):
    model.add(layers.Dense(50, activation=activation1))
    
    model.add(layers.Dense(40, activation=activation1))
    
    model.add(layers.Dense(20, activation=activation1))
    
    model.add(layers.Dense(10, activation=activation1))
    
    model.add(layers.Dense(5, activation=activation1))
    
    
    # Add fully connected layer with a sigmoid activation function
    model.add(layers.Dense(1, activation='sigmoid'))
    
    # Compile neural network
    Adam = keras.optimizers.Adam(learning_rate = learning_rate)
    
    model.compile(loss='mean_squared_error',
                    optimizer=Adam) # Optimizer
    print(model.summary())
    
    # Define the ModelCheckpoint callback to save weights
    checkpoint = ModelCheckpoint("best_model_weights.h5", 
                                monitor='val_loss',   # The quantity to monitor for saving weights (e.g., validation loss)
                                verbose=1,            # Verbosity (1: show messages)
                                save_best_only=True,  # Save only the best model
                                mode='min'            # Save mode ('min' for loss, 'max' for accuracy, etc.)
                            )
    
    history = model.fit(X_train_scaled, Y_train_scaled, validation_data =(X_validation_scaled,Y_validation_scaled), epochs=n_epochs, batch_size = 32,callbacks=[checkpoint])
    
    # Return compiled network
    return history,model


history2, model2 = train_model('LeakyReLU', 0.0001, 200,X_train_scaled, Y_train_scaled)

model2.load_weights("best_model_weights.h5")

model2.save("model2.h5")




