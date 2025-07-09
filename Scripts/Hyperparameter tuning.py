#!/usr/bin/env python3

"""
@author: Ayobami Ogunmolasuyi
Thayer School of Engineering at Dartmouth College
ayobami.o.ogunmolasuyi.th@dartmouth.edu

FirnLearn: A Neural Network based approach to Firn Densification Modeling for Antarctica

Determine the optimal set of hyperparameters
"""

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

from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import GaussianNoise
from tensorflow.keras import optimizers
from keras.callbacks import ModelCheckpoint

##Import the data
train_df= pd.read_csv("df_train.csv")
test_df = pd.read_csv('df_test.csv')
df_validation= pd.read_csv("df_validation.csv")


X_train = train_df[['Accumulation rate','Temperature','depth']].values
Y_train = train_df['density'].values

X_test = test_df[['Accumulation rate','Temperature','depth']].values
Y_test = test_df['density'].values

X_validation = df_validation[['Accumulation rate','Temperature','depth']].values
Y_validation = df_validation[['density']].values

## Scale the X variables (features)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_validation_scaled = scaler.transform(X_validation)
X_test_scaled = scaler.transform(X_test)

## Scale the Y variables (density) from 0 to 1

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
output_scaler = MinMaxScaler(feature_range=(0, 1))
Y_train_scaled = output_scaler.fit_transform(Y_train.reshape(-1, 1)).ravel()
Y_test_scaled = output_scaler.transform(Y_test.reshape(-1, 1)).ravel()
Y_validation_scaled = output_scaler.transform(Y_validation.reshape(-1, 1)).ravel()

def train_model(activation1,neuron1,neuron2,neuron3,neuron4,learning_rate,n_epochs, X_train, Y_train):
    number_of_features = 3
    # Start neural network
    model = Sequential()

    # Add fully connected layer with a ReLU activation function
    model.add(layers.Dense(3, activation=activation1, input_shape=(number_of_features,)))

    
    #for k in range (n_layers):
    model.add(layers.Dense(neuron1, activation=activation1))
    
    model.add(layers.Dense(neuron2, activation=activation1))
    
    model.add(layers.Dense(neuron3, activation=activation1))
        
    model.add(layers.Dense(neuron4, activation=activation1))
    
    model.add(layers.Dense(10, activation=activation1))
    
    
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
    
    history = model.fit(X_train_scaled, Y_train_scaled, validation_data =(X_validation_scaled,Y_validation_scaled), epochs=n_epochs, batch_size = 64,callbacks=[checkpoint])
    
    # Return compiled network
    return history,model




learning_rate = [0.001,0.0001]
neuron1 = [100,50,]
neuron2 = [50,40]
neuron3 = [40,20]
neuron4 = [20,10]
#activation1 = ['relu','sigmoid','LeakyReLU']
#activation2 = ['sigmoid','linear']

params2 = []
for a in neuron1:
    for b in neuron2:
        for c in neuron3:
            for d in neuron4:
                for e in learning_rate:
                    params2.append([a,b,c,d,e])
                    
            
            
            
random_params2 = []

while (len(random_params2) < 15):
    val = params2[np.random.randint(0, len(params2))]
    if val not in random_params2:
        random_params2.append(val)
        
        
#print(params)
print(random_params2)


store_loss2 = []
store_val_loss2 = []
for params2 in random_params2:
    a,b,c,d,e = params2
    print(params2)
    rand_history2, rand_model2 = train_model('LeakyReLU', a,b,c,d,e,100, X_train_scaled, Y_train_scaled)
    loss2 = rand_history2.history['loss']
    val_loss2 = rand_history2.history['val_loss']
    store_loss2.append(loss2)
    store_val_loss2.append(val_loss2)

plt.figure(figsize = (10,10))
for i in range(len(store_val_loss2)-14):   
    plt.plot(store_val_loss2[i+1])
    plt.title('validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')                
