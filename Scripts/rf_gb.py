"""
@author: Ayobami Ogunmolasuyi
Thayer School of Engineering at Dartmouth College
ayobami.o.ogunmolasuyi.th@dartmouth.edu

FirnLearn: A Neural Network based approach to Firn Densification Modeling for Antarctica

Create the random forest and gradient boosting files
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import math
from sklearn import ensemble
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

train_df= pd.read_csv("df_train.csv")
test_df = pd.read_csv('df_test.csv')
df_validation= pd.read_csv("df_validation.csv")

#training parameters
X_train = train_df[['Accumulation rate','Temperature','Depth']].values
Y_train = train_df['density'].values

#test parameters
X_test = test_df[['Accumulation rate','Temperature','Depth']].values
Y_test = test_df['density'].values

#validation parameters
X_validation = df_validation[['Accumulation rate','Temperature','Depth']].values
Y_validation = df_validation[['density']].values

#scale X variables
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_validation_scaled = scaler.transform(X_validation)
X_test_scaled = scaler.transform(X_test)

#scale Y variables
output_scaler = MinMaxScaler(feature_range=(0, 1))
Y_train_scaled = output_scaler.fit_transform(Y_train.reshape(-1, 1)).ravel()
Y_test_scaled = output_scaler.transform(Y_test.reshape(-1, 1)).ravel()
Y_validation_scaled = output_scaler.transform(Y_validation.reshape(-1, 1)).ravel()

rf_params = {
        "n_estimators": [50, 100,200, 300,500],
        "max_depth": [2,4,6, None],
        "min_samples_split": [1,2,5,9,10],
        #"criterion": ("absolute_error","squared_error"),
}
rf_ensemble = ensemble.RandomForestRegressor(**rf_params)

model_rf = GridSearchCV(rf_ensemble, param_grid = rf_params, verbose=20)

#fit the random forest model to the training data
model_rf.fit(X_train_scaled, Y_train_scaled)

model_rf.best_params_

gb_params = {
            "n_estimators": [25,50,100], 
            "max_depth": [2,6,10], 
            "min_samples_split": [2,5,10],
            "learning_rate": [0.1,0.5],
            "loss": ('huber', 'squared_error'), 
            "validation_fraction": [0.1,0.2]
    }
gb_ensemble = ensemble.GradientBoostingRegressor(**gb_params)

#fit the gradient boosting model to the training data
model_gb = GridSearchCV(gb_ensemble, param_grid = gb_params, verbose=20)
model_gb.fit(X_train_scaled, Y_train_scaled)

#check best parameters
model_gb.best_params_
