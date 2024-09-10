import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras
from matplotlib import gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


train_df = pd.read_csv('df_train.csv')

X_train = train_df[['Accumulation rate','Temperature','Depth']].values
Y_train = train_df['density'].values

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)

output_scaler = MinMaxScaler(feature_range=(0, 1))
Y_train_scaled = output_scaler.fit_transform(Y_train.reshape(-1, 1)).ravel()

FirnLearn_ann = keras.models.load_model("model2.h5")

def predict1(X, model1):
    Y_pred = model1.predict(X)
    return Y_pred

def firnlearn_surface(AHL, THL, model, scaler,a):
    X_check = np.column_stack((AHL, THL, np.linspace(0,0,a)))
    X_check_scaled = scaler.transform(X_check)
    Y_nn = predict1(X_check_scaled, model)
    return Y_nn

lats = dataset1['lats'][:a]
lons = dataset1['lons'][:a]
smb = dataset1['smb'][:a]
temp = dataset1['Temperature'][:a]

Racdens_save_scaled = firnlearn_surface(smb, temp, model2, scaler,a)
Racdens_save= output_scaler.inverse_transform(Racdens_save_scaled.reshape(-1, 1)).ravel()

Sumdens_save_scaled = firnlearn_surface(sum_accum_rate, sum_temp, model2, scaler,b)
Sumdens_save= output_scaler.inverse_transform(Sumdens_save_scaled.reshape(-1, 1)).ravel()

