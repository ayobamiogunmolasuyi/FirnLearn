import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras
from sklearn.preprocessing import StandardScaler


#Read in data
test_df = pd.read_csv('df_test.csv',index_col=['Profile'])
train_df = pd.read_csv('df_train.csv',index_col=['Profile'])
validation_df = pd.read_csv('df_validation.csv',index_col=['Profile'])

X_train = train_df[['Accumulation rate','Temperature','Depth']].values
Y_train = train_df['density'].values

X_test = test_df[['Accumulation rate','Temperature','Depth']].values
Y_test = test_df['density'].values

X_validation = validation_df[['Accumulation rate','Temperature','Depth']].values
Y_validation = validation_df[['density']].values

model2 = keras.models.load_model("model2.h5")


scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_validation_scaled = scaler.transform(X_validation)
X_test_scaled = scaler.transform(X_test)

def predict1(X, model1):
    Y_pred = model1.predict(X)
    return Y_pred

def firnlearn(AHL, THL, model2, z_depth, z_range):
    z = np.linspace(0, z_depth, z_range)
    AHL_values = np.full(z_range, AHL)
    THL_values = np.full(z_range, THL)
    X_check = np.column_stack([AHL_values, THL_values, z])
    X_check_scaled = scaler.transform(X_check) 
    Y_nn = model2.predict(X_check_scaled).tolist() 
    return Y_nn


#Test on the six sites described in the paper

##Predict with FirnLearn
C2_NN= firnlearn( 0.067149,225.88,model2, 100, 1000)
C24_NN = firnlearn(0.344318,244.97,model2, 70, 1000)
C10_NN = firnlearn(0.04861,229.712425,model2, 15,1000)
C26_NN = firnlearn(0.070506, 233.3,model2,98,1000)
C659_NN = firnlearn(0.420337,258.30, model2,90,1000)
C1937_NN = firnlearn(0.072584,225.415738,model2,120,1000)
C828_NN =  firnlearn(0.111068,228.551496,model2,70,1000)


C2_NN3_= [i[0]*1000 for i in C2_NN3]
C24_NN3_ = [i[0]*1000 for i in C24_NN3]
C10_NN3_ = [i[0]*1000 for i in C10_NN3]
C26_NN3_ = [i[0]*1000 for i in C26_NN3]
C659_NN3_ = [i[0]*1000 for i in C659_NN3]
C1937_NN3_ =[i[0]*1000 for i in C1937_NN3]
C828_NN3_ =[i[0]*1000 for i in C828_NN3]

##Predict with Herron and Langway model (HL80)
C2_HL= HL80(0.067149,225.88,0.4,100, 1000)
C24_HL = HL80(0.344318,244.97,0.4,70, 1000)
C10_HL = HL80(0.04861,229.712425, 0.34,15, 1000)
C26_HL = HL80(0.070506,233.30, 0.42,75, 1000)
C659_HL= HL80(0.420337,258.30,0.85,90, 1000)
C1937_HL = HL80(0.072584,225.415738,0.34,120, 1000)
C828_HL = HL80(0.052532,228.551496,0.3,70,1000)

C2_HL_= [i*1000 for i in C2_HL]
C24_HL_ = [i*1000 for i in C24_HL]
C10_HL_ = [i*1000 for i in C10_HL]
C26_HL_ = [i*1000 for i in C26_HL]
C659_HL_ = [i*1000 for i in C659_HL]
C1937_HL_ =[i*1000 for i in C1937_HL]
C828_HL_ =[i*1000 for i in C828_HL]

##Predict with HL80 + Ligtenberg
C2_HL2= HL80(0.067149,225.88,0.32,100, 1000)
C24_HL2 = HL80(0.344318,244.97,0.36,70, 1000)
C10_HL2= HL80(0.04861,229.712425, 0.32,15, 1000)
C26_HL2 = HL80(0.070506,233.30, 0.34,75, 1000)
C659_HL2= HL80(0.420337,258.30,0.44,90, 1000)
C1937_HL2 = HL80(0.072584,225.415738,0.3,120, 1000)
C828_HL2 = HL80(0.052532,228.551496,0.32,70,1000)

C2_HL2_= [i*1000 for i in C2_HL2]
C24_HL2_ = [i*1000 for i in C24_HL2]
C10_HL2_ = [i*1000 for i in C10_HL2]
C26_HL2_ = [i*1000 for i in C26_HL2]
C659_HL2_ = [i*1000 for i in C659_HL2]
C1937_HL2_ =[i*1000 for i in C1937_HL2]
C828_HL2_ =[i*1000 for i in C828_HL2]

##Predict with Gradient Boosting
C2_GB= firnlearn( 0.067149,225.88,gb_model, 100, 1000)
C24_GB = firnlearn(0.344318,244.97,gb_model, 70, 1000)
C10_GB = firnlearn(0.04861,229.712425,gb_model, 15,1000)
C26_GB = firnlearn(0.070506, 233.3,gb_model,98,1000)
C659_GB = firnlearn(0.420337,258.30, gb_model,90,1000)
C1937_GB = firnlearn(0.072584,225.415738,gb_model,120,1000)
C828_GB = firnlearn(0.111068,228.551496,gb_model,70,1000)

C2_GB_= [i*1000 for i in C2_GB]
C24_GB_ = [i*1000 for i in C2_GB]
C10_GB_ = [i*1000 for i in C2_GB]
C26_GB_ = [i*1000 for i in C2_GB]
C659_GB_ = [i*1000 for i in C2_GB]
C1937_GB_ =[i*1000 for i in C2_GB]
C828_GB_ =[i*1000 for i in C2_GB]

##Predict with Random Forest
C2_RF= firnlearn( 0.067149,225.88,rf_model, 100, 1000)
C24_RF = firnlearn(0.344318,244.97,rf_model, 70, 1000)
C10_RF = firnlearn(0.04861,229.712425,rf_model, 15,1000)
C26_RF = firnlearn(0.070506, 233.3,rf_model,98,1000)
C659_RF = firnlearn(0.420337,258.30, rf_model,90,1000)
C1937_RF = firnlearn(0.072584,225.415738,rf_model,120,1000)
C828_RF = firnlearn(0.111068,228.551496,rf_model,70,1000)

C2_RF_= [i*1000 for i in C2_RF]
C24_RF_ = [i*1000 for i in C2_RF]
C10_RF_ = [i*1000 for i in C2_RF]
C26_RF_ = [i*1000 for i in C2_RF]
C659_RF_ = [i*1000 for i in C2_RF]
C1937_RF_ =[i*1000 for i in C2_RF]
C828_RF_ =[i*1000 for i in C2_RF]

#More efficiently, this code can be written as:#
##
##
##

def apply_firnlearn(params, model):
    return [firnlearn(*param, model) for param in params]

def apply_HL80(params, densities):
    return [HL80(param[0], param[1], density, param[3], param[4]) for param, density in zip(params, densities)]

def multiply_by_1000(results):
    return [[i[0] * 1000 for i in result] for result in results]

# Define common parameters
params = [
    (0.067149, 225.88, 100, 1000),
    (0.344318, 244.97, 70, 1000),
    (0.04861, 229.712425, 15, 1000),
    (0.070506, 233.3, 98, 1000),
    (0.420337, 258.30, 90, 1000),
    (0.072584, 225.415738, 120, 1000),
    (0.111068, 228.551496, 70, 1000)
]

# Apply FirnLearn with different models
model_results = {}
for model_name, model in [('NN', model2), ('GB', gb_model), ('RF', rf_model)]:
    model_results[model_name] = multiply_by_1000(apply_firnlearn(params, model))

# Apply HL80
densities_HL = [0.4, 0.4, 0.34, 0.42, 0.85, 0.34, 0.3]
HL_results = multiply_by_1000(apply_HL80(params, densities_HL))

# Apply HL80 + Ligtenberg
densities_HL2 = [0.32, 0.36, 0.32, 0.34, 0.44, 0.3, 0.32]
HL2_results = multiply_by_1000(apply_HL80(params, densities_HL2))

# Now, all results are stored in model_results, HL_results, and HL2_results


