#Slip-22
'''Dataset name: winequality-red.csv
a. Rescaling: Normalised the dataset using MinMaxScaler tasks
b. Standardizing Data (transfrom them into a standard Gaussian distribution with a mean of 0 and a standard deviation of 1)
c. Normalizing data (rescale each observation to a length of 1(a unit form). For this,use the Normalizer class.) '''

#Slip-23
'''Write a program in python to perform following task
a. Rescaling: Normalised the dataset using MinMaxScaler class 
b. Standardizing Data (transform them into a standard Gaussian distribution with a mean of 0 and a standard deviation of 1)
c. Binarizing Data using we use the Binarizer class (Using  a binary threshold,it is possible to transform our data by marking the values 
above it 1 and those equal to or below it,0)'''

#8
'''Write a program in python to perform following task:
Standardizing data (transform them into a standard Gaussian distribution with a mean of 0
and a standard deviation of 1)
(Use winequality-red.csv)'''

import scipy.stats as s
import pandas as pd
from sklearn import preprocessing
df = pd.read_csv("winequality-red.csv")
print("\n\nData Scaled Between 0 and 1")
data_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
data_scaled = data_scaler.fit_transform(df)
print("\n Min Max Scaled Data")
print("------------------------")
print(data_scaled.round(2))


print(" Original Data \n")
print(df)
print("\n Initial Mean : ",s.tmean(df).round(2))
print("\n\nInitial Standard deviation :")
print("-----------------------------------")
print(round(df.std(),2))
df_scaled = preprocessing.scale(df)
df_scaled.mean(axis=0)
df_scaled.std(axis=0)
print("\n Standardized Data \n",df_scaled.round(2))
print("\n Scaled Mean :\n",s.tmean(df_scaled).round(2))
print("\n Scaled Standard Deviation \n",round(df_scaled.std(),2))


dn = preprocessing.Normalizer(df)
print("\n L1 Normalized Data ")
print("----------------------")
print(dn)

data_binarized = preprocessing.Binarizer(threshold=5).transform(df)
print("\nBinarized data ")
print("------------------")
print(data_binarized)