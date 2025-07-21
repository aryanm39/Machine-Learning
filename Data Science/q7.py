#4,5
'''Write a Python program to print the shape,number of rows-columns,data types,feature names and the description of the data
(Use User_Data.csv)'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('HeightWeight.csv')
print(df.shape)
print('\nShape of dataframe',df.shape)
print('\nNumber of rows and columns ',df.size)
print('\nData types for DataFrame\n',df.dtypes)
print('\nNumber of rows:',df.shape[0])
print('\nNumber of column:',df.shape[1])
print('\nFeature names in dataframe is: \n',df.columns)
print('\nDiscription of the data',df.describe())
