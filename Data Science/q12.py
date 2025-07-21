#2,6
'''Download the heights and weights dataset and load the dataset from a given csv file into a dataframe.
print the first,last 10 rows and random 20 rows also display shape of the dataset.'''
#10
'''Write a python program to Display column-wise mean,and median for SOCR-heightWeight dataset'''
#18
'''Use the heights and weights dataset and load the dataset from a given csv file into a dataframe.
print the first,last 5 rows and random 10 row'''

#2,6
import pandas as pd
import numpy as np
df=pd.read_csv('HeightWeight.csv')

print('Reading first 10 rows: ')
print(df.head(10))

print('Reading last 10 rows: ')
print(df.tail(10))

print('Reading random 20 random rows: ')
print(df.sample(20))

print('Shape of the Dataset',df.shape)

print('Mean: ',df.mean())
print('Median: ',df.median())

