#9
'''Write a program in python to perform following task(Use winequality-red.csv)
Import Dataset and do the followings:
a) Describing the dataset
b) Shape of the dataset
c) Display first 3 rows from dataset'''

import pandas as pd
df = pd.read_csv('winequality-red.csv')

print("dat5a description",df.describe())
print("\n\nData shape",df.shape)
print("\n\n First 3 rows",df.head(3))


