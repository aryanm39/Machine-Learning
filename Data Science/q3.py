#2,6
'''Write a python program for handling missing value.Replace missing value of salary.age column with mean of that column(Use Data.csv file)'''

import numpy as np
import pandas as pd

df =  pd.read_csv("Data.csv")
agemean = df['Age'].mean()
salarymean = df['Salary'].mean()
df['Age'].fillna(agemean, inplace=True)
df['Salary'].fillna(salarymean, inplace=True)
print("\n\nMissing values for age and salary Replaced with MeanValue:")
print(df)


