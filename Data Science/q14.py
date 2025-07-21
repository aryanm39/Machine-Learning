#16
'''Write a python program to create a data frame for students information such as name,graduation percentage and age.
Display average age of students,average of graduation percentage'''

import numpy as np
import pandas as pd
data = {'name':['raqhul','rohan','kiran','hitesh'],'grad_percentage':[30,40,50,60],'age':[24,27,21,22]}
df = pd.DataFrame(data)
print(df)
mean_percentage = np.mean(df['grad_percentage'])
print("mean of graduation percentage is",mean_percentage)
mean_age = np.mean(df['age'])
print("mean of age is",mean_age)
print(df.describe())