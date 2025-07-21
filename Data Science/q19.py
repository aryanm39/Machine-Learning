#2,6
'''Write a python program to generate a line plot of name vs salary'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
df=pd.read_csv('Data.csv')
df = pd.DataFrame(df,columns=['name','salary'])
df.loc[0] = ['Sanika',100000]
df.loc[1] = ['Tanaya',200000]
df.loc[2] = ['Gautami',300000]
df.loc[3] = ['Saloni',250000]
df.loc[4] = ['Komal',350000]
df.loc[5] = ['Riya',400000]
df.plot('name','salary')
plt.title('Line Plot: Name vs Salary')
plt.show()