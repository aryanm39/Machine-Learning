#1,11
'''Write the python program to view basic statistical details of the data(Use winequality-red.csv).'''
#3
'''Write a Python program to view basic statistical details of the data(Use Heights and Weights dataset)'''
#14
'''Write a python program to view basic statistical details of the data
(Use advertising.csv)'''


import pandas as pd
df=pd.read_csv('HeightWeight.csv')
print('Statistical details:- \n')
print(df.describe())
