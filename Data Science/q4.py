#12
'''Write a Python program to create data frame containing column name,salary,department
add 10 rows with some missing and duplicate values to the data frame.
Also drop all null and empty values.print the modified data frame.'''

import numpy as np
import pandas as pd
df = pd.DataFrame(columns=['name','salary','department'])
df.loc[0] = ['a',100000,'IT']
df.loc[1] = ['b',None,'IT']
df.loc[2] = ['c',100000,'Computer']
df.loc[3] = ['d',45400,None]
print("missing value\n",df.isnull())
print('duplicated value\n',df.duplicated())
print('Modified Dataframe\n')
df.dropna(inplace=True)
df.replace(' ',np.nan,inplace=True)
print(df)