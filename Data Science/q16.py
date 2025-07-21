#19,28
'''Write a python program 
1. To create  a dataframe containing columns name,age and percentage.
Add 10 rows to the dataframe.View the dataframe.
2.To print the shape,number of rows-columns,data types,feature names and the description of the data
3.To Add 5 rows with duplicate values and missing values.Add a  column 'remarks' with empty values. Display the data  '''


import pandas as pd
df = pd.DataFrame(columns=['name','age','salary'])
df

df.loc[0] = ['Aishwarya',20,99.00]
df.loc[1] = ['Mona',17,80.00]
df.loc[2] = ['Sayali',18,90.00]
df.loc[3] = ['Sita',21,95.00]
df.loc[4] = ['Sanika',15,80.00]
df.loc[5] = ['Gautami',16,85.00]
df.loc[6] = ['Komal',17,77.00]
df.loc[7] = ['Rita',20,78.00]
df.loc[8] = ['Rutuja',20,91.00]
df.loc[9] = ['Riya',20,75.00]
df.loc[10] = ['Saloni',20,68.00]

print('Shape: ',df.shape)
print('number of rows:  ',df.shape[0])
print('number of columns:  ',df.shape[1])
print('datatypes of all columns',df.dtypes)
print('Information of data: \n',df.info())
print('Describing the data: \n',df.describe())

df.loc[11]=['s',20,83]
df.loc[12]=['s',20,83]
df.loc[13]=[None,25,80]
df.loc[14]=['v',20,77]
df.loc[15]=['w',None,91]
df['remarks']=None
print(df)
print("missing value:\n",df.isnull())
print("dublicet value:\n",df.duplicated())
