#17
'''Write a python program to create a data frame containing columns name,age,salary,department.Add 10 rows to the data frame.
View the data frame.'''

import pandas as pd
df = pd.DataFrame(columns=['name','age','salary'])

df.loc[0] = ['Aishwarya',20,99.00]
df.loc[1] = ['Saloni',17,80.00]
df.loc[2] = ['Sayali',18,90.00]
df.loc[3] = ['Sita',21,95.00]
df.loc[4] = ['Sanika',15,80.00]
df.loc[5] = ['Gautami',16,85.00]
df.loc[6] = ['Komal',17,77.00]
df.loc[7] = ['Rita',20,78.00]
df.loc[8] = ['Rutuja',20,91.00]
df.loc[9] = ['Riya',20,75.00]

print(df)
