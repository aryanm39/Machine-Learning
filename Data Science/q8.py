#7
'''Write the python program to perform the following tasks:
a. Apply Onehot coding on country column
b. Apply Label encoding on purchased column
(Data.csv have two categorical column the country column,and the purchased column)'''

#27,29
'''Create a dataset data.csv having two categorial column (the country column,and the purchased column.)
a. Apply Onehot coding on country column
b. Apply Label encoding on purchased column'''

from sklearn.preprocessing import OneHotEncoder,LabelEncoder
import pandas as pd
df = pd.read_csv('Data.csv')

hotencoder = OneHotEncoder()
enc_df = pd.DataFrame(hotencoder.fit_transform(df[['Country']]).toarray())
df = df.join(enc_df)
print('\n\nOnehot encoding on country column')
print('-----------------------')
print(df)
labelencoder=LabelEncoder()
df['Purchased']=labelencoder.fit_transform(df['Purchased'])
print('\n\nLabel encoding on purchased column')
print(df)