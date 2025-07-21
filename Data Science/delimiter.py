import pandas as pd
df=pd.read_csv('winequality')
#import pandas as pd
delimiter=';'
df=pd.read_csv('winequality-red.csv',delimiter=delimiter)
print(df)
