#1. Import the libraries
import csv
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import pandas as pd

#2.  Read the data, encode the data
#create the sample dataset
dataset = []
with open('datasets/groceries.csv','r') as csvfile:
    reader = csv.reader(csvfile,delimiter=',')
    for row in reader:
        dataset+=[row]
te = TransactionEncoder()
data = te.fit_transform(dataset)
data = pd.DataFrame(data,columns=te.columns_)
print(data)
print(data.shape)

#3. Find the frequent Itemsets
freq_items=apriori(data,min_support=0.5,use_colnames=True)
print(freq_items)

#4. Generate the association rules
rules = association_rules(freq_items,metric='support',min_threshold=0.05)
#1. Import the libraries
import csv
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import pandas as pd

#2.  Read the data, encode the data
#create the sample dataset
dataset = []
with open('datasets/basket.csv','r') as csvfile:
    reader = csv.reader(csvfile,delimiter=',')
    for row in reader:
        dataset+=[row]
te = TransactionEncoder()
data = te.fit_transform(dataset)
data = pd.DataFrame(data,columns=te.columns_)
print(data)
print(data.shape)

#3. Find the frequent Itemsets
freq_items=apriori(data,min_support=0.5,use_colnames=True)
print(freq_items)

#4. Generate the association rules
rules = association_rules(freq_items,metric='support',min_threshold=0.05)
rules1 = association_rules(freq_items,metric='confidence',min_threshold=0.8)
rules2 = association_rules(freq_items,metric='lift',min_threshold=10)
rules = rules.sort_values(['support','confidence'],ascending=[False,False])
print(rules)

""" rules['ante_len'] = rules['antecedents'].apply(lambda x:len(x))
nrules = rules[(rules['ante_len'] <= 3) & 
               (rules['confidence']>= 0.9) & 
               (rules['lift'] >= 2.0)]
nrules = rules[rules['consequents'] == {'whole milk'}]
nrules = rules[rules['antecedents'] == {'cereals','curd'}]
nrules
nrules = rules[['antecedents','consequents','confidence']]
nrules.head()
nrules.to_csv('rules.csv')
 """
""" Apriori algorithm

4. Now, Convert Pandas DataFrame into a list of lists for encoding
transactions = []
for i in range(0, len(df)):
transactions.append([str(df.values[i,j]) for j in range(0, len(df.columns))])
5. Apply TransformEncoder to the transactions list
6. Apply the apriori algorithm
 """

