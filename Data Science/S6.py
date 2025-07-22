#1. Import the libraries
import pandas as pd 
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder

#2.  Read the data, encode the data
#create the sample dataset
transactions = [['Bread','Milk'],
                ['Bread','Diaper','Beer','Eggs'],
                ['Milk','Diaper','Beer','Coke'],
                ['Bread','Milk','Diaper','Beer'],
                ['Bread','Milk','Diaper','Coke']]

#transform it into the right format via Transactioon Encoder as follows:
te=TransactionEncoder()
te_arrary=te.fit(transactions).transform(transactions)
df=pd.DataFrame(te_arrary,columns=te.columns_)
print(df)

#3. Find the frequent Itemsets
freq_items=apriori(df,min_support=0.5,use_colnames=True)
print(freq_items)

#4. Generate the association rules
rules = association_rules(freq_items,metric='support',min_threshold=0.05)
rules = rules.sort_values(['support','confidence'],ascending=[False,False])
print(rules)