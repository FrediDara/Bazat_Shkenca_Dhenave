import numpy as np 
import pandas as pd 
from mlxtend.frequent_patterns import apriori, association_rules 

df = pd.read_csv("groceries_dataset.csv") 

basket = df.groupby(['Member_number', 'Date'])['itemDescription'] .apply(list).reset_index() 
transactions = basket['itemDescription'].tolist() 
print(transactions) 

from mlxtend.preprocessing import TransactionEncoder 
te = TransactionEncoder() 
te_array = te.fit(transactions).transform(transactions) 
df_encoded = pd.DataFrame(te_array, columns=te.columns_) 

frequent_itemsets = apriori(df_encoded, min_support=0.01, use_colnames=True) 
print("Gjithsej grupe artikujsh të shpeshtë:", frequent_itemsets.shape[0]) 

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)
rules = rules[rules['antecedents'].apply(lambda x: len(x) >= 1) & rules['consequents'].apply(lambda x: len(x) >= 1)]
print("Association Rules:", rules.shape[0])
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(5))

import matplotlib.pyplot as plt

plt.figure(figsize = (10,7))
top_items = df['itemDescription'].value_counts().head(10)
top_items.plot(kind='bar', title='10 artikujt me te shitur')
plt.xlabel("Artikulli")
plt.ylabel("Numri")

plt.show()
