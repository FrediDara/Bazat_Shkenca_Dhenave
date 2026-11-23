import pandas as pd

#ngarkimi i te dhënave
data = pd.read_csv('OnlineRetail.csv', encoding='windows-1252')  
data.head()                    # Shfaqja e 5 rreshtave te pare

data.dropna(inplace = True)
data['InvoiceNo'] = data['InvoiceNo'].astype('str')
data = data[~data['InvoiceNo'].str.contains('C')]


basket = (data[data['Country'] == "France"]
          .groupby(['InvoiceNo', 'Description'])['Quantity'] 
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))


def encode_units(x):
    return 1 if x >= 1 else 0

basket_sets = basket.applymap(encode_units)


from mlxtend.frequent_patterns import apriori
frequent_itemsets = apriori(basket_sets, min_support=0.05, use_colnames=True)
frequent_itemsets.head()

from mlxtend.frequent_patterns import association_rules
rules = association_rules(frequent_itemsets, metric ="lift", min_threshold = 1)
rules.head()

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize = (10,6))
sns.scatterplot(x = "support", y = 'confidence', data=rules, hue = "lift", palette ='viridis')
plt.title('Mbeshtetja kundrejt Besimit')
plt.xlabel('Mbeshtetja')
plt.ylabel('Besimi')
plt.show()
