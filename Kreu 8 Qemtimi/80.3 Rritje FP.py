import numpy as np 
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

df = pd.read_csv("groceries_dataset.csv")


# Grupimi Member + Date per te krijuat shportat
baskets = df.groupby(['Member_number', 'Date']) ['itemDescription'].apply(list).tolist()

te = TransactionEncoder()   # Kodimi nje flake 
te_ary = te.fit(baskets).transform(baskets)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = fpgrowth(df_encoded, min_support=0.001, use_colnames=True)

final_table = pd.DataFrame()  # Rregullat e shoqerimit

if not frequent_itemsets.empty:
    rules = association_rules(frequent_itemsets, metric= "confidence", min_threshold=0.1)
    
    # kthimi i frozenset ne string per tu lexuar me mire
    rules["antecedents"] = rules["antecedents"].apply(lambda x: ', '.join(list(x)))
    rules["consequents"] = rules["consequents"].apply(lambda x: ', '.join(list(x)))

    # Rirenditja e shtyllave
    final_table = rules[["antecedents", "consequents",
        "antecedent support", "consequent support",
        "support", "confidence", "lift" ]]

    # Filtrimi i rregullave te forta per te ulur num e rresht 
    final_table = final_table[(final_table['lift'] >= 1.1) & (final_table['confidence'] >= 0.1)]

# Shfaqja e DataFrame 
print(final_table)

