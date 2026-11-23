
import pandas as pd 
import numpy as np
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target     # 0 = Malignant, 1 = Benign


def discretize_atribut(df, column, bins=3):    
    etik = ["low", "medium", "high"]
    df[column+"_cat"] = pd.cut(df[column],bins=bins,labels=etik)
    return df

df = discretize_atribut(df, "mean radius")
df = discretize_atribut(df, "mean texture")

print(df[["mean radius_cat","mean texture_cat"]].head())

'''
trans = df[["mean radius_cat","mean texture_cat","target"]].astype(str)
encoded = pd.get_dummies(trans)
print(encoded.head())

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

GrupArtik_shp = apriori(encoded, min_support=0.1, use_colnames=True)
print(GrupArtik_shp.head())
rules = association_rules(GrupArtik_shp, metric="confidence", min_threshold=0.6)
print(rules[["antecedents","consequents","support","confidence","lift"]])
'''


# Logjika e DRMiner: Diskretizimi ne rrjete & Llogaritja e densitetit 
def find_dense_regions(df, atribut, bins=5, min_support=0.1):
    
    values = df[atribut]
    hist, edges = np.histogram(values, bins=bins)
    total = len(df)
    regions = []

    for i in range(len(hist)):
        support = hist[i] / total
        if support >= min_support:
            region = (edges[i], edges[i+1])
            regions.append((region, support))
    return regions

def generate_rules(df, atribut, target="target", bins=5, min_support=0.1, min_conf=0.6):
    
    regions = find_dense_regions(df, atribut, bins, min_support)
    rules = []

    for region, support in regions:
        subset = df[(df[atribut] >= region[0]) & (df[atribut] < region[1])]
        if len(subset) == 0:
            continue
        
        prob = subset[target].mean()
        confidence = max(prob, 1 - prob)
        if confidence >= min_conf:
            predicted = 1 if prob >= 0.5 else 0
            rules.append({"atribut": atribut, "interval": region,
                          "support": support, "confidence": confidence,
                          "predict": predicted })
    return rules

# Zbatimi i qëmtimit te stilit DRMiner
attributes = df.select_dtypes(include=np.number).columns.tolist()
attributes.remove("target")

all_rules = []  

for column in attributes:
    rules = generate_rules(df, column)
    all_rules.extend(rules)
    
rules_df = pd.DataFrame(all_rules)
print(rules_df.head())
atribut = "mean radius"

import matplotlib.pyplot as plt
plt.hist(df[atribut], bins=10)
plt.title("Zonat e Dendura per " + atribut)
plt.xlabel("atributi")
plt.ylabel("Denduria")
plt.show()
