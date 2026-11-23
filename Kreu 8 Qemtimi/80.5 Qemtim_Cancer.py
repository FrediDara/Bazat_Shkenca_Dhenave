
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

trans = df[["mean radius_cat","mean texture_cat","target"]].astype(str)
encoded = pd.get_dummies(trans)
print(encoded.head())

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

GrupArtik_shp = apriori(encoded, min_support=0.1, use_colnames=True)
print(GrupArtik_shp.head())
rules = association_rules(GrupArtik_shp, metric="confidence", min_threshold=0.6)
print(rules[["antecedents","consequents","support","confidence","lift"]])
