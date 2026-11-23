import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import mutual_info_classif


data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

mi = mutual_info_classif(X, y)
mi_scores = pd.Series(mi, index=X.columns)
print(mi_scores.sort_values(ascending=False))

bins = 10      # Parametrat
min_density = 5

def clique_clustering(df, bins, min_density):
   clusters = []
   for col in df.columns:
      hist, edges = np.histogram(df[col], bins=bins)
      for i in range(len(hist)):
         if hist[i] >= min_density:
            region = (edges[i], edges[i+1])
            cluster_points = df[(df[col] >= region[0]) & (df[col] < region[1])]
            clusters.append({"dimension": col, "region": region,
               "density": hist[i], "size": len(cluster_points) })
   return pd.DataFrame(clusters)

clusters = clique_clustering(df, bins=10, min_density=30)
print(clusters.head())

# Zgjedhja e atributeve me te mira duke përdorur MI
top_features = mi_scores.sort_values(ascending =False) .head(5).index
df_selected = df[top_features]
clusters = clique_clustering(df_selected, bins=10, min_density=20)
print(clusters)
