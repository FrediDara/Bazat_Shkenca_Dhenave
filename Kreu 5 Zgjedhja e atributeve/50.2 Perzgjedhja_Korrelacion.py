# importimi i bibliotekave
import pandas as pd                      
import numpy as np
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()   # Ngarkimi i te dhenave
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = pd.Series(data.target)

Matrica_korrel = df.corr(method='pearson')
Korr_qellim = Matrica_korrel['target']

k = 10
top_k = Korr_qellim.abs().sort_values(ascending=False)[:k].index
Atrib_zgjedhura = df[top_k]

Matrica_korrel_zgj = Atrib_zgjedhura.corr()
print(Matrica_korrel_zgj)

import pandas as pd

data = {     # Krijimi i nje dataframe te thjeshte
    'Rasti_A': [1, 2, 3, 4, 5],
    'Rasti_B': [1, 3, 2, 5, 4],
    'Rasti_C': [5, 4, 3, 2, 1]}
df = pd.DataFrame(data)

# Llogaritja e matrices se korrelacionit Kendall
korr_matrix = df.corr(method='kendall')
print(korr_matrix)
