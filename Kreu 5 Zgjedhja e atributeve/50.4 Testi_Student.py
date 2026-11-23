import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from scipy import stats

# Ngarkimi i te dhenave te veres 
wine = load_wine()           
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Series(wine.target)

binary_mask = y.isin([0, 1]) # Konvertimi ne problem me 2-klasa 
X_binary = X[binary_mask]
y_binary = y[binary_mask]

# Ndarja e te dhenave ne dy grupe klasash
group_0 = X_binary[y_binary == 0]
group_1 = X_binary[y_binary == 1]

# Llogaritja e t-test per cdo atribut
t_stats = []
p_values = []

for feature in X_binary.columns:
    # Kryerja e t-testit te pavarur 
    t_stat, p_val = stats.ttest_ind(group_0[feature], group_1[feature], equal_var=False)
    t_stats.append(abs(t_stat)) 
    # Perdoren vlerat absolute per rendesine e rankut 
    p_values.append(p_val)

# Krijimi i dataframe permbledhes te rankimit
feature_scores = pd.DataFrame({'Feature': X_binary.columns,
    'Abs_T_Score': t_stats, 'P_Value': p_values}).sort_values( 
      by='Abs_T_Score', ascending=False).reset_index(drop=True)

print("--- Renditja e Rëndësisë së atributeve me T-Test ---")
print(feature_scores)

top_k = 5   # Filtrimi i k atributeve kryesore (p.sh., 5)
atrib_zgj = feature_scores['Feature'].head(top_k).tolist()
X_filtered = X_binary[atrib_zgj]

print(f"\n{top_k} atributet kryesore te zgjedhura: {atrib_zgj}")

