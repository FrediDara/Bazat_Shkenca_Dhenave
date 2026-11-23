import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif

data = load_wine()   # Ngarkimi i te dhënave
df = pd.DataFrame(data.data,columns = data.feature_names)

# Ndarja ne ndryshore te pavarura dhe e varur 
X = df   
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)      # Ndarja

# Gjetja e informacionit te ndersjelle
Inf_ndersjelle = mutual_info_classif(X_train, y_train )

Inf_ndersjelle = pd.Series(Inf_ndersjelle)     # Ristrukturimi
Inf_ndersjelle.index = X_train.columns         # i vlerave te 
Inf_ndersjelle.sort_values(ascending = False)  # Inf ndersjelle

from sklearn.feature_selection import SelectKBest

Kol_zgj = SelectKBest(mutual_info_classif, k = 10)
Kol_zgj.fit(X_train, y_train)
X_train.columns[Kol_zgj.get_support()]

from sklearn.feature_selection import SelectPercentile

Kol_zgj = SelectPercentile(mutual_info_classif, percentile = 10)
Kol_zgj.fit(X_train, y_train)
X_train.columns[Kol_zgj.get_support()]
