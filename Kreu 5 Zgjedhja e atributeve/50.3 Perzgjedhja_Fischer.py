import pandas as pd
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = load_iris()     # Ngarkimi i te dhenave
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

zgjedhesi = SelectKBest(score_func=f_classif, k=2)  
X_zgjedhur = zgjedhesi.fit_transform(X, y)

atribute_zgjedhur = X.columns[zgjedhesi.get_support()]
f_scores = zgjedhesi.scores_[zgjedhesi.get_support()]
print(f"Atributet e zgjedhura: {atribute_zgjedhur}")
print(f"Vlera e F-se: {f_scores}")
