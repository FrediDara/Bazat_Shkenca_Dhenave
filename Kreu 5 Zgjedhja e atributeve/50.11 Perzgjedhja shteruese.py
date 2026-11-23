import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS

breast_cancer = load_breast_cancer()
X = pd.DataFrame(breast_cancer.data, columns= breast_cancer.feature_names)
y = breast_cancer.target
X_train, X_test, y_train, y_test = train_test_split( X, y, random_state=0)


efs = EFS(estimator=RandomForestClassifier(n_estimators=2, random_state=0), 
          min_features=1, max_features=4, scoring='roc_auc',cv=0,print_progress=True )
efs = efs.fit(X_train, y_train)  #  Stervitja e modelit

efs.best_feature_names_

efs.subsets_[60]

X_train_t = efs.transform(X_train)
X_test_t = efs.transform(X_test)



import matplotlib.pyplot as plt
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

# 1. Gjenerimi i grafikut duke përdorur sanxhakun e të dhënave nga EFS
# Pasi ke bërë efs.fit(X_train, y_train):
fig, ax = plot_sfs(efs.get_metric_dict(), kind='std_dev')

# 2. Personalizimi i pamjes së grafikut
ax.get_lines().set_color('royalblue')  # Ndrysho ngjyrën e vijës në blu
ax.get_lines().set_marker('s')         # Ndrysho pikat në formë katrore (squares)

plt.title('Përzgjedhja Shteruese e Veçorive (EFS Performance)')
plt.xlabel('Numri i atributeve (Feature Count)')
plt.ylabel('Performanca (ROC AUC Score)')
plt.grid(True)
plt.show()
