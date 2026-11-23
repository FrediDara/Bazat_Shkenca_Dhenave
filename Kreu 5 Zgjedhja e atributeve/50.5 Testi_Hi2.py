from sklearn.datasets import load_iris

# Ngarkimi i te dhenave
iris = load_iris()            
X, y = iris.data, iris.target

print(X.shape, y.shape)       # (150, 4) (150,)

from sklearn.feature_selection import SelectKBest, chi2

# Zgjedhja e atributeve 
selector = SelectKBest(chi2, k=2) 

X_new = selector.fit_transform(X, y)

# Printimi i indeksit te atributeve te zgjedhura and vleresimit
selected_features = selector.get_support(indices=True)
scores = selector.scores_

print("Atributet e zgjedhura: ", selected_features) 
print("Vleresimi: ", scores) 
