from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.datasets import load_iris 

irisData = load_iris()   # Ngarkimi i te dhenave 
X = irisData.data        # Krijimi i vektorit te atributeve
y = irisData.target      # Krijimi i vektorit te qellimit

# Ndarja ne bashkesine e stervitjes dhe testimit 
X_strv, X_test, y_strv, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y) 

knn = KNeighborsClassifier(n_neighbors=7) 
knn.fit(X_strv, y_strv) 

# Parashikimi me te dhenat qe modeli nuk i ka ndeshur me pare
y_pred = knn.predict(X_test)

print(knn.score(X_test, y_test)) # Llogaritja e saktesise

