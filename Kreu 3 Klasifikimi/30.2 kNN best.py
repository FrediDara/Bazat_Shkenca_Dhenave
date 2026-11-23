from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.datasets import load_iris 

irisData = load_iris()   # Ngarkimi i te dhenave 
X = irisData.data        # Krijimi i vektorit te atributeve
y = irisData.target      # Krijimi i vektorit te qellimit

import numpy as np
import matplotlib.pyplot as plt

# Ndarja e te dhenave per stervitje dhe testim
X_strv, X_test, y_strv, y_test = train_test_split( 
             X, y, test_size = 0.4, random_state=42, stratify=y) 

fqinjet = np.arange(1, 9) 
saktesia_strv = np.empty(len(fqinjet)) 
saktesia_test = np.empty(len(fqinjet)) 
  
for i, k in enumerate(fqinjet): # Cikel per vlerat K 
    knn = KNeighborsClassifier(n_neighbors=k) 
    knn.fit(X_strv, y_strv) 
    # Llogaritja e saktesise per stervitjen dhe testimin 
    saktesia_strv[i] = knn.score(X_strv, y_strv) 
    saktesia_test[i] = knn.score(X_test, y_test) 

# Gjenerimi i grafikut 
plt.plot(fqinjet, saktesia_test, label = 'Saktesia e te dhenave te Testimit') 
plt.plot(fqinjet, saktesia_strv, linestyle='dashed', label = 'Saktesia e te dhenave te stervitjes') 
plt.legend() 
plt.xlabel('Nr fqinjeve',fontsize=12) 
plt.ylabel('Saktesia',fontsize=12)
plt.show() 

