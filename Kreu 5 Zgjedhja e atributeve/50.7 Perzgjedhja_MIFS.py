import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

def MIFS(X, y, k=3, beta=0.5):
    n_atribute = X.shape[1]

    # MI midis secilës atributi dhe qellimit
    mi_qellimi = mutual_info_classif(X, y, random_state=42)

    # MI ndërmjet atributeve
    mi_atribute = np.zeros((n_atribute, n_atribute))
    for i in range(n_atribute):
        for j in range(n_atribute):
            if i != j:
                mi_atribute[i, j] = mutual_info_regression(
                     X[:, [i]], X[:, j], random_state=42)[0]
    zgjedhur = []

    for hapi in range(k):
        scores = []
        for i in range(n_atribute):
            if i in zgjedhur:
                scores.append(-np.inf)
                continue
            # Rendesia
            rendesia = mi_qellimi[i]
            # Tepria
            tepria = sum(mi_atribute[i, j] for j in zgjedhur)

            # kriteri i MIFS 
            score = rendesia - beta * tepria
            scores.append(score)

        # Zgjidhja e atributeve me rezultatin më të lartë të MIFS
        metemire = np.argmax(scores)
        zgjedhur.append(metemire)

    return zgjedhur

from sklearn.datasets import load_iris

iris = load_iris()  # Ngarkimi i te dhenave
X = iris.data       # Ndarja e atributeve
y = iris.target    # Zgjedhja e qellimit
emrate_atributeve = iris.feature_names

zgjedhur = MIFS(X,y,k=2,beta=0.5)

print("Indeksi i atributit te zgjedhur:", zgjedhur)
print("\nAtributet e zgjedhura:")
for i in zgjedhur:
    print(i, " ", emrate_atributeve[i])
