import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv('heart_cleveland_hungary.csv')
X = df.drop(columns=['target']).values # I kthejmë në numpy array 
y = df['target'].value                 # për lehtësi manipulimi

X_tr, X_te, y_tr, y_te = train_test_split(X,y,test_size = 0.2, 
random_state=42)  # Ndarja e dhënave stervitje/testim

X_et,X_pa,y_et,y_pa = train_test_split(X_tr,y_tr,train_size=30, 
                   random_state=42)
svc_kalib = make_pipeline(StandardScaler(), 
CalibratedClassifierCV(SVC(random_state=42)))
classifiers = [svc_kalib, # Klasifikuesit ansambel 
               RandomForestClassifier(), GaussianNB()]

prag_besimi = 0.8     # Cikli gjysme i mbikëqyrur
max_iters = 5

for iteration in range(max_iters):
    print(f"Iteracioni {iteration+1}")

    for clf in classifiers:    # stërvitja e klasifikuesve
        clf.fit(X_et, y_et)

    raste_bes = []
    etiketa_bes = []

    for i, x in enumerate(X_pa):
        preds = []
        probs = []

        for clf in classifiers:
            pred = clf.predict([x])[0]
            prob = np.max(clf.predict_proba([x]))
            preds.append(pred)
            probs.append(prob)

        # Votimi Mazhoritar 
        majority_label = max(set(preds), key=preds.count)
        agreement = preds.count(majority_label) / len(preds)

        if agreement >= 0.66 and np.mean(probs) >= prag_besimi:
            raste_bes.append(x)
            etiketa_bes.append(majority_label)

    if not raste_bes:
        break

    # Shtimi i rasteve te besueshme te bashkësia e etiketave
    X_lab = np.vstack([X_et, raste_bes])
    y_lab = np.hstack([y_et, etiketa_bes])

    # Largimi i tyre nga bashkësia e paetiketuar
    mask = np.ones(len(X_pa), dtype=bool)
    mask[:len(raste_bes)] = False
    X_unlab = X_pa[mask]

print("Madhesia e te dhënave finale te etiketuara:", len(X_lab))
