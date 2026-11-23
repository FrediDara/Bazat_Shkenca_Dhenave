import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# 1. NGARKIMI I DATASETIT REAL
df = pd.read_csv('heart_cleveland_hungary.csv')
X = df.drop(columns=['target']).values # I kthejmë në numpy array për lehtësi manipulimi
y = df['target'].values 

# Ndajmë një pjesë për testim final (20%) që të shohim nëse modeli përmirësohet
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SIMULIMI I TË DHËNAVE GJYSMË TË MBIKËQYRARE
# Nga pjesa e stërvitjes, marrim vetëm 30 raste të etiketuara (shumë pak)
# dhe pjesën tjetër (psh. 700+ raste) e bëjmë pa etiketa për ciklin tonë
X_lab, X_unlab, y_lab, _ = train_test_split(X_train_full, y_train_full, train_size=30, random_state=42)


# 2. NDËRTIMI I KLASIFIKUESVE (Zgjidhja e paralajmërimit)
# Përdorim CalibratedClassifierCV dhe make_pipeline për SVC-në e re
svc_e_kalibruar = make_pipeline(StandardScaler(), 
                                CalibratedClassifierCV(SVC(random_state=42)))

classifiers = [svc_e_kalibruar, 
    RandomForestClassifier(random_state=42), 
    GaussianNB()]

confidence_pragu = 0.8     # Cikli gjysmë i mbikëqyrur
max_iters = 5

for iteration in range(max_iters):
    print(f"Iteracioni {iteration+1}")

    # Stërvitja e klasifikuesve
    for clf in classifiers:    
        clf.fit(X_lab, y_lab)

    raste_bes = []
    etiketa_bes = []
    indekset_e_perzgjedhura = [] # Ndihmon për të fshirë saktë të dhënat e përdorura

    for i, x in enumerate(X_unlab):
        preds = []
        probs = []

        for clf in classifiers:
            pred = clf.predict([x])[0]
            # Merr probabilitetin maksimal për klasën e parashikuar
            prob = np.max(clf.predict_proba([x]))
            preds.append(pred)
            probs.append(prob)

        # Votimi Mazhoritar 
        majority_label = max(set(preds), key=preds.count)
        agreement = preds.count(majority_label) / len(preds)

        # Kontrolli i kushteve të besueshmërisë
        if agreement >= 0.66 and np.mean(probs) >= confidence_pragu:
            raste_bes.append(x)
            etiketa_bes.append(majority_label)
            indekset_e_perzgjedhura.append(i) # Ruajmë indeksin ekzak të elementit

    if not raste_bes:
        print("Nuk u gjet asnjë rast i ri i besueshëm. Ndalohet cikli.")
        break

    # Shtimi i rasteve të besueshme te bashkësia e etiketave
    X_lab = np.vstack([X_lab, raste_bes])
    y_lab = np.hstack([y_lab, etiketa_bes])

    # KORRIGJIMI I BUG-UT: Largimi i saktë i rasteve nga bashkësia e paetiketuar
    # Krijojmë një maskë ku True do të thotë që e dhëna mbetet, False që fshihet
    mask = np.ones(len(X_unlab), dtype=bool)
    mask[indekset_e_perzgjedhura] = False # Fshijmë vetëm indekset që u përzgjodhën
    X_unlab = X_unlab[mask]
    
    print(f"U shtuan {len(raste_bes)} raste të reja. Kanë mbetur {len(X_unlab)} pa etiketuar.")

print("\nMadhësia e të dhënave finale të etiketuara:", len(X_lab))
