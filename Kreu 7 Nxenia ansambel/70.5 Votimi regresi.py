import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import VotingRegressor
from matplotlib import pyplot

# 1. Ngarkimi i një dataseti real për Regres (Diabetes Dataset)
# Ky dataset përdoret për të parashikuar një vlerë numerike (progresin e sëmundjes)
diabeti = load_diabetes(as_frame=True)
X = diabeti.data
y = diabeti.target

# Ndarja e të dhënave në Stërvitje (80%) dhe Testim (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Funksion për të krijuar ansamblin e regresit
# Krijojmë Decision Trees me thellësi (max_depth) të ndryshme që të kenë "mendime" të ndryshme
def get_voting():
    models = list()
    models.append(('DT2', DecisionTreeRegressor(max_depth=2, random_state=42)))
    models.append(('DT3', DecisionTreeRegressor(max_depth=3, random_state=42)))
    models.append(('DT5', DecisionTreeRegressor(max_depth=5, random_state=42)))
    models.append(('DT7', DecisionTreeRegressor(max_depth=7, random_state=42)))
    models.append(('PaKufi', DecisionTreeRegressor(max_depth=None, random_state=42)))
    
    # Krijojmë ansamblin e regresit (bën mesataren e parashikimeve të modeleve më lart)
    ansambli = VotingRegressor(estimators=models)
    return ansambli

# 3. Funksion për të mbledhur të gjithë modelet që do të vlerësojmë
def get_models():
    models = dict()
    models['DT2'] = DecisionTreeRegressor(max_depth=2, random_state=42)
    models['DT3'] = DecisionTreeRegressor(max_depth=3, random_state=42)
    models['DT5'] = DecisionTreeRegressor(max_depth=5, random_state=42)
    models['DT7'] = DecisionTreeRegressor(max_depth=7, random_state=42)
    models['PaKufi'] = DecisionTreeRegressor(max_depth=None, random_state=42)
    models['Votimi'] = get_voting()
    return models

# 4. Vlerësimi i modelit me Cross-Validation
# Për regres përdorim RepeatedKFold (jo Stratified) dhe si metrikë 'neg_mean_absolute_error' (MAE)
def evaluate_model(model, X, y):
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # cross_val_score kthen vlera negative për gabimin (MAE), prandaj i kthejmë në pozitive më pas
    scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    return np.abs(scores)

# Ekzekutimi dhe krahasimi i modeleve
models = get_models()
results, names = list(), list()

print("Gabimi Mesatar Absolut (MAE) - Sa më i ulët, aq më mirë:")
for name, model in models.items():
    scores = evaluate_model(model, X_train, y_train)
    results.append(scores)
    names.append(name)
    print('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))

# Ndërtimi i grafikut Boxplot për të parë gabimin e secilit model
pyplot.boxplot(results, tick_labels=names, showmeans=True)
pyplot.title("Krahasimi i Gabimit (MAE) të modeleve")
pyplot.ylabel("Gabimi absolut")
pyplot.show()

# 5. Stërvitja e Ansamblit dhe Parashikimi i një rasi të ri
ansambel_regres = models['Votimi']
ansambel_regres.fit(X_train, y_train)

# Marrim rreshtin e parë nga dataseti i testimit për të bërë një parashikim provë
nje_pacient = X_test.iloc[[0]]
vleara_reale = y_test.iloc[0]

parashikimi = ansambel_regres.predict(nje_pacient)

print("-" * 50)
print(f"Vlera REALE e progresit të diabetit: {vleara_reale}")
print(f"Vlera e PARASHIKUAR nga Ansambli: {parashikimi[0]:.2f}")
print("-" * 50)
