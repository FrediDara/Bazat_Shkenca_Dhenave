import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

# Për modelet, ansamblin dhe kalibrimin e ri të parashikimeve
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import VotingClassifier

# Për llogaritjet matematike (mean dhe std)
from numpy import mean, std

# Për ndërtimin e grafikut (pyplot)
from matplotlib import pyplot


# Funksion ndihmës për të krijuar një model SVC të shkallëzuar dhe të kalibruar për Soft Voting
def nderto_model(degree):
    # 1. Krijojmë modelin bazë SVC (pa parametrin e vjetëruar probability=True)
    svc_baze = SVC(kernel='poly', degree=degree, random_state=42)
    
    # 2. E kalibrojmë që të llogarisë saktë probabilitetet (zgjidhja e paralajmërimit)
    svc_kalibruar = CalibratedClassifierCV(svc_baze)
    
    # 3. E fusim në një Pipeline me StandardScaler që të dhënat të shkallëzohen automatikisht
    modeli_final = make_pipeline(StandardScaler(), svc_kalibruar)
    return modeli_final


def get_voting():
    models = list()
    models.append(('svm1', nderto_model(degree=1)))
    models.append(('svm2', nderto_model(degree=2)))
    models.append(('svm3', nderto_model(degree=3)))
    models.append(('svm4', nderto_model(degree=4)))
    models.append(('svm5', nderto_model(degree=5)))
    
    ansambli = VotingClassifier(estimators=models, voting='soft')
    return ansambli


def get_models():      
    models = dict()
    models['svm1'] = nderto_model(degree=1)
    models['svm2'] = nderto_model(degree=2)
    models['svm3'] = nderto_model(degree=3)
    models['svm4'] = nderto_model(degree=4)
    models['svm5'] = nderto_model(degree=5)
    models['soft_voting'] = get_voting()
    return models


def evaluate_model(model, X, y): 
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # Përdorim n_jobs=1 nëse kodi ngesh apo jep vonesa gjatë paralelizimit
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    return scores


# Lexoni skedarin CSV 
def get_dataset():
    # Ndryshuam emrin e skedarit sipas emrit që keni vendosur në kodin tuaj
    df = pd.read_csv('heart_cleveland_hungary.csv')
    X = df.drop(columns=['target']) 
    y = df['target']
    return X, y


# Ekzekutimi kryesor
X, y = get_dataset()      
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = get_models()
results, names = list(), list()

for name, model in models.items():     
    scores = evaluate_model(model, X_train, y_train)  
    results.append(scores)
    names.append(name)
    print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))

# Ndërtimi i grafikut
pyplot.boxplot(results, tick_labels=names, showmeans=True)   
pyplot.show()        

# Stërvitja e ansamblit dhe parashikimi i një rreshti
ansambel = models.get("soft_voting")
ansambel.fit(X_train, y_train)

# Parashikimi për rreshtin numër 10 të datasetit të testimit
yhat = ansambel.predict(X_test.iloc[[10]])
print('Klasa e parashikuar:', yhat)
