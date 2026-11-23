import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report

data = pd.read_csv("iris.csv")

X = data.drop(columns=["variety"])
y = data["variety"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

scaler = StandardScaler()     # Percaktimi i shkallezuesit

X_train = scaler.fit_transform(X_train)      # Shkallëzimi
X_test = scaler.transform(X_test)

Klasifik_pare = LogisticRegression() # Percaktimi i tre modeleve
Klasifik_dyte = DecisionTreeClassifier()
Klasifik_trete = SVC()

ansambel = VotingClassifier(estimators=[
    ("regresi logjistik", Klasifik_pare), 
    ("pema e vendimit", Klasifik_dyte),
    ("ndaresi i percaktuar nga vektoret", Klasifik_trete)   
    ], voting="hard")

ansambel.fit(X_train, y_train)


parashikime = ansambel.predict(X_test)
raport = classification_report(y_test, parashikime)
print(raport)

