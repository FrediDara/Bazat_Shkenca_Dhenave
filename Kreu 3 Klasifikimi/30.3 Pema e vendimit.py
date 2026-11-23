import numpy as np # algjebra lineare
import pandas as pd # perpunimi i te dhenave, H/D skedaret
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt 

Lentet = pd.read_csv('Lenses.data',sep='\s+',header=None)
Lentet.rename(columns={0:'idd', 1:'age', 2:'perscription', 3:'astigmatic', 4:'tears',5:'Lentet'}, inplace=True)
Lentet.head(16)

import matplotlib.pyplot as plt
import seaborn as sns    #sns.set(font_scale=1.5)
hm = sns.heatmap(Lentet.corr(),
            cbar=True, #brezi anesor me intensitet
            square=True, yticklabels=Lentet.columns, #Etiketat Y 
            xticklabels=Lentet.columns) # Etiketat X 
plt.show()

X = Lentet.drop(["Lentet","idd"], axis=1).values
Y = Lentet["Lentet"].values

X_strv, X_test, Y_strv, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
X_strv.shape

model = DecisionTreeClassifier(criterion="gini") 
model.fit(X_strv, Y_strv)

y_pred_strv = model.predict(X_strv)
y_pred = model.predict(X_test) # Parashikimi i modelit

saktesia_strv = accuracy_score(Y_strv, y_pred_strv)
saktesia_test = accuracy_score(Y_test, y_pred)

print("SAKTESIA: STERVITJA =%.4f TESTIMI=%.4f" % (saktesia_strv,saktesia_test))

from sklearn import tree

fig, ax = plt.subplots(figsize=(6, 6)) 
tree.plot_tree(model, ax=ax, feature_names = ['age', 'persciption', 'astigmatic', 'tears']) 
