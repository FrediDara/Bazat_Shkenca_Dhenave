from sklearn import datasets
import pandas as pd
import numpy as np

iris = datasets.load_iris() # Ngarkimi i te dhenave
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Pasqyrimi i qellimit ne emra 
emrat_qellimi = {0:'setosa', 1:'versicolor', 2:'virginica'}

df['species'] = df['target'].map(emrat_qellimi) 
df['species'] = df['target'].replace(emrat_qellimi)
import matplotlib.pyplot as plt

setosa = df[df.species == "setosa"]
versicolor = df[df.species=='versicolor']
virginica = df[df.species=='virginica']

fig, ax = plt.subplots()
fig.set_size_inches(13, 7) # vendosja e gjatësisë dhe gjerësisë 

# etiketat dhe pikat 
ax.scatter(setosa['petal length (cm)'], setosa['petal width (cm)'], label="Setosa Petal", facecolor="blue" ,marker="x")
ax.scatter(versicolor['petal length (cm)'],versicolor['petal width (cm)'], label="Versicolor", facecolor="green", marker="o")
ax.scatter(virginica['petal length (cm)'], virginica['petal width (cm)'], label="Virginica", facecolor="red", marker="^")

ax.set_xlabel("Gjatesia e petalit (cm)",fontsize=16 )
ax.set_ylabel("Gjeresia e petalit (cm)",fontsize=16 )
ax.grid()
ax.set_title("Petalet Iris ",fontsize=20 )
ax.legend(fontsize=12)


# Pasqyrimi i qellimit ne emrat e specieve 
emrat_qellimi = {0:'setosa',1:'non-setosa', 2:'non-setosa'}
df['species'] = df['target'].map(emrat_qellimi)

# Pasqyrimi speciet te numrat e qellimit 
numrat_qellimi = {'setosa':0,'non-setosa':1} 
df['target'] = df['species'].map(numrat_qellimi)

from sklearn.model_selection import train_test_split

X = df.drop(['sepal length (cm)', 'sepal width (cm)', 'target', 'species'], axis=1)
y = df.target
X_strv, X_test, y_strv, y_test = train_test_split(X, y, test_size=0.25, random_state=42,stratify=y)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Perdorimi i berthames lineare 
model = SVC(kernel='linear') 
model.fit(X_strv, y_strv)
pred = model.predict(X_test)
print("Saktesia {}:", accuracy_score(pred, y_test))

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

pred = model.predict(X_test)
print("Saktesia duke perdorur berthamen lineare:", accuracy_score(pred, y_test))
fig, ax = plt.subplots()    
title = ('Siperfaqja e vendimit te SVC lineare')
X0, X1 = X['petal length (cm)'], X['petal width (cm)']
xx, yy = make_meshgrid(X0, X1)     # Rrjeta e grafikut

plot_contours(ax, model, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_ylabel('Etiketa y')
ax.set_xlabel('Etiketa x')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title(title)
ax.legend()
plt.show()
