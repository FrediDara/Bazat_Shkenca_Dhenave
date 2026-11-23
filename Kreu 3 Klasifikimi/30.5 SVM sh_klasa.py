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


from sklearn.model_selection import train_test_split

X = df.drop(['sepal length (cm)', 'sepal width (cm)', 'target', 'species'], axis=1)
y = df.target
X_strv, X_test, y_strv, y_test = train_test_split(X, y, test_size=0.25, random_state=42,stratify=y)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Perdorimi i berthames se nenkuptuar 
model = SVC()  
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



# Apply a clean, light background style
plt.style.use('seaborn-v0_8-whitegrid')


# Define markers for each unique class in y
markers = ['^', 'o', 's','+']  # Circle for class 0, Square for class 1, Triangle for class 2
unique_classes = np.unique(y)
colors = plt.cm.coolwarm(np.linspace(0, 1, len(unique_classes)))

kernels = ['linear', 'rbf', 'poly']

for kernel in kernels:
    model = SVC(kernel=kernel)
    model.fit(X_strv, y_strv)

    pred = model.predict(X_test)
    print("Saktesia duke perdorur {}:".format(kernel), accuracy_score(pred, y_test))
    # Create figure and force a solid light background color
    fig, ax = plt.subplots(facecolor='#f7f9fa')  
    ax.set_facecolor('#f7f9fa') # #ffffff Pure white inside the plot grid

    title = ('Kufiri i vendimt SVC ' + model.kernel)
    X0, X1 = X['petal length (cm)'], X['petal width (cm)']
    xx, yy = make_meshgrid(X0, X1)

    # plot_contours(ax, model, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    # ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    

    # Loop through each class to plot with its own marker
    for i, klasa in enumerate(unique_classes):
        idx = (y == klasa)
        ax.scatter(X0[idx], X1[idx], 
            color=colors[i], 
            marker=markers[i], 
            s=30, 
            edgecolors='k', 
            label=f'Klasa {klasa}'  # Label for the legend
        )
    plot_contours(ax, model, xx, yy, cmap=plt.cm.coolwarm, alpha=0.3)
    ax.legend()
    
    
    ax.set_ylabel('Etiketa y',fontsize=14 )
    ax.set_xlabel('Etiketa x',fontsize=14 )
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title,fontsize=16)
    ax.legend()
    plt.show()

'''


kernels = ['linear', 'rbf', 'poly']
for kernel in kernels:
    model = SVC(kernel=kernel)
    model.fit(X_strv, y_strv)

    pred = model.predict(X_test)
    print("Saktesia duke perdorur {}:".format(kernel), accuracy_score(pred, y_test))
    fig, ax = plt.subplots()
    title = ('Kufiri i vendimt SVC ' + model.kernel)
    X0, X1 = X['petal length (cm)'], X['petal width (cm)']
    xx, yy = make_meshgrid(X0, X1)

    plot_contours(ax, model, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_ylabel('Etiketa y')
    ax.set_xlabel('Etiketa x ')
    ax.set_title(title)
    ax.legend()
    plt.show()
'''