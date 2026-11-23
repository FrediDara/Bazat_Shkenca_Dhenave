import numpy as np

class Perceptron:
    def __init__(self, shpejtesia = 0.1, n_iters = 100):
        self.shp = shpejtesia
        self.n_iters = n_iters
        self.peshat = None
        self.animi = None


    def fit(self, X, y):
        n_raste, n_atrib = X.shape
        self.peshat = np.zeros(n_atrib)
        self.animi = 0

        y_ = np.array([1 if i > 0 else 0 for i in y])
     
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                rezultati=np.dot(x_i, self.peshat)+ self.animi
                y_parashikuar = self.aktivizo(rezultati)
                # Rregulli i perditesimit te Perceptron-it 
                ndryshimi = self.shp * (y_[idx] - y_parashikuar)
                self.peshat += ndryshimi * x_i 
                self.animi += ndryshimi 


    def parashiko(self, hyrja):
        rezultati = np.dot(hyrja, self.peshat) + self.animi
        y_parashikuar = self.aktivizo(rezultati)
        return y_parashikuar


    def aktivizo(self,z):
        return np.where(z>=0, 1, 0) 


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

iris = load_iris()       # Ngarkimi i datasetit Iris

X = iris.data[:,:2]     # Marrja e atributeve dhe qellimit
y = np.where(iris.target!=0, 0, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.20, random_state=234, stratify = y)


# Krijimi i Perceptron-it me dy atribute
perceptron = Perceptron(shpejtesia = 0.01, n_iters = 1000)

# Stervitja e Perceptron-it ne te dhenat e stervitjes 
perceptron.fit(X_train, y_train)

# Marrja e parashikimeve nga bashkesia e testimit
y_pred = np.array([perceptron.parashiko(x) for x in X_test])


# vizualizimi
import matplotlib.pyplot as plt
import pandas as pd
df = pd.DataFrame(X_test, columns=iris.feature_names[:2])
df['Parashikimi'] = y_pred

columns=['gjatesia e sepalit (cm)', 'gjeresia e sepalit (cm)', 'Parashikimi'] 
df.columns = columns
xmin, xmax = df[columns[0]].min(),df[columns[0]].max()
X_range = np.linspace(xmin,xmax,39)

df.plot.scatter(x=columns[0],y=columns[1], c=columns[2], colormap='viridis')
w = perceptron.peshat
b = perceptron.animi
y = -(w[0] / w[1]) * X_range - b / w[1] 
plt.plot(X_range,y)



# 1. Use 'gray' colormap for clear dark/light printing contrast
# 2. Added edgecolors and linewidths so white/light dots don't vanish into the background
ax = df.plot.scatter(
    x=columns[0], 
    y=columns[1], 
    c=columns[2], 
    colormap='gray',       # Perfect for black and white printing
    edgecolors='black',    # Puts a sharp outline around every dot
    linewidths=1,
    sharex=False,fontsize=13           # Prevents a common pandas plotting bug with x-axis labels
)

w = perceptron.peshat
b = perceptron.animi
y = -(w[0] / w[1]) * X_range - b / w[1] 

# 3. Style the decision boundary line to be stark black and clearly visible
ax.plot(X_range, y, color='black', linestyle='--', linewidth=2, label='Kufiri i vendimit')
plt.title("Ndarja me anë të perceptronit",fontsize=16)
ax.set_xlabel("Gjatesia e petalit (cm)",fontsize=13 )
ax.set_ylabel("Gjeresia e petalit (cm)",fontsize=13 )
ax.tick_params(axis='both', labelsize=11) # Changes size of numbers on axes
plt.legend(fontsize=13 )
plt.show()