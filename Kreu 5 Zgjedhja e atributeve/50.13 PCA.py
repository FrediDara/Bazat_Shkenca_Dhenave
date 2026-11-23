import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = load_iris()  # Ngarkimi i te dhenave IRIS

X=df.data  # Ndarja e atributeve 
Y=df.target  # Ndarja e qellimit    

x = StandardScaler().fit_transform(X) # Standardizimi

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)

principalDf = pd.DataFrame(data = principalComponents, columns =['Komponentja 1', 'Komponentja 2'])

y = pd.DataFrame(data = Y, columns =['target'])
finalDf = pd.concat([principalDf, y], axis = 1)


fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Komponentja kryesore 1', fontsize = 15)
ax.set_ylabel('Komponentja kryesore 2', fontsize = 15)
ax.set_title('PCA me 2 komponente', fontsize = 20)

lulet = ['Setosa', 'Versicolor', 'Virginica']
targets = [0,1,2]
colors = ['r', 'g', 'b']
# Define markers for each unique class in y
markers = ['^', 'o', 's','+'] 

for target, marker in zip(targets,markers):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'Komponentja 1']
            , finalDf.loc[indicesToKeep, 'Komponentja 2']
            ,         facecolors='none',     # <--- Makes the inside of the marker completely transparent
        edgecolors='black',      # Sets the color of the outline contour
        marker=marker,         # Uses the distinct shapes
        linewidths=1.5,        # Makes the contour outline slightly thicker/bolder
        s=60                   # Slightly larger size so empty shapes are easy to see
    )
ax.legend(lulet)
ax.grid()

pca.explained_variance_ratio_
