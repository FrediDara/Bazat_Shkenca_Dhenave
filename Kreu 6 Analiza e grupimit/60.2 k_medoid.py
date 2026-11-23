import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from kmedoids import KMedoids

def euclidean(a, b): # Largesa euklidiane
    return np.linalg.norm(a - b)

def manhattan(a, b): # Largesa Manhattan
    return np.sum(np.abs(a - b))

iris = load_iris()   # Ngarkimi i datasetit
X = iris.data

scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X) # Standardizimi

pca = PCA(n_components=2)    # Zvogelimi ne 2D
X_pca = pca.fit_transform(X_scaled)

kmedoids_init = KMedoids(n_clusters=3, init='build', max_iter=0, metric='manhattan',random_state=42) #'k-medoids++' te scikit-learn-extra
kmedoids_init.fit(X_pca)        # Stervitja e modelit kmedoids
initial_medoids = kmedoids_init.cluster_centers_

plt.scatter(X_pca[:, 0], X_pca[:, 1], color='gray', alpha=0.5)

plt.scatter(initial_medoids[:, 0], initial_medoids[:, 1],
            c='red', marker='X', s=300, label='Initial Medoids')

plt.title("Medoidet fillestare (k-medoids++)")
plt.legend()
plt.show()




import numpy as np
klusterat_unike = np.unique(initial_medoids.labels_)

# 1. Define distinct marker shapes for each data cluster
# 'o' = circle, 's' = square, '^' = triangle, 'D' = diamond
markers = ['o', 's', '^'] 

# Loop through each cluster to assign both a unique marker shape and color
for i, k in enumerate(klusterat_unike):
    indekset = (kmeans.labels_ == k)
    
    # Selecting colors manually or grabbing them from a print-friendly map like 'cividis'
    ngjyra = plt.cm.cividis(i / len(klusterat_unike)) 
    
    plt.scatter(
        X_scaled[indekset, 0], 
        X_scaled[indekset, 1], 
        color='white',# ngjyra,
        # edgecolors='black',     # Sharp border line for print clarity
        # linewidths=0.7,
        label=f'Klusteri {k}',
        facecolors='none',     # <--- Makes the inside of the marker completely transparent
        edgecolors='black',      # Sets the color of the outline contour
        marker=markers[i],         # Uses the distinct shapes
        linewidths=1.1,        # Makes the contour outline slightly thicker/bolder
        s=50                   # Slightly larger size so empty shapes are easy to see
    )

# 2. Keep the cluster centers massive and highly distinct (e.g., an 'X' or an empty star '*')
plt.scatter(
    qendrat[:, 0], 
    qendrat[:, 1], 
    s=250, 
    c='red', 
    marker='X',                 # Massive bold 'X' for centroids
    edgecolors='black', 
    linewidths=1.5,
    label='Qendrat'
)

plt.xlabel("Te ardhurat (te shkallezuara)", fontsize=13)
plt.ylabel("Vlera e huase (e shkallezuar)", fontsize=13)
plt.title("Grupimi K-Means i Aplikanteve te kredise", fontsize=14, pad=12)

# Display the legend cleanly so the marker styles match perfectly
plt.legend(fontsize=11, frameon=True, facecolor='white')
plt.show()