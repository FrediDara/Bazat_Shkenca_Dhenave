import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

iris = load_iris() 
X = iris.data
y = iris.target 

scaler = StandardScaler()     # Transformimi
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_scaled)

optics_model = OPTICS(min_samples=5, max_eps=np.inf, cluster_method='xi')

optics_model.fit(X_reduced)    
labels = optics_model.labels_

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f'Numri i vleresuar i grupeve: {n_clusters}')
print(f'Cluster labels: {labels}')

plt.figure(figsize=(10, 5))
reachability_distances = optics_model.reachability_[ optics_model.ordering_]
plt.plot(reachability_distances)
plt.title('Grafiku i largesës së arritshmërisë për të dhënat Iris')
plt.xlabel(' Pikët (të renditura sipas arritshmërisë)')
plt.ylabel('Largesa e arritshmërisë')
plt.grid(True)
plt.show()

unique_labels = set(labels) # Paraqitja e grupeve
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

plt.figure(figsize=(9, 7))
for k, col in zip(unique_labels, colors):
    if k == -1:
        col = [0, 0, 0, 1]  # Ngjyra e zeze per zhurmat

    class_member_mask = (labels == k)
    xy = X_reduced[class_member_mask]
    plt.scatter(xy[:, 0], xy[:, 1], color=tuple(col), s=50, label=f'Cluster {k}')

plt.title(f'Grupimi OPTICS mbi Iris (Grupet e vleresuara: {n_clusters})')
plt.xlabel('PCA atributi 1')
plt.ylabel('PCA atributi 2')
plt.legend()
plt.show()
