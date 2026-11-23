import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=150, centers=4, cluster_std=0.60, random_state=42)

inertia_values = []
k_range = range(1, 7)
for k in range(1, 7):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(X)
    inertia_values.append(kmeans.inertia_)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(k_range,inertia_values,marker='o',color='b',linewidth=2)
ax1.set_title("K-Mesatar: Metoda e berrylit")
ax1.set_xlabel("Numri i grupeve (k)")
ax1.set_ylabel("Inercia (gabimi)")
ax1.grid(True)

Z = linkage(X, method='ward')
dendrogram(Z, ax=ax2, truncate_mode='lastp', p=15)
ax2.set_title("Hierarkik: Dendrograma")
ax2.set_xlabel("Te dhenat (ose grupet)")
ax2.set_ylabel("Largesa (Mosngjashmeria)")

plt.tight_layout()
plt.show()


