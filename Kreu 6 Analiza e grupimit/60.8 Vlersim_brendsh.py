from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from permetrics import ClusteringMetric
from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score

iris = load_iris()              # 1. Ngarkimi i te dhenave Iris 
X = iris.data
y = iris.target

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)   # Modeli i grupimit
kmeans.fit(X)                   # Stërvitja e modelit 
labels = kmeans.labels_
y_pred = kmeans.fit_predict(X)
silueta = silhouette_score(X, labels )
cm = ClusteringMetric(X=X, y_pred=labels, decimal = 5)
db_index = davies_bouldin_score(X, labels)

ch_index = calinski_harabasz_score(X, labels)

print(f"Vlera e Siluetes: {silueta:.2f}")
print(f"Indeksi Davies-Bouldin: {db_index:.2f}")

print("Indeksi Dunn", cm.dunn_index())

print(f"Indeksi Calinski-Harabasz: {ch_index:.2f}")



