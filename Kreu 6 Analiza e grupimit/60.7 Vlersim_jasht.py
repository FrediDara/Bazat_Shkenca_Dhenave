from sklearn.cluster import KMeans
from sklearn.metrics import jaccard_score, mutual_info_score,\
normalized_mutual_info_score,adjusted_mutual_info_score, \
          adjusted_rand_score, rand_score, fowlkes_mallows_score
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

FM_score = fowlkes_mallows_score(iris.target, kmeans.labels_) 
jacc_koef = jaccard_score(iris.target, kmeans.labels_, average="macro")
randscore = rand_score(iris.target, kmeans.labels_)
ari = adjusted_rand_score(iris.target, kmeans.labels_)
mi = mutual_info_score(iris.target, kmeans.labels_)
ami = adjusted_mutual_info_score(iris.target, kmeans.labels_)
nmi = normalized_mutual_info_score(iris.target, kmeans.labels_)

print(f"Indeksi Fowlkes-Mallows: {FM_score}")
print(f"Koeficienti Jaccard: {jacc_koef}")
print(f"Indeksi Rand: {randscore:.2f}")
print(f"Indeksi Rand i rregulluar: {ari:.2f}")
print(f"Informacioni reciprok(MI): {mi:.2f}")
print(f"Informacioni reciprok i rregulluar (AMI): {ami:.2f}")
print(f"Informacioni reciprok i normalizuar (NMI): {nmi:.2f}")
