import pandas as pd
import matplotlib.pyplot as plt
import kmedoids
from sklearn.metrics import pairwise_distances

data = pd.read_csv('clustering.csv')
X = data[["LoanAmount","ApplicantIncome"]]
dist_matrix = pairwise_distances(X, metric='euclidean')

kostot = []
intervali = range(2, 10)
for k in intervali:
    kmds =  kmedoids.KMedoids(n_clusters=k, method='fasterpam',metric='euclidean',random_state=42)
    kmds.fit(dist_matrix)
    kostot.append(kmds.inertia_) 

plt.plot(intervali, kostot, marker='o', linestyle='--')
plt.xlabel("Numri i Grupeve (K)")
plt.ylabel("Kosto e pergjithshme (Inercia)")
plt.title("Metoda e berrylit per K optimale ")
plt.show()

