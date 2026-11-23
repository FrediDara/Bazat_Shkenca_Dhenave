import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

customer_data = pd.read_csv('shopping-data.csv')
kolonat = ['Annual Income (k$)', 'Spending Score (1-100)']
data = customer_data[kolonat] 
sns.pairplot(data)

ss = StandardScaler() # krijimi i shkallezuesit
scaled_data = pd.DataFrame(ss.fit_transform(data),columns=kolonat)

# krijimi i vektorit te rastit W, me vlera nga {0,1}
nn = NearestNeighbors(n_neighbors=4) # pikat minimum -1
nbrs = nn.fit(scaled_data)
distances, indices = nbrs.kneighbors(scaled_data)

distances = np.sort(distances, axis=0) # Renditja e largesave
distances = distances[:,1] #Zgjedhja vetem e largesave me te vogla

plt.figure(figsize=(6,3))
plt.plot(distances)                # Vija e berrylit
plt.axhline(y=0.24, color='r', linestyle='--', alpha=0.4) 
plt.title('Grafiku i largesave te fqinjeve')
plt.xlabel('Te dhenat')
plt.ylabel('Vlera epsilon')
plt.show()


dbs = DBSCAN(eps=0.24, min_samples=5)
dbs.fit(scaled_data) 


labels = dbs.labels_

scaled_data['labels'] = dbs.labels_
scaled_data.head()

total_labels = np.unique(labels)

n_labels = 0
for n in total_labels:
    if n != -1:
        n_labels += 1
print("Numri i grupeve:", n_labels)

sns.scatterplot(data=scaled_data, x='Annual Income (k$)', hue='labels',
          y='Spending Score (1-100)', 
           palette='muted').set_title('Grupet e gjetura nga DBSCAN')


from sklearn.metrics import silhouette_score

s_score = silhouette_score(scaled_data, labels)
print(f"Silueta: {s_score:.3f}")


sns.scatterplot(
    data=scaled_data,
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    hue='labels',
    style='labels',
    markers = {-1: 'X',   # Noise
     0: 'o',     1: '^',     2: 's',     3: 'D',     4: 'P',     5: '*'},
    palette='muted'
).set_title('Grupet e gjetura nga DBSCAN')
plt.legend(title='Etiketa')