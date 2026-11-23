import pandas as pd # importimi i bibliotekave te nevojshme
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('clustering.csv')  # Ngarkimi i te dhenave

X = data[["LoanAmount","ApplicantIncome"]]
plt.scatter(X["ApplicantIncome"],X["LoanAmount"],c='black')
plt.xlabel('Te ardhurat vjetore')
plt.ylabel('Vlera e huase (Ne mije)')
plt.show()

# Zgjedhja e vezhgimeve te rastit
Qendrat = (X.sample(n=3)) 
plt.scatter(X["ApplicantIncome"],X["LoanAmount"],c='black')
plt.scatter(Qendrat["ApplicantIncome"], 
           Qendrat["LoanAmount"], s=100, marker='x', c='red')
plt.xlabel('Te ardhurat vjetore')
plt.ylabel('Vlera e huase (Ne mije)')
plt.show()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, init='k-means++',random_state=42)
kmeans.fit(X_scaled)               # Stervitja e KMeans

data['Cluster'] = kmeans.labels_    # Shtimi i etiketave 

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans.labels_, cmap='viridis')

qendrat = kmeans.cluster_centers_     # Shfaqja e qendrave

plt.scatter(qendrat[:, 0], qendrat[:, 1], s=300, c='red', marker='X')
plt.xlabel("Te ardhurat (te shkallezuara)")
plt.ylabel("Vlera e huase (e shkallezuar)")
plt.title("Grupimi K-Means i Aplikanteve te kredise")
plt.show()


import numpy as np
klusterat_unike = np.unique(kmeans.labels_)

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