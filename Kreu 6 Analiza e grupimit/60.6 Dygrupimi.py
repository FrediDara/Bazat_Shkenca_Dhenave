import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import SpectralBiclustering

df = pd.read_csv("gene_expression.csv")

# Keep gene names separately
gene_names = df.iloc[:, 0]

# Use only numerical expression values
data = df.iloc[:, 1:].to_numpy()

model = SpectralBiclustering(n_clusters=4)
model.fit(data)


# Reorder rows and columns
fit_data = data[np.argsort(model.row_labels_)]
fit_data = fit_data[:, np.argsort(model.column_labels_)]

# Plot
plt.imshow(fit_data, cmap='viridis', aspect='auto')
plt.title("Rezultati i dygrupimit spektral")
plt.colorbar()
plt.show()