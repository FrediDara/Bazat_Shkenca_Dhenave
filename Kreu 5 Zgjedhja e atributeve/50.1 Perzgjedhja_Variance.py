# Importimi i bibliotekave te nevojshme
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
np.random.seed(36)

# Krijimi i dataframe-it me dhjetë atribute
data = pd.DataFrame(data={
    "feature_1": np.random.rand(1000),
    "feature_2": np.random.rand(1000) * 10,
    "feature_3": np.random.rand(1000),
    "feature_4": np.random.rand(1000) * 100,
    "feature_5": np.random.rand(1000),
    "feature_6": np.random.rand(1000) * 0.1,
    "feature_7": np.random.rand(1000),
    "feature_8": np.random.rand(1000) * 0.01,
    "feature_9": np.random.rand(1000),
    "feature_10": np.random.rand(1000) * 50})

# Caktimi i pragut 0.1, tregon se nëse varianca e një atributi 
# është më e vogël se 0.1, ai do të hiqet.
selector = VarianceThreshold(threshold=0.1)

# Përshtatja me të dhënat në objektin VarianceThreshold
vlerat = data.values
vlerat_zvogel = selector.fit_transform(vlerat)

# Shfaqja e formës se të dhënave të zvogeluara      # (1000, 3)
print("Forma e të dhënave të zvogeluara: ", vlerat_zvogel.shape) 

atribute_mbaj = data.columns[selector.get_support(indices=True)]
print("Atributet e mbajtura: ", atribute_mbaj)
