import pandas as pd
# import numpy as np
from keras.datasets import boston_housing 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
# from sklearn.metrics import r2_score

# 1. Ngarkimi i te dhenave te shtepive te Boston-it
(X, y), (_, _) = boston_housing.load_data(test_split=0) 

# 2. Percaktimi sipas radhes te shtyllave ne te dhenat
atribute_emer = [ 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 
    'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'] 

# 3. Krijimi i dataframe dhe vendosja e etiketave
X = pd.DataFrame(X, columns=atribute_emer)
y = pd.DataFrame(y, columns=['MEDV'])

# 4. Ndarja e te dhenave
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) 

# 5. Percaktimi i Zgjedhjes Sekuenciale nga Fillimi (sfs)
sfs = SFS(LinearRegression(), k_features=5, forward=True,
          floating=False, scoring = 'r2', cv = 5)
# 6. Stervitja e SFS
sfs.fit(X, y)        

df_rezultate = pd.DataFrame(sfs.subsets_).transpose()
print("\nDataFrame SFS:\n", df_rezultate)


print(f"Indekset e atributeve te zgjedhura: {sfs.k_feature_idx_}")
print(f"Emra e atributeve te zgjedhura: {sfs.k_feature_names_}")
print(f"Vlera CV R2 me e mire: {sfs.k_score_:.3f}")
