import pandas as pd
from sklearn.datasets import fetch_openml
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LinearRegression

# 1. Ngarkimi i te dhenave nga OpenML duke perdorur ID-ne e datasetit
# 'as_frame=True' kthen automatikisht te dhenat si DataFrame e Pandas
boston = fetch_openml(data_id=531, as_frame=True, parser="auto")
boston = fetch_openml(name="boston", version=1,  as_frame=True, parser="auto")
# 2. Marrja e matricave X dhe y (OpenML i ka te emertuara gati kolonat)
X = boston.data
y = boston.target

# 3. Përzgjedhja e veçorive me SFFS
sffs = SFS(estimator=LinearRegression(), k_features=5,
           forward=True, floating=True, scoring='r2', cv=0)
sffs.fit(X, y)       # Stervitja e SFS

# 4. Shfaqja e rezultateve
df_SFFS_results = pd.DataFrame(sffs.subsets_).transpose()
df_SFFS_results
