from sklearn.datasets import fetch_openml
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LinearRegression
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import matplotlib.pyplot as plt


# 1. Ngarkimi i te dhenave nga OpenML duke perdorur ID-ne e datasetit
# 'as_frame=True' kthen automatikisht te dhenat si DataFrame e Pandas
boston = fetch_openml(data_id=531, as_frame=True, parser="auto")
boston = fetch_openml(name="boston", version=1,  as_frame=True, parser="auto")
# 2. Marrja e matricave X dhe y (OpenML i ka te emertuara gati kolonat)
X = boston.data
y = boston.target

# 3. Përzgjedhja e atributeve me SFS
sfs = SFS(estimator=LinearRegression(), k_features=6,
           forward=True, floating=True, scoring='neg_mean_squared_error', cv=2) #scoring='r2', 
sfs.fit(X, y)       # Stervitja e SFS

# 4. Vizualizimi i rezultateve
fig = plot_sfs(sfs.get_metric_dict(), kind='std_err')


plt.title('Zgjedhja Sekuenciale nga Fillimi (w. StdErr)')
plt.xlabel('Numri i atributeve')
plt.grid(True)
plt.show()


# 3. Vizatimi i grafikut me ngjyra të personalizuara (p.sh. e Gjelbër / Green)
fig, ax = plot_sfs(sfs.get_metric_dict(), kind='std_err')

# Ndryshimi i ngjyrës së linjës kryesore dhe pikave
ax.get_lines()[0].set_color('forestgreen')
ax.get_lines()[0].set_marker('o')

# Ndryshimi i ngjyrës së brezit të hijes (Standard Error band)
# plot_sfs krijon një objekt PolyCollection për hije, i cili është te ax.collections
if ax.collections:
    ax.collections[0].set_color('darkgreen')
    ax.collections[0].set_alpha(0.15) # Transparenca

plt.title('Zgjedhja Sekuenciale nga Fillimi (w. StdErr)')
plt.xlabel('Numri i atributeve')
plt.grid(True)
plt.show()