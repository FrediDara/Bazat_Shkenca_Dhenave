import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression


# 1. Fetch the Boston dataset from OpenML (as_frame=False returns NumPy arrays)
boston = fetch_openml(name="boston", version=1, as_frame=True, parser="auto")
x = boston.data
y = boston.target

print("Feature data dimension: ", x.shape)

# 2. Set up the estimator and RFE selector
selector = RFE(LinearRegression(), n_features_to_select=5, step=1)
selector = selector.fit(x, y)

# 3. Get the filter mask and ranking
filter_mask = selector.support_
ranking = selector.ranking_

print("Mask data: ", filter_mask)
print("Ranking: ", ranking)

# 4. Print all vs. selected features
Atributet = np.array(boston.feature_names)
print("Te gjitha atributet:")
print(Atributet)

print("Atributet e zgjedhura:")
print(Atributet[filter_mask])



# 3. Create a DataFrame to hold the results matching your table format
results_df = pd.DataFrame({
    'Atributi': boston.feature_names,
    'I zgjedhur': selector.support_,
    'Ranku (RFE)': selector.ranking_
})

# Make the index named 'Columns' to perfectly match your header layout
results_df.index.name = 'Columns'

# 4. Print the formatted table
print(results_df.to_string())
