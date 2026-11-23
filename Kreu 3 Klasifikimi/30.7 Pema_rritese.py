
import pandas as pd
from sklearn.datasets import load_iris

# Load the dataset directly as a pandas DataFrame
iris_data = load_iris(as_frame=True)
df = iris_data.frame

# View the first few rows
print(df.head())




from sklearn.datasets import load_iris

# Load the dataset directly as a Pandas DataFrame
iris_bunch = load_iris(as_frame=True)

# Access the combined dataframe (includes features and target)
df = iris_bunch.frame

print(df.head())