
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Ngarkimi i të dhënave
df= pd.read_csv('heat_cost.csv')
X = df[['Temp', 'Madh']]
y = df['Kosto']

regr = LinearRegression()
regr.fit(X, y)
slr_koef = regr.coef_  # Koeficienti dhe Pikeprerja me origjinen
slr_pikprer = regr.intercept_
# Ekuacioni i regresit   y = 36.15 - 4.28 * Temp + 60.36 * Madh

slr_score = regr.score(X,y)   # Gjetja e R-katrorit 

from sklearn.metrics import mean_squared_error
import math
y_predict=regr.predict(X)     # Llogaritja e gabimit RMSE 
slr_rmse = math.sqrt(mean_squared_error(y, y_predict))

# Ndëetimi i grafikut te rezultateve.

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X['Temp'], X['Madh'],
           y, color='blue', label='Te Dhenat reale')

x1_range = np.linspace(X['Temp'].min(), X['Temp'].max(), 100)
x2_range = np.linspace(X['Madh'].min(), X['Madh'].max(), 100)
x1, x2 = np.meshgrid(x1_range, x2_range)
z = regr.predict(np.c_[x1.ravel(), x2.ravel()]).reshape(x1.shape)

ax.plot_surface(x1, x2, z, color='red', alpha=0.5, rstride=100, cstride=100)
ax.set_xlabel('Temp mesat')
ax.set_ylabel('Madhesia')
ax.set_zlabel('Kosto e ngrohjes')
ax.set_title('Plani me i mire i Regresit te shumefishte Linear (3D)')
plt.show()
