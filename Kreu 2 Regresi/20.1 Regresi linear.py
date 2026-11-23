import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('pulebardha.csv')

# Ndarja ne X (e pavarur) dhe Y (e parashikuar)
X = df.iloc[:, :-1]  
Y = df.iloc[:,  -1]

mod_reg = LinearRegression()
mod_reg.fit(X, Y)

mod_koef = mod_reg.coef_
mod_pikprer= mod_reg.intercept_
y_predict=mod_reg.predict(X)
slr_score = mod_reg.score(X, Y)

from sklearn.metrics import mean_squared_error
import math
slr_rmse = math.sqrt(mean_squared_error(Y, y_predict))

plt.scatter(X, Y)
x_min, x_max = min(df['X']), max(df['X'])
x_plt =pd.DataFrame(np.linspace(x_min,x_max, num=10))
x_plt.columns =['X']
y_plt=pd.DataFrame(mod_reg.predict(x_plt))

plt.plot(x_plt,y_plt)
plt.ylim(ymin=0)
plt.show()
