import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Ngarkimi i të dhënave
df = pd.read_csv('polinom_brom.csv')
X = df[['Kapaciteti']]
y = df[['Temp']]

lin = LinearRegression()
lin.fit(X, y)


poli2 = PolynomialFeatures(degree=2)
X_poli2 = poli2.fit_transform(X)

poli2.fit(X_poli2, y)
lin2 = LinearRegression()
lin2.fit(X_poli2, y)

# Vizualizojmë rezultatet e modelit të regresit linear 

plt.scatter(X, y, color='blue')
plt.plot(X, lin2.predict(X_poli2), color='red')
plt.title(' Regresi kuadratik ')
plt.xlabel('Kapaciteti')
plt.ylabel('Temperatura')
plt.show()


pred2 = 11.5
pred2array = np.array([[pred2]])
lin2.predict(poli2.fit_transform(pred2array))
