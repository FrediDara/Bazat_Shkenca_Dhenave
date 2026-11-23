from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

data = load_diabetes()   # Ngarkimi i te dhenave
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)   # Ndarja strervitje/testim

# Krijimi i objektit Regresi Linear (pa regullim)
lr = LinearRegression()   
lr.fit(X_train, y_train)
print("=" * 35)
print(f"Regresi Linear MSE: {mean_squared_error(y_test, lr.predict(X_test)):.2f}")
print(f"R^Regresi Linear: {lr.score(X_train, y_train):.2f}")
print("=" * 35); print()

# Krijimi i objektit Regresi Ridge 
ridge = Ridge(alpha=0.1) 
ridge.fit(X_train, y_train)
print("=" * 35)
print(f"Ridge MSE: {mean_squared_error(y_test, ridge.predict(X_test)):.2f}")
print(f"R^2 Ridge: {ridge.score(X_train, y_train):.2f}")
print("=" * 35); print()

lasso = Lasso(alpha=0.01)   # Krijimi i objektit Regresi Lasso
lasso.fit(X_train, y_train)
print("=" * 35)
print(f"Lasso MSE: {mean_squared_error(y_test, lasso.predict(X_test)):.2f}")
print(f"R^2 Lasso: {lasso.score(X_train, y_train):.2f}")
print("=" * 35); print()

# Krijimi i objektit Rrjeta Elastike 
elastic = ElasticNet(alpha=0.053, l1_ratio=0.9) # me e shpeshta
elastic.fit(X_train, y_train)
print("=" * 35)
print(f"Rrjeta Elastike MSE: {  mean_squared_error(y_test, elastic.predict(X_test)):.2f}")
print(f"R^2 Rrjeta Elastike: {elastic.score(X_train, y_train):.2f}")
print("=" * 35)
