import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error



X = np.random.rand(100, 10) # 100 samples, 10 features
y = 2 * X[:, 0] + 3 * X[:, 1] - 1 + np.random.randn(100) * 0.5
X_train, X_test, y_train, y_test = train_test_split( 
X, y, test_size=0.3, random_state=42)

elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, random_state=42)
elastic_net.fit(X_train, y_train)

y_pred = elastic_net.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mesatarja e Gabimit kuadratik: {mse}")

print(f"Koeficientët: {elastic_net.coef_}")
print(f"Prerja: {elastic_net.intercept_}")
