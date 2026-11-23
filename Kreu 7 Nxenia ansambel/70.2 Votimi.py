import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.ensemble import VotingRegressor, RandomForestRegressor
import matplotlib.pyplot as plt


raw_df = pd.read_csv("Boston.txt", sep='\s+', skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)

linear_reg = LinearRegression()
svr_reg = SVR(kernel='sigmoid')  

xgb_reg = XGBRegressor(random_state=42)
rf_reg = RandomForestRegressor(random_state=42)

# Krijimi i një Votimi per regres me modelet individuale
voting_reg = VotingRegressor(estimators=[('linear', linear_reg), 
             ('svr', svr_reg), ('xgb', xgb_reg), ('rf', rf_reg)],
             n_jobs=5)

voting_reg.fit(X_train, y_train)

xt = data[:20]

linear_reg.fit(X_train, y_train)
svr_reg.fit(X_train, y_train)
xgb_reg.fit(X_train, y_train)
rf_reg.fit(X_train, y_train)
voting_reg.fit(X_train, y_train)

pred_linear_reg = linear_reg.predict(xt)
pred_svr_reg = svr_reg.predict(xt)
pred_xgb_reg = xgb_reg.predict(xt)
pred_rf_reg = rf_reg.predict(xt)
pred_voting_reg = voting_reg.predict(xt)

plt.figure()
plt.plot(pred_xgb_reg, "go", label="Regresori Gradient Boosting")
plt.plot(pred_rf_reg, "b^", label="Regresori Pylli i Rastit")
plt.plot(pred_svr_reg, "g^", label="Regresori i Vektoreve percaktues")
plt.plot(pred_linear_reg, "bo", label="Regresi Linear")

plt.plot(pred_voting_reg, "r*", ms=10, label="Regresori i Votimit")
plt.plot(target[:20],"y*--", label = 'Pikat e te dhenave te Testimit')

plt.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
plt.ylabel("e parashikuar")
plt.xlabel("rastet e stervitjes")
plt.legend(loc="best")
plt.title("Parashikimet e regresorit dhe mesataret e tyre")

plt.show()

