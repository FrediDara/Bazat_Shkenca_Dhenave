import pandas as pd

# Leximi i skedarit CSV
dataset = pd.read_csv('ridge.csv') 
df = dataset.copy()

# Ndarja ne X (ndryshorja e pavarur)
X= df.iloc[:,0]   
# dhe Y (ndryshorja e varur)
Y= df.iloc[:,-1]  


from sklearn.linear_model import Ridge

ridge = Ridge(alpha = 1)
ridge.fit(X.array.reshape(-1, 1), Y)


ridge_koef = ridge.coef_  # Gjetja e koeficienteve te regresit
ridge_pikprer = ridge.intercept_   # Gjetja e pikeprerjes 


X_plt = [0,1,2,3,4] 
Y_plt = ridge.predict(pd.DataFrame(X_plt))
import matplotlib.pyplot as plt
plt.plot(X_plt, Y_plt)

lmbd = 1
x_pos = X_plt[-1] - (max(X_plt) - min(X_plt)) * 0.1
y_pos = Y_plt[-1] + (max(Y_plt) - min(Y_plt)) * 0.05

plt.text(X_plt[-1], Y_plt[-1], f"y = {ridge_koef[0]:.2f} * X + {ridge_pikprer:.2f} për λ ose α = {lmbd}", 
    fontsize=12)

# plt.text(X_plt[-1], Y_plt[-1],'y = ' + str('%.2f' %ridge_koef[0]) + '* X +' + str('%.2f' %ridge_ pikprer) + 'për $\lambda$ ose $\alpha$ =' str(lmbd), fontsize = 12)
plt.ylim(ymin=0, ymax=9) 
plt.xlim(xmin=0, xmax=8)
plt.show()


landat = [0,1,10,100]
for i, lmbd in enumerate(landat):
    ridge = Ridge(alpha= lmbd)
    ridge.fit(X.array.reshape(-1, 1), Y)

    ridge_koef = ridge.coef_
    ridge_pikprer = ridge.intercept_
    Y_plt = ridge.predict(pd.DataFrame(X_plt))

    plt.plot(X_plt, Y_plt)
    plt.ylim(ymin=0, ymax=9)
    plt.xlim(xmin=0, xmax=6)
    plt.text(X_plt[-1], Y_plt[-1], f"y = {ridge_koef[0]:.2f} * X + {ridge_pikprer:.2f} për λ ose α = {lmbd}", fontsize=12)

