from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


mnist = fetch_openml('mnist_784')

X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=1/7.0, random_state=0)

scaler = StandardScaler()

# Stervitja
scaler.fit(X_train)    

# Zbatimi i transformimit
train_img = scaler.transform(X_train) 
test_img = scaler.transform(X_test)


# Krijimi i nje objekti te klases PCA
pca = PCA(.95)   
pca.fit(X_train)

pca.n_components_ 

train_img = pca.transform(X_train)
test_img = pca.transform(X_test)

lower_dim_data = pca.fit_transform(mnist.data)

pca.n_components_

from sklearn.linear_model import LogisticRegression

Reg_log = LogisticRegression(solver = 'lbfgs')
Reg_log.fit(train_img, y_train)
# Parashikimi per nje vëzhgim dhe shumë vëzhgime (image)
Reg_log.predict(test_img[0].reshape(1,-1))
Reg_log.predict(test_img[0:10])

Reg_log.score(test_img, y_test)

perafrim = pca.inverse_transform(lower_dim_data)
import matplotlib.pyplot as plt
plt.figure(figsize=(8,4))

plt.subplot(1, 2, 1)     # Imazhi origjinal 
plt.imshow(mnist.data.iloc[1].to_numpy().reshape(28,28),
              cmap = plt.cm.gray, interpolation='nearest',
              clim=(0, 255))
plt.xlabel('784 komponente', fontsize = 14)
plt.title('Imazhi origjinal', fontsize = 20)

plt.subplot(1, 2, 2);
plt.imshow(perafrim[1].reshape(28, 28), cmap = plt.cm.gray,
              interpolation='nearest', clim=(0, 255))
plt.xlabel('330 komponente', fontsize = 14)
plt.title('95% e Variances se shpjeguar', fontsize = 20)
plt.show()
