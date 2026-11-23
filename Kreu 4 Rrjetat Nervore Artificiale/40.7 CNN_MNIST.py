# Importini i bibliotekave
import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as k
from keras.models import load_model

import psutil

# Konfigurimi i CPU 
cores = psutil.cpu_count(logical=False)
threads = psutil.cpu_count(logical=True)
# tf.config.threading.set_intra_op_parallelism_threads(cores)
# tf.config.threading.set_inter_op_parallelism_threads(threads)

# Ngarkimi i te dhenave
from keras.datasets import mnist   

# Ndarja e te dhenave
(X_train , y_train), (X_test, y_test)=mnist.load_data()

# Kombinimi i te gjitha imazheve dhe etiketave 
X = np.concatenate([X_train, X_test], axis=0)
y = np.concatenate([y_train, y_test], axis=0)

# Marrja e nenbashkesise me 10,000 imazhe, perdorimi i 
# shtresezimit qe numrat nga (0-9) te jene te balancuar
X_subset, _, y_subset, _ = train_test_split(X, y, train_size = 10000, stratify = y, random_state = 42)

# Ndarja e 10,000 imazheve ne nje raport 8:2
x_train, x_test, y_train, y_test = train_test_split(X_subset, y_subset, train_size=0.8, stratify=y_subset, random_state=42)

# Verifikimi i permasave
print(f"Imazhet e stervitjes: {x_train.shape}, Etiketat e stervitjes: {y_train.shape}")  # (8000, 28, 28)
print(f"Imazhet e testimit:  {x_test.shape}, Etiketat e testimit:  {y_test.shape}")   # (2000, 28, 28)

# Riformatimi i te dhenave
X_train=x_train.reshape(x_train.shape[0],28,28,1)
X_test=x_test.reshape(x_test.shape[0],28,28,1)
input_shape=(28,28,1)

# Konvertimi i klases se vektoreve ne klasa binare 
num_classes=10
y_train=keras.utils.to_categorical(y_train, num_classes)
y_test=keras.utils.to_categorical(y_test,num_classes)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Normalizimi i te dhenave
X_train=X_train/255
X_test=X_test/255

print('X_train.shape', X_test.shape)
print(X_train.shape[0],'Te dhenat e stervitjes')
print(X_test.shape[0], 'Te dhenat e testimit')


# Ndertimi i dataseteve eficiente 
train_ds = (tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(10000)
    .cache().batch(128).prefetch(tf.data.AUTOTUNE))

test_ds = (tf.data.Dataset.from_tensor_slices((X_test, y_test))
    .batch(128).prefetch(tf.data.AUTOTUNE)) 

# Krijimi i Modelit
model=Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))


# Kompilimi i modelit
model.compile(loss=keras.losses.categorical_crossentropy,optimizer = keras.optimizers.Adadelta(), metrics=['accuracy'])

# Stervitja e Modelit
model.fit(train_ds,validation_data=test_ds,epochs=20,verbose=1)
print("Modeli u stërvit me sukses")

# Vleresimi i Modelit
score=model.evaluate(X_test,y_test, verbose = 2)
print('Humbja e testimit:', score[0]),
print('Saktesia e testimit:', score[1])

# Ruajtja e Modelit,
model.save('mnist.model.h5')
print('Ruajtja e modelit si mnist.h5')

# Parashikimi 
new_model=load_model('mnist.model.h5')

prediction=new_model.predict(X_test)
print(prediction)

for i in range(6):
    print(np.argmax(prediction[i]))

plt.imshow((tf.squeeze(X_test[5])))
plt.show()

# parashikimi mbi pese imazhet e para
pred=model.predict(X_test[:5])
# Shfaqja e parashikimit te modelit tone 
print(np.argmax(pred, axis = 1))
print(y_test[:5])

for i in range(10,16):
    first_img = X_test[i]
    first_img = np.array(first_img, dtype = 'float')
    pixels = first_img.reshape((28, 28))
    plt.imshow(pixels, cmap ='gray')
    plt.show()
