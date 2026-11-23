import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay 

iris = load_iris()
X = iris.data        
y = iris.target      

y_encoded = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

model = Sequential([Dense(8, input_shape=(4,), activation='relu'),   
                    Dense(3, activation='softmax') ])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


historia = model.fit(X_train, y_train, epochs=100, batch_size=8, validation_split=0.2, verbose=0)

plt.figure(figsize=(8,5))
plt.plot(historia.history['loss'], label='Humbja e stervitjes',
         linestyle='-',           # Solid line
    linewidth=2,
    color='black'            # Great for black & white printing
    )
plt.plot(historia.history['val_loss'], label='Humbja e miratimit',
         linestyle='--',          # Dashed line
         linewidth=2,
         color='gray'             # A lighter contrast for the second line
         )
plt.xlabel('Etapat',fontsize=14)
plt.ylabel('Humbja',fontsize=14)
plt.title('Humbja e stervitjes dhe miratimit',fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()


zgjedhja = np.array([[5.1, 3.5, 1.4, 0.2]])  
parashikimi = model.predict(zgjedhja)
klasa_prsh = np.argmax(parashikimi)

print("\nProbabilitetet e parashikuar (Softmax):", parashikimi)
print("Klasa e parashikuar:", iris.target_names[klasa_prsh])


y_predict = model.predict(X_test)

# Gjetja e pozicionit te '1' ne cdo rresht
y_test2 = np.argmax(y_test.round(), axis=1)
y_predict2 = np.argmax(y_predict.round(), axis=1)

# Ndertimi i matrices se qartesise y_predict
cm = confusion_matrix(y_test2, y_predict2)
cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [0, 1]) 
cm_display.plot(cmap=plt.cm.Blues)
plt.figure(figsize=(8,5))
plt.title('Matrica e qartësimit')
plt.xlabel('Etiketa e Parashikuar')
plt.ylabel('Etiketa e Vertete')
plt.show()


from sklearn.metrics import classification_report

# Perftimi i raportit te klasifikimit
class_report = classification_report(y_test2, y_predict2)
print("\nRaporti i klasifikimit:\n", class_report)
