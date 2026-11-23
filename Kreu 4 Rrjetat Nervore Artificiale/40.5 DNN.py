# Importimi i bibliotekave te nevojshme
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # RREGULLIMI 1: Për shkallëzimin
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay 
import matplotlib.pyplot as plt

# Leximi i bashkesise se te dhenave
diabetes = pd.read_csv('diabetes.csv')
diabetes['Outcome'] = diabetes['Outcome'].replace({'pos': 1, 'neg': 0})
diabetes['Outcome'] = pd.to_numeric(diabetes['Outcome'], errors='coerce')
# Kontrolli per vlerat qe mungojne, ndarja e ndryshoreve e varur dhe e pavarur
X = diabetes.iloc[:, 0:-1]
Y = diabetes.iloc[:, -1]

# Ndarja per stervitje dhe testim
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1234, stratify=Y)

# RREGULLIMI 2: Shkallëzimi i të dhënave (Standardizimi)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Percaktimi i modelit sekuencial 
model = Sequential()

# Ndryshuar nga input_shape=(8,) ne input_dim=8 për të shmangur gabimet e versioneve
model.add(Dense(24, input_dim=8, activation='relu', kernel_initializer='RandomNormal'))
model.add(Dense(124, activation='relu', kernel_initializer='RandomNormal'))
model.add(Dense(1, activation='sigmoid', kernel_initializer='RandomNormal'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Ekzekutimi i modelit
model.fit(X_train, Y_train, epochs=160, batch_size=10)

# Gjetja e saktesise se modelit
loss, accuracy_test = model.evaluate(X_test, Y_test)
print(f"\nSaktësia në testim (Accuracy): {accuracy_test * 100:.2f}%")

# --- SHFAQJA E MATRICËS SË KONFUZIONIT ---
# 1. Bëjmë parashikimet me modelin e stërvitur (kthehen vlera midis 0 dhe 1)
y_pred_prob = model.predict(X_test)
# 2. Kthejmë probabilitetet në 0 ose 1 (nëse > 0.5 është 1, ndryshe 0)
y_pred = (y_pred_prob > 0.5).astype(int)

# 3. Krijojmë dhe shfaqim Matricën e Konfuzionit
cm = confusion_matrix(Y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Jo Diabetik (0)', 'Diabetik (1)'])

disp.plot(cmap=plt.cm.Blues)
plt.title("Matrica e qartesimit - Parashikimi i Diabetit")
plt.show()
