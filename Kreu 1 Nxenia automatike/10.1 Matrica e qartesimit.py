import numpy as np


reale = np.random.binomial(1, 0.9, size = 1000)
parashikim = np.random.binomial(1, 0.9, size = 1000)


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

c_matrix = confusion_matrix(reale, parashikim) 
cm_display = ConfusionMatrixDisplay(confusion_matrix = c_matrix, display_labels = [0, 1]) 

import matplotlib.pyplot as plt 
cm_display.plot(cmap=plt.cm.Blues)

for text in cm_display.text_.ravel():
    text.set_fontsize(16) 

plt.title('Matrica e qartësimit', fontsize=20)
plt.xlabel('Etiketa e Parashikuar', fontsize=16)
plt.ylabel('Etiketa e verteta', fontsize=16)
plt.tick_params(axis='both', labelsize=12)  
plt.show()
plt.show()


from sklearn.metrics import accuracy_score
saktesia = accuracy_score(reale, parashikim)
print(saktesia)

from sklearn.metrics import precision_score
Perpik = precision_score(reale, parashikim)
print(Perpik)
from sklearn.metrics import recall_score

rikujtimi = recall_score(reale, parashikim)
print(rikujtimi)

Vecantia = recall_score(reale, parashikim, pos_label=0) 
print(Vecantia)


from sklearn.metrics import f1_score

Vlera_f1= f1_score(reale, parashikim)
print(Vlera_f1)


