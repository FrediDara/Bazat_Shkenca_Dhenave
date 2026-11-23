
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
# Krijimi i te dhenave
reale = ['Mace', 'Qen', 'Lepur', 'Mace', 'Mace', 'Lepur']
parashikim = ['Qen', 'Qen', 'Lepur', 'Qen', 'Qen', 'Lepur']
etiketa = ['Mace', 'Qen', 'Lepur']

c_matrix =confusion_matrix(reale, parashikim, labels= etiketa)
cm_display =ConfusionMatrixDisplay(confusion_matrix = c_matrix, display_labels = etiketa)
cm_display.plot(cmap=plt.cm.Blues)
plt.title('Matrica e qartesimit')
plt.xlabel('Etiketa e Parashikuar')
plt.ylabel('Etiketa e vertete')
plt.plot()

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

saktesia = accuracy_score(reale, parashikim)
print(saktesia)

Perpikmeria = precision_score(reale, parashikim, average='macro')
print(Perpikmeria)

rikujtimi = recall_score(reale, parashikim, average='macro')
print(rikujtimi)

Vecantia = recall_score(reale, parashikim, pos_label=0, average='macro') 
print(Vecantia)

Vlera_f1= f1_score(reale, parashikim, average='macro')
print(Vlera_f1)
