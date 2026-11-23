
# Importimi i bibliotekave te nevojshme
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
 
# Ngarkimi i te dhenave te kancerit te gjirit 
cancer_data = load_breast_cancer()
X, y = cancer_data.data, cancer_data.target
  
# Ndarja e te dhenave ne te stervitjes dhe testimit
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Standardizimi i atributeve me qenderzim dhe normalizim
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
 
# Krijimi i nje objeki te klases MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(64, 32),
                    max_iter=1000, random_state=42) 
  
# Stervitja e model me te dhënat e stërvitjes 
mlp.fit(X_train, y_train)
 
# Parashikimi per bashkësinë e testimit 
y_pred = mlp.predict(X_test)
 
# Llogaritja e saktesise se modelit
saktesia = accuracy_score(y_test, y_pred)
print(f"Saktesia: {saktesia:.2f}")

# Perftimi i raportit te klasifikimit
class_report = classification_report(y_test, y_pred)
print("Raporti i klasifikimit:\n", class_report)

