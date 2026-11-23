# Për krijimin e të dhënave dhe vlerësimin e modelit
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

# Për modelet dhe ansamblin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier

# Për llogaritjet matematike (mean dhe std)
from numpy import mean, std

# Për ndërtimin e grafikut (pyplot)
from matplotlib import pyplot


# Marrja e votimit te ansamblit te modeleve

def get_voting():
	models = list()              # percaktimi i modeleve baze
	models.append(('knn1', KNeighborsClassifier(n_neighbors=1)))
	models.append(('knn3', KNeighborsClassifier(n_neighbors=3)))
	models.append(('knn5', KNeighborsClassifier(n_neighbors=5)))
	models.append(('knn7', KNeighborsClassifier(n_neighbors=7)))
	models.append(('knn9', KNeighborsClassifier(n_neighbors=9)))
	ansambli = VotingClassifier(estimators=models, voting='hard')
	return ansambli             # kthimi i votimit te ansamblit


def get_models():
	models = dict()
	models['knn1'] = KNeighborsClassifier(n_neighbors=1)
	models['knn3'] = KNeighborsClassifier(n_neighbors=3)
	models['knn5'] = KNeighborsClassifier(n_neighbors=5)
	models['knn7'] = KNeighborsClassifier(n_neighbors=7)
	models['knn9'] = KNeighborsClassifier(n_neighbors=9)
	models['hard_voting'] = get_voting()
	return models


def evaluate_model(model, X, y):
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
	return scores

# Lexoni skedarin CSV 
def get_dataset():
    df = pd.read_csv('heart_cleveland_hungary.csv')
    X = df.drop(columns=['target']) 
    y = df['target']
    
    return X, y

X, y = get_dataset()      # percaktimi i te dhenave
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = get_models()
results, names = list(), list()
for name, model in models.items():     # vleresimi i modeleve dhe
	scores = evaluate_model(model, X_train, y_train)  # ruajtja e rezultateve
	results.append(scores)
	names.append(name)
	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
pyplot.boxplot(results, tick_labels=names, showmeans=True)   # labels
pyplot.show()        # Grafiku i performances se model 

ansambel = models.get("hard_voting")
ansambel.fit(X_train, y_train)
yhat = ansambel.predict(X_test.iloc[[10]])
print('Klasa e parashikuar:', yhat)

