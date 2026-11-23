import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

breast_cancer = load_breast_cancer()
X = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
y = breast_cancer.target
X_train, X_test, y_train, y_test = train_test_split( X, y, random_state=0)

from sklearn.feature_selection import SequentialFeatureSelector as SFS

sfs = SFS(estimator=RandomForestClassifier(n_estimators=5, 
    random_state=0), n_features_to_select='auto', 
    tol=0.001, direction='forward', scoring='roc_auc', cv=3)
sfs = sfs.fit(X_train, y_train)

sfs.get_feature_names_out()

X_train_t = sfs.transform(X_train)
X_test_t = sfs.transform(X_test)
