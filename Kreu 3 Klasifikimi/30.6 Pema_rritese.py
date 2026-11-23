# pip install river graphviz   
from river import evaluate
from river import metrics
from river import tree
from river import datasets

dataset = datasets.Phishing()
model = tree.HoeffdingTreeClassifier(grace_period=50)
metric = metrics.Accuracy()

for x, y in dataset:
     # Parashikimi
    y_pred = model.predict_one(x) 
    metric.update(y, y_pred)
    
    model.learn_one(x, y)

print(f'Saktesia finale: {metric.get()}')    # Printimi

import os
os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin'
model.draw()
