import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

from logistic_regression import LogisticRegression
db = datasets.load_breast_cancer()
X, y = db.data, db.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

#fig = plt.figure(figsize=(8,6))
#plt.scatter(X[:, 0], y, color='b', marker='o', s=30)
#plt.show()

def accuracy(y_valid, y_pred):
    accuracy = np.sum(y_pred == y_valid) / len(y_valid)
    return accuracy

r = LogisticRegression(lr=0.5, n_iter=10000)
r.fit(X_train, y_train)

preds = r.predict(X_test)

print('Acurracy: ', accuracy(y_test, preds))
