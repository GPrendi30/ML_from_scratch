import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

#fig = plt.figure(figsize=(8,6))
#plt.scatter(X[:, 0], y, color='b', marker='o', s=30)
#plt.show()


from linear_regression import LinearRegression, MSE

# plotting points
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(20,14))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)

# # regressor model 2
r1 = LinearRegression(lr=0.001)
r1.fit(X_train, y_train)

# calculate the line
y_pred_line = r1.predict(X)

# plotting the first line
plt.plot(X, y_pred_line, color='black', linewidth=2, label='Prediction')


# regressor model 2
r2 = LinearRegression(lr=0.1)
r2.fit(X_train, y_train)

# calculate the line
y_pred_line2 = r2.predict(X)
# plotting the line
plt.plot(X, y_pred_line2, color='red', linewidth=2)

# showing the plot
#plt.show()

print('Error with lr 0.001:', MSE(r1.predict(X_test), y_test))
print('Error with lr 0.1:', MSE(r2.predict(X_test), y_test))

