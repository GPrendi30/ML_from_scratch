import numpy as np

class LinearRegression:
    
    def __init__(self, lr=0.001, n_iters=1000):
        '''
        lr => learning rate
        n_iters => epochs
        weights => coefficients
        bias => the offset
        '''
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        '''
        X = training_samples
        y = labels
        
        invovles training and gradient descent
        '''
        
        # init parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # gradient descent
        for _ in range(self.n_iters):
            # predicting the values
            y_pred = np.dot(X, self.weights) + self.bias

            # calculating gradient
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # updating the rules
            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.bias * db
    
    def predict(self, X):
        '''
        X = test_samples
        returns sum(w * x) + b
        '''
        return np.dot(X, self.weights) + self.bias


def MSE(y_pred, y_test):
    '''
    Median Squared Error
    
    mse = 1/N * sum((y_pred - y_label) ** 2)
    '''
    return np.mean((y_pred - y_test) **2)