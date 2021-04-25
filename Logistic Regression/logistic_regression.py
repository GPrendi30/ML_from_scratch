import numpy as np

class LogisticRegression:
    
    def __init__(self, lr=0.001, n_iter=1000):
        self.lr = lr 
        self.n_iter = n_iter
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        '''
        X -> numpy array
        '''
        # trainingf
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # gradient descent
        for _ in range(self.n_iter):
            # predicting the values
            linear_model = np.dot(X, self.weights) + self.bias
            
            # turns it in a number from 0 to 1
            y_pred = self._sigmoid(linear_model)
            
            # calculating gradient
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
             
            # updating the rules
            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.bias * db
        
    
    def predict(self, X):
        '''
        X 
        '''
        linear_model = np.dot(X, self.weights) + self.bias
            
        # turns it in a number from 0 to 1
        y_pred = self._sigmoid(linear_model)
        
        y_predicted_class = [1 if i > 0.5 else 0 for i in y_pred]
        return y_predicted_class

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))