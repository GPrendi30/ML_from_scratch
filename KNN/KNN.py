import numpy as np
from collections import Counter

def euclidian_distance(x1, x2):
    # sqrt(sum(x2 - x1)) ^ 2
    return np.sqrt(np.sum(x2 - x1) **2)
    
    
#KNN Algorithm
#K nearest Neighbour
# We calculate the euclidian distances of each two points
# we predict the nearest neighbours
# We get the most voted class from our list of neighbors
# We return a lit of predictions for each point

class KNN:
    
    def __init__(self, k=3):
        self.k = k
    
    
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
    

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return predicted_labels
        
    
    def _predict(self, x):
        # distance
        distances = [euclidian_distance(x, x_train) for x_train in self.X_train]
        # get k nearest samples
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        #most voted class
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
        
        
