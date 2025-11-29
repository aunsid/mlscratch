"""
@author: Aun
"""
import numpy as np


class NearestNeighbour():
    
    
    def __init__(self,k = 5, dist = 'l1'):
        
        self.k    = k 
        self.dist_metric  = dist
        
    
    def fit(self, X, y):
        self.Xtr = X
        self.ytr = y
        
        
    def predict(self,X):
        
        # get number of sample in test
        num_test = X.shape[0]
        num_train = self.Xtr.shape[0]
        # create placeholder for predictions, in X
        # size num_test and dtype same as that of ytr
        Ypred  = np.zeros(num_test, dtype = self.ytr.dtype)
        
        # distance is num_test x num_train
        # where each row is the sample from the test and how close it is to the train sample
        distances = np.zeros((num_test, num_train))
        
        # iterate over the test samples
        for i in range(num_test):
            
            # find the distance of X[i] with each in Xtr
            if self.dist_metric == 'l1':
                distances[i,:]= self.l1_distance(self.Xtr, X[i,:])
            elif self.dist_metric == 'l2':
                distances[i,:]= self.l2_distance(self.Xtr, X[i,:])
            else:
                raise Exception('Distance Metric not defined')
            
            # find the n smallest indices
            closest = np.argsort(distances[i,:])[:self.k]
            
            
#            Debug
#            print("distances", list(distances[i,:]))
#            print("closest",   closest)
            
            
            # get mode of these indices for voting the label and store in Ypred
            y_hat = [self.ytr[i] for i in closest] 
            
#            Debug            
#            print ("All closest", y_hat)
            
            y_hat = max(set(y_hat), key=y_hat.count)
            
#            Debug
#            print("Mode y_hat", y_hat)
            
        
            Ypred[i] = y_hat
            
#            Debug
#            print(y_hat)
            
        
        return Ypred
    
    @staticmethod    
    def l1_distance(x_train, x_test):
        return np.sum(np.abs(x_train - x_test), axis = 1)

    @staticmethod
    def l2_distance(x_train, x_test):
        return np.sqrt(np.sum(np.square(x_train- x_test), axis = 1))






    
    
