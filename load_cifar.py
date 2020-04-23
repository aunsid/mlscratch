# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 19:32:25 2020

@author: Owner
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 19:32:25 2020

@author: Owner
"""

import pickle 
import numpy as np
import os


def unpickle(file):
    print('unpickle')
    data = {}
    with open(file,'rb') as fo:
        data = pickle.load(fo, encoding = 'latin1')
        x = data['data']
        y = data['labels']
#        x = x.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
#        x = np.array(y)
        return x,y
    


def data_batch_files(path):
    files = os.listdir(path)
    xs = []
    ys = []
   
#    print(files)
    for file in files:
        if 'data_batch' in file:
            print(file)
            X,Y = unpickle(path+file)
            xs.append(X)
            ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X,Y
    return Xtr, Ytr
    




if __name__ =='__main__':
    
    path  = "C:/Users/Owner/Documents/CS231n/cifar-10-batches-py/"
    X,Y = data_batch_files(path)
    X = X.reshape(-1, 3, 32, 32).transpose(0,2,3,1).astype("float")
    
    X[]