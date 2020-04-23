# -*- coding: utf-8 -*-
"""
@author: Aun
"""
import numpy as np
import pandas as pd
from classifiers.knn import NearestNeighbour 

def train_test_split(data, train_split = 0.8, seed = 21):
    
    # split into train and test
    train_data = data.sample(frac =train_split , random_state = seed)
    test_data  = data[~data.index.isin(train_data.index)]
    
    # get x (features) and y (labels)
    Xtr = train_data[attributes[:-1]].values
    ytr = train_data[attributes[-1]].values
    Xte = test_data[attributes[:-1]].values
    yte = test_data[attributes[-1]].values
    
    return Xtr,ytr,Xte,yte




if __name__ == '__main__':
    
    # load data
    path = "C:/Users/Owner/Documents/mlscratch/data/iris/"
    file_name = "iris.data"

    # create dataframe
    attributes = ['Sepal Length', 'Sepal Width','Petal Length', 'Petal Width', 'Classes']
    df = pd.read_csv(path+file_name, sep=',', names = attributes)
    
    # split into train and test set
    Xtr,ytr,Xte,yte = train_test_split(df) 
    
    # clasification
    knn = NearestNeighbour(k = 5, dist='l2')
    knn.fit(Xtr,ytr)
    predictions = knn.predict(Xte)

    accuracy =np.sum(yte== predictions)/ len(predictions)
    
    print("% Accuracy : \t", accuracy*100)
    pred = pd.DataFrame()
    pred['Actual']     = yte
    pred['Prediction'] = predictions
    
    print(pred) 
    

# plot the train and test set
#
#groups_train = train_df.groupby('Classes')
#groups_test  = test_df.groupby('Classes')

#fig, axs = plt.subplots(2)
#axs[0].margins (0.05)
#axs[1].margins (0.05)
#for name, group in groups_train:
#    axs[0].scatter(group['Sepal Length'], group['Sepal Width'], marker ='o', label =name)
#
#for name, group in groups_test:    
#    axs[1].scatter(group['Sepal Length'], group['Sepal Width'], marker ='o', label =name)
#    
#
#axs[0].legend()
#axs[1].legend()
#axs[0].set_title('Train')
#axs[1].set_title('Test')
#plt.setp(axs[0], xlabel='Sepal Length')
#plt.setp(axs[0], ylabel='Sepal Width')
#plt.setp(axs[1], xlabel='Sepal Length')
#plt.setp(axs[1], ylabel='Sepal Width')
#plt.show()

# train and test



 
#predicted = pd.DataFrame(data = test_df[])

#fig, axs = plt.subplots(2)
#axs[0].margins (0.05)
#axs[1].margins (0.05)
#for name, group in groups_train:
#    axs[0].scatter(group['Sepal Length'], group['Sepal Width'], marker ='o', label =name)
#
#for name, group in groups_test:    
#    axs[1].scatter(group['Sepal Length'], group['Sepal Width'], marker ='o', label =name)
#    
#
#axs[0].legend()
#axs[1].legend()
#axs[0].set_title('Train')
#axs[1].set_title('Test')
#plt.setp(axs[0], xlabel='Sepal Length')
#plt.setp(axs[0], ylabel='Sepal Width')
#plt.setp(axs[1], xlabel='Sepal Length')
#plt.setp(axs[1], ylabel='Sepal Width')
#plt.show()