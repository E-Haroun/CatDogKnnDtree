# -*- coding: utf-8 -*-
"""
Created on Sun May 16 20:40:25 2021

@author: harou
"""



import numpy as np ;import pandas as pd

from skimage import io; 
from skimage.transform import resize;
from skimage.color import rgb2gray;
import matplotlib.pyplot as plt;

#train data
x_train=[];y_train=[]
for i in range(1,2001):
    cat=rgb2gray(resize(io.imread('../input/training_set/training_set/cats/cat.{}.jpg'.format(i)), (200,200)))
    x_train.append(cat); y_train.append(0)
for i in range(1,2001):
    dog=rgb2gray(resize(io.imread('../input/training_set/training_set/dogs/dog.{}.jpg'.format(i)), (200,200)))
    x_train.append(dog); y_train.append(1)
x_train,y_train=np.asarray(x_train),np.asarray(y_train)


#test data
x_test=[];y_test=[]
for i in range(4001,5001):
    cat=rgb2gray(resize(io.imread('../input/test_set/test_set/cats/cat.{}.jpg'.format(i)), (200,200)))
    x_test.append(cat); y_test.append(0)
for i in range(4001,5001):
    dog=rgb2gray(resize(io.imread('../input/test_set/test_set/dogs/dog.{}.jpg'.format(i)), (200,200)))
    x_test.append(dog); y_test.append(1)
x_test,y_test=np.asarray(x_test),np.asarray(y_test)

print('x_test shape : ',x_test.shape,'y_test shape : ',y_test.shape)



#reshape from data 3d to 2d because sklearn accepts just 2d 
x_train=x_train.reshape((x_train.shape[0],x_train.shape[1]*x_train.shape[2]))
x_test=x_test.reshape((x_test.shape[0],x_test.shape[1]*x_test.shape[2]))


# test of new hyperparameter on Knn optimized
from sklearn.neighbors import KNeighborsClassifier

knn =  KNeighborsClassifier(algorithm='kd_tree', leaf_size=30, metric='manhattan',
           metric_params=None, n_jobs=-1, n_neighbors=19, p=2,
           weights='uniform')
knn.fit(x_train,y_train)
print("Score for K = 19 :")
print("Score Train : ")
print(knn.score(x_train, y_train))
print("Score Test : ")
print(knn.score(x_test, y_test))


# test of new hyperparameter on D-Tree optimized

from sklearn.tree import DecisionTreeClassifier
dt =  DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best')
dt.fit(x_train,y_train)
print("Score Train : ")
print(dt.score(x_train, y_train))
print("Score Test : ")
print(dt.score(x_test, y_test))

