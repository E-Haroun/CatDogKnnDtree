# -*- coding: utf-8 -*-
"""
Created on  Sat May 15 05:20:11 2021

@author: harou
"""

import numpy as np ;import pandas as pd
from skimage.color import rgb2gray;
import matplotlib.pyplot as plt;
from skimage.transform import resize;
from skimage import io; 



#get training data
x_train=[];y_train=[]   
for i in range(1,2001):
    cat=rgb2gray(resize(io.imread('training_set/training_set/cats\cat.{}.jpg'.format(i)), (200,200)))
    x_train.append(cat); y_train.append(0)
for i in range(1,2001):
    dog=rgb2gray(resize(io.imread('training_set/training_set/dogs\dog.{}.jpg'.format(i)), (200,200)))
    x_train.append(dog); y_train.append(1)
x_train,y_train=np.asarray(x_train),np.asarray(y_train)
print('x_test shape : ',x_train.shape,'y_test shape : ',y_train.shape)

#get testing data
x_test=[];y_test=[]
for i in range(4001,5001):
    cat=rgb2gray(resize(io.imread('test_set/test_set/cats\cat.{}.jpg'.format(i)), (200,200)))
    x_test.append(cat); y_test.append(0)
for i in range(4001,5001):
    dog=rgb2gray(resize(io.imread('test_set/test_set/dogs\dog.{}.jpg'.format(i)), (200,200)))
    x_test.append(dog); y_test.append(1)
x_test,y_test=np.asarray(x_test),np.asarray(y_test)

print('x_test shape : ',x_test.shape,'y_test shape : ',y_test.shape)


#Import Libraries for K_Neighbors Classifier
from sklearn.neighbors import KNeighborsClassifier

#==== Remodeler les données de 3D vers 2D car sklearn n'accepte que les 2D =====#
x_train=x_train.reshape((x_train.shape[0],x_train.shape[1]*x_train.shape[2]))
x_test=x_test.reshape((x_test.shape[0],x_test.shape[1]*x_test.shape[2]))

K = [3,5,7]
for j in K :
    knn = KNeighborsClassifier(n_neighbors= j,weights ='uniform', algorithm='auto')
    #pour weights on peut avoir ('uniform','distance')
    #pour algorithm on peut avoir ('auto','ball_tree','kd_tree','brute')
    knn.fit(x_train,y_train)
    print("Score for K = ",str(j))
    print("Score Train : ")
    print(knn.score(x_train, y_train))
    print("Test Train : ")
    print(knn.score(x_test, y_test))


#=========== implementer KNN avec GRIDSEARCH
from sklearn.model_selection import cross_val_score #
from sklearn.model_selection import GridSearchCV

k_range = np.arange(7,20,2)
print(k_range)

#p#param_grid = dict(n_neighbors=k_range)
#'algorithm': (' auto','ball_tree','kd_tree','brute'),'weights':('uniform','distance')
aram_grid = dict(n_neighbors=k_range)
param_grid={'n_neighbors':k_range,'metric':['euclidean','manhattan']}
print(param_grid)

grid = GridSearchCV(KNeighborsClassifier() ,param_grid ,cv=3 ,verbose =1 
                    ,n_jobs =-1 ,scoring='accuracy' ,return_train_score=False )
grid.fit(x_train, y_train)

print(grid.cv_results_['params'])
print(grid.cv_results_['mean_test_score'])

grid_mean_scores = grid.cv_results_['mean_test_score']
print(grid_mean_scores)

print('score for KNN : ' , grid.best_score_)
print('params : ' , grid.best_params_)
print('best : ' , grid.best_estimator_)

#=========== implementer D-Tree avec GRIDSEARCH

#Import Libraries
from sklearn.tree import DecisionTreeClassifier
#----------------------------------------------------
DecisionTreeClassifierModel = DecisionTreeClassifier()

param_grid={'criterion':('gini','entropy')}
print(param_grid)

grid = GridSearchCV(DecisionTreeClassifierModel, param_grid, cv=10, scoring='accuracy', return_train_score=False)
grid.fit(x_train, y_train)

print(grid.cv_results_['params'])
print(grid.cv_results_['mean_test_score'])

grid_mean_scores = grid.cv_results_['mean_test_score']
print(grid_mean_scores)

print('score using decision tree: ' , grid.best_score_)
print('params : ' , grid.best_params_)
print('best : ' , grid.best_estimator_)

