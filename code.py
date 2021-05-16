# -*- coding: utf-8 -*-
"""
Created on Sat May 15 00:20:16 2021

@author: harou
"""

#=============> Chargement de bibliotheques.
import numpy as np ; import pandas as pd
from skimage import io; cat1 = io.imread(r'C:\Users\harou\Desktop\CatsDogs\cat1.jpg')
from skimage.transform import resize; cat1_ = resize(cat1, (200,200,3))
from skimage.color import rgb2gray
cat1_gs = rgb2gray(cat1_)
import matplotlib.pyplot as plt
#### test du code de slide Tp:29
# fig=plt.figure()
# columns = 3; rows = 1
# fig.add_subplot(rows, columns, 1)
# plt.imshow(cat1)
# fig.add_subplot(rows, columns, 2) 
# plt.imshow(cat1_)
# fig.add_subplot(rows, columns, 3)
# plt.imshow(cat1_gs); plt.show()

#############################################
# L'ensemble des données d'apprentissage contient 4000 photos de cat et 4000 photos de dog
# L'ensemble des données de test contient 1000 photos de cat et 1000 de dog
#(indice dommence de 4001 pour les images de test)

#########################___ la base d'apprentissage ___###################
#=>>>>>> recuperer les donnée (images cats/dogs) pour train
x_train = []
y_train = []
for i in range(1,2001):
    #
    cat = rgb2gray(resize(io.imread('../CatsDogs/training_set/training_set/cats/cat.{}.jpg'.format(i)), (200,200)))
    x_train.append(cat)
    y_train.append(0)
for i in range(1,2001):
    #
    dog = rgb2gray(resize(io.imread('../CatsDogs/training_set/training_set/dogs/dog.{}.jpg'.format(i)), (200,200)))
    x_train.append(dog)
    y_train.append(1)
x_train, y_train = np.asarray(x_train), np.asarray(y_train)
print('x_train shape: ', x_train.shape, 'y_train shape: ', y_train.shape)

#########################___ la base de test ___###################
#=>>>>>> recuperer les donnée (images cats/dogs) pour test
x_test = []
y_test = []
#
for i in range(4001, 5001):
    cat = rgb2gray(resize(io.imread('../CatsDogs/test_set/test_set/cats/cat.{}.jpg'.format(i)), (200,200)))
    x_test.append(cat)
    y_test.append(0)
for i in range(4002, 5001):
    dog = rgb2gray(resize(io.imread( '../CatsDogs/test_set/test_set/dogs/dog.{}.jpg'.format(i)), (200, 200)))
    x_test.append(dog)
    y_test.append(1)
x_test, y_test = np.asarray(x_test), np.asarray(y_test)
print('x_test shape: ', x_test.shape, 'y_test shape: ', y_test.shape)

############################################
# Bien comprendre le fonctionnement de knn et l'importance de la valeur K 
# 
from numpy import *
distances=[0.5,8,1.2,4,5.6,2,1.25,3.01,1.05,0.36]
k=3
L=np.argsort(distances)
print('Les positions des '+str(k)+' valeurs minimales:', L[:k])
Y1=array([0, 0, 0,1,0,0,1,0,0, 1, 0, 0,0,0,1])
counts1 = np.bincount(Y1); print('np.bincount(Y1): ', counts1)
print('np.argmax(counts1)=', np.argmax(counts1))
Y2=array([0, 1, 1,1,1,1,1,0,1, 1, 1, 1,1,0,1])
counts2 = np.bincount(Y2); print('np.bincount(Y2): ',counts2)
print('np.argmax(counts2)=', np.argmax(counts2))
############################################

# to find the nearest training image to the i'th test image
def predict(X, k):
    distances=[]
    for i in range(0, len(x_train)):
        distances+= [np. sum(np.abs(x_train[i] - X))]
    #print(distances) #array of 4000 dist
    min_indexs = np.argsort(distances)[:k]
    #print(min_indexs)
    y_ = y_train[min_indexs] ; counts = np.bincount(y_)
    #print(y_)
    if np.argmax(counts)==0:return('cat')
    else:return ('dog')
    #print(np.argmax(counts))
############################################

# tester pour deux images 
numeros_images_a_predire=[4,1089 ] #914 #1129
fig=plt. figure()
predictions=[]
columns = 2
rows = 1
i=1
for num in numeros_images_a_predire:
    predictions+=[predict(x_test[num],3)]
    fig.add_subplot(rows, columns, i)
    plt.imshow(x_test[num])
    i+=1
plt.show()
print(predictions)













