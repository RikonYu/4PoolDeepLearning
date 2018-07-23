import util64
import pickle
import time
import numpy
import os
import ReplayBuffer
from keras import backend as KB
import tensorflow as tf
from ClassConstr import getUnitClass
import threading
import struct
import matplotlib.pyplot as plt
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Concatenate, BatchNormalization, UpSampling2D, Layer, Add
from keras.layers import Reshape, Dense, Dropout, Embedding, LSTM, Flatten, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras import backend as KTF
from consts import WINDOW_SIZE
from drawer import plots
N=21
model=Sequential()
model.add(Conv2DTranspose(16,(3,3), input_shape=[N, N,1], activation='relu', padding='same'))
model.add(Conv2DTranspose(16,(3,3), activation='relu', padding='same'))
model.add(Conv2DTranspose(16,(3,3), activation='relu', padding='same'))
model.add(Conv2DTranspose(16,(3,3), activation='relu', padding='same'))
model.add(Conv2DTranspose(16,(3,3), activation='relu', padding='same'))
model.add(Conv2DTranspose(16,(3,3), activation='relu', padding='same'))
model.add(Conv2DTranspose(16,(3,3), activation='relu', padding='same'))
model.add(Conv2DTranspose(16,(3,3), activation='relu', padding='same'))


model.add(Conv2D(1,(1,1), padding='same'))
model.compile(optimizer='rmsprop', loss='mse')
X=numpy.zeros([1,N, N,1])
X[0,2,2,0]=1.0
Y=numpy.zeros([1,N, N,1])
pos1=[2,2]
pos2=[14,15]
for ind, _ in numpy.ndenumerate(Y[0,:, :, 0]):
    Y[0,ind[0], ind[1], 0] = numpy.linalg.norm(numpy.array(ind) - pos1)  +  numpy.linalg.norm(numpy.array(ind) - pos2)

for i in range(5000):
    model.fit(X,Y,verbose=0)
print(model.evaluate(X,Y))
plots(Y[0])
plots(model.predict(X)[0])