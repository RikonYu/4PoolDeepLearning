import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Input,Concatenate,BatchNormalization,UpSampling2D,Layer
from keras.layers import Reshape,Dense, Dropout, Embedding, LSTM,Flatten,Conv2D,MaxPooling2D,Conv2DTranspose
from keras.optimizers import Adam
import numpy
import os
def get_pred(action,target,initial):
    k=initial
    action=numpy.array(action)
    #print(action)
    for i in range(len(action[0])):
        #print(action[0][i],action[1][i],action[2][i],target)
        k[i,action[i][0],action[i][1],action[i][2]]=target[i]
    #k[0,action[0],action[1],action[2]]=target
    return k

class MiningNet:
    def __init__(self):
        if(os.path.isfile('MiningNet.h5')):
            self.model=load_model('MiningNet.h5')
        else:
            self.model=Sequential()
            self.model.add(Conv2D(96,(5,5),activation='relu',padding='same',input_shape=(360,360,4),kernel_initializer='zeros',bias_initializer='zeros'))
            #self.model.add(Conv2D(96,(3,3),activation='sigmoid',padding='same',kernel_initializer='zeros',bias_initializer='zeros'))
            #self.model.add(Conv2D(96,(3,3),activation='relu',padding='same'))
            self.model.add(MaxPooling2D((2,2)))
            #self.model.add(Conv2D(96,(3,3),activation='sigmoid',padding='same',kernel_initializer='zeros',bias_initializer='zeros'))
            #self.model.add(MaxPooling2D((2,2)))
            #self.model.add(Conv2D(96,(3,3),activation='relu',padding='same'))
            #self.model.add(MaxPooling2D((2,2)))
            #self.model.add(Conv2D(96,(3,3),activation='relu',padding='same'))
            #self.model.add(MaxPooling2D((2,2)))
            #self.model.add(UpSampling2D((2,2)))
            #self.model.add(Conv2DTranspose(96,(3,3),padding='same',activation='sigmoid',kernel_initializer='zeros',bias_initializer='zeros'))
            self.model.add(UpSampling2D((2,2)))
            #self.model.add(Conv2DTranspose(96,(3,3),padding='same',activation='sigmoid',kernel_initializer='zeros',bias_initializer='zeros'))
            self.model.add(Conv2DTranspose(4,(5,5),padding='same',activation='relu',kernel_initializer='zeros',bias_initializer='zeros'))
            opt=Adam(lr=0.1)
            self.model.compile(optimizer=opt,loss='MSE')
        self.experience=[]
        #self.model.add()
    def save_model(self):
        self.model.save('MiningNet.h5')
    def set_weights(self,x):
        self.model.set_weights(x)
    def get_weights(self):
        return self.model.get_weights()
    def predict(self,X):
        return self.model.predict(X)
    def predict_masked(self,X,mask):
        return self.predict(numpy.reshape(X,[1,360,360,4]))[0]*mask
    def mask_pos(self,X,mask):
        ans=self.predict(numpy.reshape(X,[1,360,360,4]))[0]
        #print(ans,mask)
        ans=numpy.argmax(numpy.where(mask,ans,-123456789))
        #print(ans)
        #print(ans,numpy.unravel_index(ans,(360,360,4)))
        return numpy.unravel_index(ans,(360,360,4))
    def train(self,X,Y):
        print(X.shape,Y.shape)
        self.model.fit(X,Y)
    def train_experience(self,X):
        states,actions,targets=X
        labels=get_pred(actions,targets,self.predict(numpy.array(states)))
        self.train(numpy.array(states),numpy.array(labels))
        
