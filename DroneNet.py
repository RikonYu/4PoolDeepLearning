
import keras
import tensorflow as tf
from keras.models import Sequential, Model, load_model
from keras.layers import Input,Concatenate,BatchNormalization,UpSampling2D,Layer,Add
from keras.layers import Reshape,Dense, Dropout, Embedding, LSTM,Flatten,Conv2D,MaxPooling2D,Conv2DTranspose
from keras.optimizers import Adam
from keras import backend as KTF
import numpy
import os,sys

class DroneNet:
    def __init__(self,loading=False):
        INP_CHANNEL=18
        OUT_CHANNEL=6
        self.session=KTF.get_session()
        self.graph=tf.get_default_graph()
        with self.session.as_default():
            with self.graph.as_default():       
                self.inp=Input((360,360,INP_CHANNEL),dtype='float32')
                self.conv1=Conv2D(64,(5,5),activation='relu',padding='same')(self.inp)
                self.conv2=Conv2D(64,(5,5),activation='relu',padding='same')(self.conv1)
                self.conv2=Add()([self.conv2,self.conv1])
                self.pool1=MaxPooling2D((2,2))(self.conv2)
                self.conv3=Conv2D(64,(5,5),activation='relu',padding='same')(self.pool1)
                self.conv3=Add()([self.conv3,self.pool1])
                self.conv4=Conv2D(64,(5,5),activation='relu',padding='same')(self.conv3)
                self.conv4=Add()([self.conv4,self.conv3])
                self.pool2=MaxPooling2D((2,2))(self.conv4)
                self.conv5=Conv2D(64,(5,5),activation='relu',padding='same')(self.pool2)
                self.conv5=Add()([self.conv5,self.pool2])
                self.conv6=Conv2D(64,(5,5),activation='relu',padding='same')(self.conv5)
                self.conv6=Add()([self.conv6,self.conv5])
                
                self.deconv1=Conv2DTranspose(64,(5,5),activation='relu',padding='same')(self.conv6)
                self.deconv1.set_shape([None,90,90,64])
                self.deconv2=Conv2DTranspose(64,(5,5),activation='relu',padding='same')(self.deconv1)
                self.deconv2=Add()([self.deconv2,self.deconv1])
                self.deconv2.set_shape([None,90,90,64])
                self.up1=UpSampling2D((2,2))(self.deconv2)
                self.deconv3=Conv2DTranspose(64,(5,5),activation='relu',padding='same')(Concatenate(axis=3)([self.up1,self.conv4]))
                self.deconv3.set_shape([None,180,180,64])
                self.deconv4=Conv2DTranspose(64,(5,5),activation='relu',padding='same')(self.deconv3)
                self.deconv4=Add()([self.deconv4,self.deconv3])
                self.deconv4.set_shape([None,180,180,64])
                self.up2=UpSampling2D((2,2))(self.deconv4)
                #print(self.deconv4.get_shape(),self.conv2.get_shape())
                self.deconv5=Conv2DTranspose(64,(5,5),activation='relu',padding='same')(Concatenate(axis=3)([self.up2,self.conv2]))
                #self.deconv5.set_shape([None,360,360,64])
                self.deconv6=Conv2DTranspose(OUT_CHANNEL,(5,5),activation='softmax',padding='same')(self.deconv5)
                #self.deconv6.set_shape([None,360,360,OUT_CHANNEL])
                self.model=Model(inputs=self.inp,outputs=self.deconv6)
                print(self.model.summary())
                opt=Adam(lr=0.001)
                self.model.compile(optimizer=opt,loss='categorical_crossentropy')
                if(os.path.isfile('DroneNet.h5') and loading==True):
                    self.model.load_weights("DroneNet.h5",by_name=True)
        
    def save(self):
        self.model.save_weights('DroneNet.h5',by_name=True)
    def predict_all(self,X):
        with self.session.as_default():
            with self.graph.as_default():
                return self.model.predict(numpy.reshape(X,[-1,360,360,18]))
    def predict_ans(self,X):
        ans=numpy.argmax(self.predict_all([X]))
        return numpy.unravel_index(ans,(360,360,6))
    def predict_max(self,X):
        return numpy.amax(self.predict_all(X))
    def predict_all_masked(self,X,mask):
        Y=self.predict_all(X)
        return numpy.where(mask,Y,-1234)
    def predict_ans_masked(self,X,mask):
        ans=numpy.argmax(self.predict_all_masked(X,mask))
        return numpy.unravel_index(ans,(360,360,6))
    def train(self,X,Y):
        with self.session.as_default():
            with self.graph.as_default():
                return self.model.fit(X,Y)
    def evaluate(self,X,Y):
        with self.session.as_default():
            with self.graph.as_default():
                return self.model.evaluate(X,Y,verbose=0)
    def train_experience(self,states,actions,rewards,next_states,discount):
        targets=rewards+discount*self.predict_max(next_states)
        
