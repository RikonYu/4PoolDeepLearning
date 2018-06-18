
import keras
import tensorflow as tf
from keras.models import Sequential, Model, load_model
from keras.layers import Input,Concatenate,BatchNormalization,UpSampling2D,Layer,Add
from keras.layers import Reshape,Dense, Dropout, Embedding, LSTM,Flatten,Conv2D,MaxPooling2D,Conv2DTranspose
from keras.optimizers import Adam
from keras import backend as KTF
import numpy
class UnitNet:
    def __init__(self,loading=False):
        self.model=None    
    def set_weights(self,weights):
        self.model.set_weights(weights)
    def get_weights(self):
        return self.model.get_weights()
    def save(self):
        self.model.save('DroneNet.h5')
    def predict_all(self,X):
        with self.session.as_default():
            with self.graph.as_default():
                return self.model.predict(numpy.reshape(X,[-1,360,360,18]))
    def predict_ans(self,X):
        ans=numpy.argmax(self.predict_all([X]))
        return numpy.unravel_index(ans,(360,360,6))
    def predict_max(self,X):
        return numpy.amax(self.predict_all(X),axis=(1,2,3))
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
    @staticmethod
    def msg2mask(disGame):
        pass
    @staticmethod
    def msg2state(disGame):
        pass
    @staticmethod
    def y2state(ind):
        pass
