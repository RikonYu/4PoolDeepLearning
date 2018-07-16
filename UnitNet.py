
import keras
import tensorflow as tf
from keras.models import Sequential, Model, load_model
from keras.layers import Input,Concatenate,BatchNormalization,UpSampling2D,Layer,Add
from keras.layers import Reshape,Dense, Dropout, Embedding, LSTM,Flatten,Conv2D,MaxPooling2D,Conv2DTranspose
from keras.optimizers import Adam
from keras import backend as KTF
from consts import WINDOW_SIZE
import numpy
class UnitNet:
    _in_channel=1
    _out_channel=1
    def __init__(self,loading=False):
        self.model=None
    def set_weights(self,weights):
        with self.session.as_default():
            with self.graph.as_default():
                self.model.set_weights(weights)
    def get_weights(self):
        with self.session.as_default():
            with self.graph.as_default():
                return self.model.get_weights()
    def save(self):
        with self.session.as_default():
            with self.graph.as_default():
                self.model.save('DroneNet.h5')
    def predict_all(self,X):
        with self.session.as_default():
            with self.graph.as_default():
                return self.model.predict(numpy.reshape(X,[-1,WINDOW_SIZE,WINDOW_SIZE,self._in_channel]))
    def predict_ans(self,X):
        ans=numpy.argmax(self.predict_all([X]))
        return numpy.unravel_index(ans,(WINDOW_SIZE,WINDOW_SIZE,self._out_channel))
    def predict_max(self,X):
        return numpy.amax(self.predict_all(X))
    def predict_all_masked(self,X,mask):
        Y=self.predict_all(X)
        return numpy.where(mask,Y,-numpy.inf)
    def predict_ans_masked(self,X,mask, want_val=False):
        allval=self.predict_all(X)[0]
        ini,inj,ink=numpy.nonzero(mask)
        pos=numpy.argmax(allval[ini,inj,ink])
        ans=[ini[pos],inj[pos],ink[pos]]
        if(want_val):
            #ans=numpy.unravel_index(ans,(WINDOW_SIZE,WINDOW_SIZE,self._out_channel))
            return (ans, allval[tuple(ans)])
        #return numpy.unravel_index(ans,(WINDOW_SIZE,WINDOW_SIZE,self._out_channel))
        return ans
    def sample_ans_masked(self, X, mask):
        allval=self.predict_all(X)
        X,Y,Z=numpy.nonzero(mask)
        total=numpy.sum(numpy.exp(allval[0]*mask))
        ans=numpy.random.choice(len(X),p=numpy.exp(allval[X,Y,Z])/total)
        return [X[ans],Y[ans],Z[ans]]
    def train(self,X,Y):
        with self.session.as_default():
            with self.graph.as_default():
                return self.model.fit(X,Y)
    def evaluate(self,X,Y):
        with self.session.as_default():
            with self.graph.as_default():
                return self.model.evaluate(X,Y,verbose=0)
    @staticmethod
    def msg2mask(disGame, msg):
        pass
    @staticmethod
    def msg2state(disGame, msg):
        pass
    @staticmethod
    def y2state(ind):
        pass

class NeuralMap:
    def __init__(self, _in_channel, mapHeight, mapWidth):
        self.board=numpy.zeros([mapHeight,mapWidth,_in_channel])
        self.build_global_read()
        self._in_channel=_in_channel
        self.build_write()
    def build_global_read(self):
        self.read_inp=Input(self.board.shape,dtype='float32')
        self.read_conv1=Conv2D(32, (3,3), padding='same', activation='relu')(self.read_inp)
        self.read_conv1=MaxPooling2D((2,2))(self.read_conv1)
        self.read_conv2=Conv2D(32,(3,3), padding='same', activation='relu')(self.read_conv1)
        self.read_conv2 = MaxPooling2D((2, 2))(self.read_conv2)
