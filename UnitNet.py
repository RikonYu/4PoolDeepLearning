
import keras
import tensorflow as tf
from keras.models import Sequential, Model, load_model
from keras.layers import Input,Concatenate,BatchNormalization,UpSampling2D,Layer,Add
from keras.layers import Reshape,Dense, Dropout, Embedding, LSTM,Flatten,Conv2D,MaxPooling2D,Conv2DTranspose
from keras.optimizers import Adam
from keras import backend as KTF
from consts import WINDOW_SIZE
import pickle
from util64 import conv_block
import util64
import numpy
class UnitNet:
    _in_channel=1
    _out_channel=1
    def __init__(self,loading=False, output_type='linear'):
        self.model=None
    def apply_gradient(self, gradient):
        with self.session.as_default():
            with self.graph.as_default():
                opt=tf.train.AdamOptimizer()
                for i in gradient:
                    opt.apply_gradients(i)
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
    def predict_max_masked(self,X,mask):
        k=self.predict_all(X)[0]
        ini,inj,ink=numpy.nonzero(mask)
        return numpy.amax(k[ini,inj,ink])
    def predict_ans_masked(self,X,mask, want_val=False):
        allval=self.predict_all(X).reshape([WINDOW_SIZE,WINDOW_SIZE,self._out_channel])
        #print(allval)
        ini,inj,ink=numpy.nonzero(mask)
        pos=numpy.argmax(allval[ini,inj,ink])
        ans=[ini[pos],inj[pos],ink[pos]]
        if(want_val):
            f=open('allval.txt','wb')
            pickle.dump(allval, f)
            pickle.dump(mask,f)
            f.close()
            #ans=numpy.unravel_index(ans,(WINDOW_SIZE,WINDOW_SIZE,self._out_channel))
            return (ans, allval[tuple(ans)])
        #return numpy.unravel_index(ans,(WINDOW_SIZE,WINDOW_SIZE,self._out_channel))
        return ans
    def sample_ans_masked(self, X, mask):
        allval=self.predict_all(X).reshape([WINDOW_SIZE,WINDOW_SIZE,self._out_channel])
        X,Y,Z=numpy.nonzero(mask)
        total=numpy.sum(allval[X,Y,Z])
        ans=numpy.random.choice(len(X),p=allval[X,Y,Z]/total)
        return [X[ans],Y[ans],Z[ans]]
    def train(self,X,Y):
        with self.session.as_default():
            #with self.graph.as_default():
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

class ValueNetwork:
    def __init__(self, in_channel):
        self.session = KTF.get_session()
        self.graph = tf.get_default_graph()
        self.in_channel=in_channel
        self.inp=Input((WINDOW_SIZE,WINDOW_SIZE,in_channel),dtype='float32')
        self.conv1=conv_block(self.inp,3)
        self.pool1=MaxPooling2D((2,2))(self.conv1)
        self.conv2=conv_block(self.inp,3)
        self.pool2=MaxPooling2D((2,2))(self.conv2)
        self.conv3=conv_block(self.inp,3)
        self.pool3=MaxPooling2D((2,2))(self.conv3)
        self.dense1=Dense(256,activation='relu')(Flatten()(self.pool3))
        self.output=Dense(1,activation='linear')(self.dense1)
        self.model=Model(inputs=self.inp,outputs=self.output)
        self.model.compile(optimizer='adam',loss='mse')

    def train_batch(self, X, Y):
        with self.session.as_default():
            with self.graph.as_default():
                return self.model.fit(X,Y)
    def predict(self,X):
        with self.session.as_default():
            with self.graph.as_default():
                return self.model.predict(X.reshape([-1,WINDOW_SIZE,WINDOW_SIZE, self.in_channel]))
    def save(self):
        with self.session.as_default():
            with self.graph.as_default():
                self.model.save('ValueNet%d.h5'%self.in_channel)
    def load(self):
        with self.session.as_default():
            with self.graph.as_default():
                self.model=load_model('ValueNet%d.h5'%self.in_channel)
    def get_weights(self):
        with self.session.as_default():
            with self.graph.as_default():
                return self.model.get_weights()

    def set_weights(self,weights):
        with self.session.as_default():
            with self.graph.as_default():
                self.model.set_weights(weights)
    @staticmethod
    def init_weight(shape, name):
        k=numpy.zeros(shape=shape, name=name)
        
