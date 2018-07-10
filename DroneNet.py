import keras
import tensorflow as tf
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Concatenate, BatchNormalization, UpSampling2D, Layer, Add
from keras.layers import Reshape, Dense, Dropout, Embedding, LSTM, Flatten, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras import backend as KTF
import numpy
from util64 import conv_block, deconv_block, shrinkScr
from consts import WINDOW_SIZE
from UnitNet import UnitNet
import os, sys


class DroneNet(UnitNet):
    _in_channel=4
    _out_channel=2
    def __init__(self,loading=False):
        self._in_channel=DroneNet._in_channel
        self._out_channel=DroneNet._out_channel
        self.session = KTF.get_session()
        self.graph = tf.get_default_graph()
        with self.session.as_default():
            with self.graph.as_default():
                self.inp=Input((WINDOW_SIZE,WINDOW_SIZE,self._in_channel),dtype='float32')
                self.conv1 = Conv2D(32, (1, 1), activation='relu', padding='same')(self.inp)
                self.pool1 = MaxPooling2D((2, 2))(self.conv1)
                self.conv2 = conv_block(self.pool1, 1)
                self.pool2 = MaxPooling2D((2, 2))(self.conv2)

                self.deconv1 = deconv_block(self.pool2, 2)
                self.up1 = UpSampling2D((2, 2))(self.deconv1)
                self.deconv2 = deconv_block(Concatenate(axis=3)([self.up1, self.conv2]), 1)
                self.up3 = UpSampling2D((2, 2))(self.deconv2)
                self.deconv4 = Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(self.up3)
                self.out = Conv2DTranspose(DroneNet._out_channel, (3, 3), activation='softmax', padding='same')(
                    self.deconv4)
                self.model = Model(inputs=self.inp, outputs=self.out)
                opt = Adam(lr=0.0001)
                self.model.compile(optimizer=opt, loss='categorical_crossentropy')
                self.model._make_predict_function()
                self.model._make_test_function()
                self.model._make_train_function()
                if (os.path.isfile('DroneNet.h5') and loading):
                    self.model = load_model("DroneNet.h5")
    def save(self):
        with self.session.as_default():
            with self.graph.as_default():
                self.model.save('DroneNet.h5')
    @staticmethod
    def msg2state(disGame, msg):
        ans=numpy.zeros([WINDOW_SIZE,WINDOW_SIZE,DroneNet._in_channel])
        x,y=msg[0]
        X,Y=disGame.regions.shape
        ax=max(0,WINDOW_SIZE//2-x)
        ay=max(0,WINDOW_SIZE//2-y)
        ans[ax:min(WINDOW_SIZE,X-x+WINDOW_SIZE//2),
            ay:min(WINDOW_SIZE,Y-y+WINDOW_SIZE//2),0]=disGame.regions[max(0,x-WINDOW_SIZE//2):min(x+WINDOW_SIZE//2,X),
                                                                      max(0,y-WINDOW_SIZE//2):min(y+WINDOW_SIZE//2,Y)]
        for i in range(WINDOW_SIZE*32//X):
            for j in range(WINDOW_SIZE * 32 // X):
                ans[i::WINDOW_SIZE*32//X,j::WINDOW_SIZE*32//X,2]=msg[6]
        ans[x*WINDOW_SIZE//X,y*WINDOW_SIZE//Y,1]=1
        for u in msg[3]:
            ans[u[0][0]*WINDOW_SIZE//X,u[0][1]*WINDOW_SIZE//Y,3]=1
        return ans
    @staticmethod
    def msg2mask(disGame, msg):
        x,y=msg[0]
        X, Y = disGame.regions.shape
        ax=max(0,WINDOW_SIZE//2-x)
        ay=max(0,WINDOW_SIZE//2-y)
        ans=numpy.zeros([WINDOW_SIZE,WINDOW_SIZE,DroneNet._out_channel])
        ans[WINDOW_SIZE//2,WINDOW_SIZE//2,0]=1
        ans[ax:min(WINDOW_SIZE,X-x+WINDOW_SIZE//2),
            ay:min(WINDOW_SIZE,Y-y+WINDOW_SIZE//2),1]=disGame.regions[max(0,x-WINDOW_SIZE//2):min(x+WINDOW_SIZE//2,X),
                                                                      max(0,y-WINDOW_SIZE//2):min(y+WINDOW_SIZE//2,Y)]
        for u in msg[3]:
            top,bot,left,right=u[4]
            ans[shrinkScr(top - x + WINDOW_SIZE // 2):shrinkScr(bot - x + WINDOW_SIZE // 2),
                shrinkScr(left - y + WINDOW_SIZE // 2):shrinkScr(right - x + WINDOW_SIZE // 2),1] = 0
        for u in msg[4]:
            top,bot,left,right=u[2]
            ans[shrinkScr(top - x + WINDOW_SIZE // 2):shrinkScr(bot - x + WINDOW_SIZE // 2),
                shrinkScr(left - y + WINDOW_SIZE // 2):shrinkScr(right - x + WINDOW_SIZE // 2),1] = 0
        return ans
