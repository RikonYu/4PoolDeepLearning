from UnitNet import UnitNet
import keras
import tensorflow as tf
from keras.models import Sequential, Model, load_model
from keras.layers import Input,Concatenate,BatchNormalization,UpSampling2D,Layer,Add
from keras.layers import Reshape,Dense, Dropout, Embedding, LSTM,Flatten,Conv2D,MaxPooling2D,Conv2DTranspose
from keras.optimizers import Adam, SGD
from keras import backend as KTF
import numpy
from util64 import conv_block, deconv_block, shrinkScr
from consts import WINDOW_SIZE
from scipy import misc
import os,sys

class VultureNet(UnitNet):
    _in_channel=2
    _out_channel=3
    def __init__(self, loading=False, output_type='linear'):
        self._in_channel = VultureNet._in_channel
        self._out_channel = VultureNet._out_channel
        self.session = KTF.get_session()
        self.graph = tf.get_default_graph()
        with self.session.as_default():
            with self.graph.as_default():
                self.inp = Input((WINDOW_SIZE, WINDOW_SIZE, self._in_channel), dtype='float32')
                self.conv1 = Conv2D(32, (1, 1), activation='relu', padding='same')(self.inp)
                self.conv1 = conv_block(self.conv1, 1)
                self.pool1 = MaxPooling2D((2, 2))(self.conv1)
                self.conv2 = conv_block(self.pool1, 1)
                self.pool2 = MaxPooling2D((2, 2))(self.conv2)

                self.deconv1 = deconv_block(self.pool2, 1)
                self.up1 = UpSampling2D((2, 2))(self.deconv1)
                self.deconv2 = deconv_block(Concatenate(axis=3)([self.up1, self.conv2]), 1)
                self.up3 = UpSampling2D((2, 2))(self.deconv2)
                self.deconv4 = Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(self.up3)
                self.out = Conv2DTranspose(VultureNet._out_channel, (3, 3), activation=output_type, padding='same')(
                    self.deconv4)
                self.model = Model(inputs=self.inp, outputs=self.out)
                optz = Adam(lr=0.002, momentum=0.9, decay=1e-8)
                self.model.compile(optimizer=optz, loss='MSE')
                self.model._make_predict_function()
                self.model._make_test_function()
                self.model._make_train_function()
                if (os.path.isfile('VultureNet.h5') and loading):
                    self.model = load_model("VultureNet.h5")
    def save(self):
        with self.session.as_default():
            with self.graph.as_default():
                self.model.save('VultureNet.h5')
    @staticmethod
    def msg2state(disGame, msg):
        x,y=msg.myInfo.coord
        X,Y=disGame.regions.shape
        ans=numpy.zeros([WINDOW_SIZE,WINDOW_SIZE,VultureNet._in_channel])
        for u in msg.enemies:
            if (abs(u.coord[0] - x) < WINDOW_SIZE // 2 and abs(u.coord[1] - y) < WINDOW_SIZE // 2):
                ans[shrinkScr(u.bounds[0]-x+WINDOW_SIZE//2):shrinkScr(u.bounds[1]-x+WINDOW_SIZE//2),
                    shrinkScr(u.bounds[2]-y+WINDOW_SIZE//2):shrinkScr(u.bounds[3]-y+WINDOW_SIZE//2),0]=1
        ax=max(0,WINDOW_SIZE//2-x)
        ay=max(0,WINDOW_SIZE//2-y)
        ans[WINDOW_SIZE//2,WINDOW_SIZE//2,0]=1
        ans[ax:min(WINDOW_SIZE,X-x+WINDOW_SIZE//2),
            ay:min(WINDOW_SIZE,Y-y+WINDOW_SIZE//2),1]=1
        return ans
    @staticmethod
    def msg2mask(disGame, msg):
        x,y=msg.myInfo.coord
        X, Y = disGame.regions.shape
        ans=numpy.zeros([WINDOW_SIZE,WINDOW_SIZE,VultureNet._out_channel])
        ans[WINDOW_SIZE//2,WINDOW_SIZE//2,0]=1
        ax=max(0,WINDOW_SIZE//2-x)
        ay=max(0,WINDOW_SIZE//2-y)
        ans[WINDOW_SIZE//2,WINDOW_SIZE//2,0]=1
        ans[ax:min(WINDOW_SIZE,X-x+WINDOW_SIZE//2),
            ay:min(WINDOW_SIZE,Y-y+WINDOW_SIZE//2),1]=1
        myPos=msg.myInfo.coord
        if(msg.myInfo.canFireGround==0):
            for u in msg.enemies:
                if (abs(u.coord[0] - myPos[0]) < WINDOW_SIZE // 2 and abs(u.coord[1] - myPos[1]) < WINDOW_SIZE // 2):
                    ans[u.coord[0]-myPos[0]+WINDOW_SIZE//2,u.coord[1]-myPos[1]+WINDOW_SIZE//2, 2]=1
        return ans
