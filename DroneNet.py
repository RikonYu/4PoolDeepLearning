import keras
import tensorflow as tf
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Concatenate, BatchNormalization, UpSampling2D, Layer, Add, Activation
from keras.layers import Reshape, Dense, Dropout, Embedding, LSTM, Flatten, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam, SGD
from keras import backend as KTF
import numpy
from util64 import conv_block, deconv_block, shrinkScr
from consts import WINDOW_SIZE
from UnitNet import UnitNet
import pickle
from keras.layers.advanced_activations import LeakyReLU
import os, sys


class DroneNet(UnitNet):
    _in_channel=5
    _out_channel=2
    def __init__(self,loading=False, output_type='linear'):
        self._in_channel=DroneNet._in_channel
        self._out_channel=DroneNet._out_channel
        self.session = KTF.get_session()
        self.graph = tf.get_default_graph()
        with self.session.as_default():
            #with self.graph.as_default():
            self.inp=Input((WINDOW_SIZE,WINDOW_SIZE,self._in_channel),dtype='float32')
            self.conv1 = LeakyReLU()(Conv2D(128, (3, 3),  padding='same')(self.inp))
            self.conv1= conv_block(self.conv1,2)
            self.pool1 = MaxPooling2D((2, 2))(self.conv1)
            self.conv2 = conv_block(self.pool1, 2)
            self.pool2 = MaxPooling2D((2, 2))(self.conv2)
            self.conv3=conv_block(self.pool2, 2)

            self.pool3=MaxPooling2D((2,2))(self.conv3)
            self.deconv1 = deconv_block(self.pool3, 1)
            self.up1 = UpSampling2D((2, 2))(self.deconv1)

            self.deconv2 = deconv_block(Concatenate(axis=3)([self.up1, self.conv3]), 1)
            self.up2=UpSampling2D((2,2))(self.deconv2)
            self.deconv3 = deconv_block(Concatenate(axis=3)([self.up2, self.conv2]), 1)
            self.up3 = UpSampling2D((2, 2))(self.deconv3)
            self.deconv4 = LeakyReLU()(Conv2DTranspose(128, (3, 3), padding='same')(Concatenate(axis=3)([self.up3, self.conv1])))
            if(output_type=='softmax'):
                self.out=Conv2DTranspose(DroneNet._out_channel, (3, 3), padding='same')(
                self.deconv4)
                self.out=Flatten()(self.out)
                self.out=Activation('softmax')(self.out)
                self.out=Reshape([-1,WINDOW_SIZE,WINDOW_SIZE,self._out_channel])(self.out)
            else:
                self.out = Conv2DTranspose(DroneNet._out_channel, (3, 3), padding='same')(
                    self.deconv4)
            #optz=SGD(lr=0.001,momentum=0.9)
            self.model = Model(inputs=self.inp, outputs=self.out)
            self.optz=Adam()
            self.model.compile(optimizer=self.optz, loss='MSE')

            self.model._make_predict_function()
            self.model._make_test_function()
            self.model._make_train_function()
            if (os.path.isfile('DroneNet.h5') and loading):
                self.model = load_model("DroneNet.h5")
    def save(self):
        with self.session.as_default():
            with self.graph.as_default():
                self.model.save('DroneNet.h5')
    def gradient(self, loss):
        with self.session.as_default():
            #with self.graph.as_default():
            return KTF.gradients(loss, self.model.trainable_weights)
    @staticmethod
    def msg2state(disGame, msg):
        ans=numpy.zeros([WINDOW_SIZE,WINDOW_SIZE,DroneNet._in_channel])
        x, y = msg.myInfo.coord
        X, Y = disGame.regions.shape
        #print(disGame.name, disGame.regions.shape,x,y)
        ax=max(0,WINDOW_SIZE//2-x)
        ay=max(0,WINDOW_SIZE//2-y)
        '''
        ans[ax:min(WINDOW_SIZE,X-x+WINDOW_SIZE//2),
            ay:min(WINDOW_SIZE,Y-y+WINDOW_SIZE//2),0]=disGame.regions[max(0,x-WINDOW_SIZE//2):min(x+WINDOW_SIZE//2,X),
                                                                      max(0,y-WINDOW_SIZE//2):min(y+WINDOW_SIZE//2,Y)]

        ans[WINDOW_SIZE//2,WINDOW_SIZE//2,1]=1
        for u in msg.allies:
            ans[u.coord[0]*WINDOW_SIZE//X,u.coord[1]*WINDOW_SIZE//Y,3]=1
        '''
        for u in msg.resources:
            #print(u.type)
            if(u.type=='Resource_Vespene_Geyser'):
                #ans[shrinkScr(u.coord[0]-x+WINDOW_SIZE//2),shrinkScr(u.coord[1]-y+WINDOW_SIZE//2),4]=1
                ans[shrinkScr(u.bounds[0] - x + WINDOW_SIZE // 2):shrinkScr(u.bounds[1] - x + WINDOW_SIZE // 2),
                shrinkScr(u.bounds[2] - y + WINDOW_SIZE // 2) : shrinkScr(u.bounds[3] - y + WINDOW_SIZE // 2), 4] = 1
        fstate=open('state.txt','wb')
        pickle.dump(ans, fstate)
        fstate.close()
        return ans
    @staticmethod
    def msg2mask(disGame, msg):
        x,y=msg.myInfo.coord
        X, Y = disGame.regions.shape
        ax=max(0,WINDOW_SIZE//2-x)
        ay=max(0,WINDOW_SIZE//2-y)
        ans=numpy.zeros([WINDOW_SIZE,WINDOW_SIZE,DroneNet._out_channel])
        ans[WINDOW_SIZE//2,WINDOW_SIZE//2,0]=1
        ans[ax:min(WINDOW_SIZE,X-x+WINDOW_SIZE//2),
            ay:min(WINDOW_SIZE,Y-y+WINDOW_SIZE//2),1]=disGame.regions[max(0,x-WINDOW_SIZE//2):min(x+WINDOW_SIZE//2,X),
                                                                      max(0,y-WINDOW_SIZE//2):min(y+WINDOW_SIZE//2,Y)]
        '''
        for u in msg.allies:
            top,bot,left,right=u.bounds
            ans[shrinkScr(top - x + WINDOW_SIZE // 2):shrinkScr(bot - x + WINDOW_SIZE // 2),
                shrinkScr(left - y + WINDOW_SIZE // 2):shrinkScr(right - x + WINDOW_SIZE // 2),1] = 0
        for u in msg.resources:
            if(inReach(u.coord, msg.myInfo.coord)):
                top,bot,left,right=u.bounds
                ans[shrinkScr(top - x + WINDOW_SIZE // 2):shrinkScr(bot - x + WINDOW_SIZE // 2),
                    shrinkScr(left - y + WINDOW_SIZE // 2):shrinkScr(right - x + WINDOW_SIZE // 2),1] = 0
        '''
        return ans
