
import keras
import tensorflow as tf
from keras.models import Sequential, Model, load_model
from keras.layers import Input,Concatenate,BatchNormalization,UpSampling2D,Layer,Add
from keras.layers import Reshape,Dense, Dropout, Embedding, LSTM,Flatten,Conv2D,MaxPooling2D,Conv2DTranspose
from keras.optimizers import Adam
from keras import backend as KTF
import numpy
from util64 import conv_block, deconv_block, shrinkScr
from consts import WINDOW_SIZE
from UnitNet import UnitNet
from scipy import misc
import os,sys

#common_graph=tf.Graph()
#common_session=tf.Session(graph=common_graph)
class DragoonNet(UnitNet):
    _in_channel=10
    _out_channel=6
    def __init__(self,loading=False):
        #global common_graph, common_session
        self._in_channel = DragoonNet._in_channel
        self._out_channel = DragoonNet._out_channel
        #self.session=common_session
        #self.graph=common_graph
        self.graph=tf.Graph()
        self.session =tf.Session(graph=self.graph)
        with self.session.as_default():
            with self.graph.as_default():
                self.inp=Input((WINDOW_SIZE,WINDOW_SIZE,self._in_channel),dtype='float32')
                self.conv1=Conv2D(32,(1,1),activation='relu',padding='same')(self.inp)
                self.pool1=MaxPooling2D((2,2))(self.conv1)
                self.conv2=conv_block(self.pool1,1)
                self.pool2=MaxPooling2D((2,2))(self.conv2)
                self.conv3=conv_block(self.pool2,1)
                self.pool3=MaxPooling2D((2,2))(self.conv3)

                self.deconv1=deconv_block(self.pool3,2)
                self.up1=UpSampling2D((2,2))(self.deconv1)
                self.deconv2=deconv_block(Concatenate(axis=3)([self.up1,self.conv3]),1)
                self.up2=UpSampling2D((2,2))(self.deconv2)
                self.deconv3=deconv_block(Concatenate(axis=3)([self.up2,self.conv2]),1)
                self.up3=UpSampling2D((2,2))(self.deconv3)
                self.deconv4=Conv2DTranspose(64,(3,3),activation='relu',padding='same')(self.up3)
                self.out=Conv2DTranspose(self._out_channel,(3,3),activation='softmax',padding='same')(self.deconv4)
                self.model=Model(inputs=self.inp,outputs=self.out)
                opt=Adam(lr=0.0001)
                self.model.compile(optimizer=opt,loss='categorical_crossentropy')
                self.model._make_predict_function()
                self.model._make_test_function()
                self.model._make_train_function()
                if(loading and os.path.isfile('DragoonNet.h5')):
                    self.model=load_model('DragoonNet.h5')
    def save(self):
        with self.session.as_default():
            with self.graph.as_default():
                self.model.save('DragoonNet.h5')
    @staticmethod
    def msg2state(disGame, msg):
        x,y=msg[0]
        X,Y=disGame.regions.shape
        ax=max(0,WINDOW_SIZE//2-x)
        ay=max(0,WINDOW_SIZE//2-y)
        ans=numpy.zeros([WINDOW_SIZE,WINDOW_SIZE,DragoonNet._in_channel])
        ans[ax:min(WINDOW_SIZE,X-x+WINDOW_SIZE//2),
            ay:min(WINDOW_SIZE,Y-y+WINDOW_SIZE//2),0]=disGame.regions[max(0,x-WINDOW_SIZE//2):min(x+WINDOW_SIZE//2,X),
                                                                      max(0,y-WINDOW_SIZE//2):min(y+WINDOW_SIZE//2,Y)]
        for u in msg[2]:#enemy
            nx = u[0][0] - x + WINDOW_SIZE // 2
            ny = u[0][1] - y + WINDOW_SIZE // 2
            ans[nx, ny, 2] = 1
            ans[nx, ny, 5] = u[1]
            ans[nx, ny, 6] = u[4]
            ans[nx, ny, 7] = u[5]
        for u in msg[3]:#ally
            nx = u[0][0] - x + WINDOW_SIZE // 2
            ny = u[0][1] - y + WINDOW_SIZE // 2
            ans[nx, ny, 1] = 1
            ans[nx, ny, 4] = u[1]
        ans[:,:,3]=msg[1][0]
        for i in range(WINDOW_SIZE*32//X):
            for j in range(WINDOW_SIZE * 32 // X):
                #print(ans[i::WINDOW_SIZE*32//X,j::WINDOW_SIZE*32//X,8].shape,msg[6].shape)
                ans[i::WINDOW_SIZE*32//X,j::WINDOW_SIZE*32//X,8]=msg[6]
        ans[x*WINDOW_SIZE//X,y*WINDOW_SIZE//Y,9]=1
        return ans
    @staticmethod
    def msg2mask(disGame, msg):
        ans=numpy.zeros([WINDOW_SIZE, WINDOW_SIZE, DragoonNet._out_channel])
        x,y=msg[0]
        X, Y = disGame.regions.shape
        ax=max(0,WINDOW_SIZE//2-x)
        ay=max(0,WINDOW_SIZE//2-y)
        ans[WINDOW_SIZE//2,WINDOW_SIZE//2,0]=1
        ans[ax:min(WINDOW_SIZE,X-x+WINDOW_SIZE//2),
            ay:min(WINDOW_SIZE,Y-y+WINDOW_SIZE//2),1]=disGame.regions[max(0,x-WINDOW_SIZE//2):min(x+WINDOW_SIZE//2,X),
                                                                      max(0,y-WINDOW_SIZE//2):min(y+WINDOW_SIZE//2,Y)]
        for u in msg[2]:
            nx = u[0][0] - x + WINDOW_SIZE // 2
            ny = u[0][1] - y + WINDOW_SIZE // 2
            ans[nx, ny, 4] = 1
            top,bot,left,right=u[6]
            #ans[shrinkScr(top - x + WINDOW_SIZE // 2):shrinkScr(bot - x + WINDOW_SIZE // 2),
            #    shrinkScr(left - y + WINDOW_SIZE // 2):shrinkScr(right - x + WINDOW_SIZE // 2),1] = 0
        for u in msg[3]:
            top,bot,left,right=u[4]
            #ans[shrinkScr(top - x + WINDOW_SIZE // 2):shrinkScr(bot - x + WINDOW_SIZE // 2),
            #    shrinkScr(left - y + WINDOW_SIZE // 2):shrinkScr(right - x + WINDOW_SIZE // 2),1] = 0
        return ans