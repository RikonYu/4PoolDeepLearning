from UnitNet import UnitNet
import keras
import tensorflow as tf
from keras.models import Sequential, Model, load_model
from keras.layers import Input,Concatenate,BatchNormalization,UpSampling2D,Layer,Add, Activation
from keras.layers import Reshape,Dense, Dropout, Embedding, LSTM,Flatten,Conv2D,MaxPooling2D,Conv2DTranspose
from keras.optimizers import Adam, SGD
from keras import backend as KTF
import numpy
from util64 import conv_block, deconv_block, shrinkScr
from consts import WINDOW_SIZE
from keras.layers.advanced_activations import LeakyReLU
import os,sys

class VultureNet(UnitNet):
    _in_channel=7
    _out_channel=3
    def __init__(self, loading=False, output_type='linear'):
        self._in_channel = VultureNet._in_channel
        self._out_channel = VultureNet._out_channel
        self.session = KTF.get_session()
        self.graph = tf.get_default_graph()
        with self.session.as_default():
            with self.graph.as_default():
                self.inp = Input((WINDOW_SIZE, WINDOW_SIZE, self._in_channel), dtype='float32')
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
                    self.out=Conv2DTranspose(self._out_channel, (3, 3), padding='same')(
                    self.deconv4)
                    self.out=Flatten()(self.out)
                    self.out=Activation('softmax')(self.out)
                    self.out=Reshape([-1,WINDOW_SIZE,WINDOW_SIZE,self._out_channel])(self.out)
                else:
                    self.out = Conv2DTranspose(self._out_channel, (3, 3), padding='same')(
                        self.deconv4)
                #optz=SGD(lr=0.001,momentum=0.9)
                self.model = Model(inputs=self.inp, outputs=self.out)
                self.optz=Adam()
                self.model.compile(optimizer=self.optz, loss='MSE')
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
        #[enemy, myBound, region, myCooldown, myRange, enemyCooldown, myHp]
        x,y=msg.myInfo.coord
        X,Y=disGame.regions.shape
        ans=numpy.zeros([WINDOW_SIZE,WINDOW_SIZE,VultureNet._in_channel])
        for u in msg.enemies:
            if (abs(u.coord[0] - x) < WINDOW_SIZE // 2 and abs(u.coord[1] - y) < WINDOW_SIZE // 2):
                ans[shrinkScr(u.bounds[0]-x+WINDOW_SIZE//2):shrinkScr(u.bounds[1]-x+WINDOW_SIZE//2),
                    shrinkScr(u.bounds[2]-y+WINDOW_SIZE//2):shrinkScr(u.bounds[3]-y+WINDOW_SIZE//2),0]=1
                ans[u.coord[0]-x+WINDOW_SIZE//2,u.coord[1]-y+WINDOW_SIZE//2,5]= u.canFireGround
        ax=max(0,WINDOW_SIZE//2-x)
        ay=max(0,WINDOW_SIZE//2-y)
        ans[(msg.myInfo.bounds[0]-x)+WINDOW_SIZE//2:msg.myInfo.bounds[1]-x+WINDOW_SIZE//2,
            (msg.myInfo.bounds[2] - y) + WINDOW_SIZE // 2:msg.myInfo.bounds[3] - y + WINDOW_SIZE // 2,1]=1
        ans[ax:min(WINDOW_SIZE,X-x+WINDOW_SIZE//2),
            ay:min(WINDOW_SIZE,Y-y+WINDOW_SIZE//2),2]=1
        ans[:,:,3]=msg.myInfo.canFireGround
        rng=msg.myInfo.rangeGround[1]
        for i in range(-rng,rng+1):
            for j in range(-int(numpy.sqrt(rng*rng-i*i)),1+int(numpy.sqrt(rng*rng-i*i))):
                ans[WINDOW_SIZE//2+i,WINDOW_SIZE//2+j,4]=1
        ans[:, :, 6]=msg.myInfo.HP
        return ans
    @staticmethod
    def msg2mask(disGame, msg):
        x,y=msg.myInfo.coord
        X, Y = disGame.regions.shape
        ans=numpy.zeros([WINDOW_SIZE,WINDOW_SIZE,VultureNet._out_channel])
        #ans[WINDOW_SIZE//2,WINDOW_SIZE//2,0]=1
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
