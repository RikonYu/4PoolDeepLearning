
import keras
import tensorflow as tf
from keras.models import Sequential, Model, load_model
from keras.layers import Input,Concatenate,BatchNormalization,UpSampling2D,Layer,Add
from keras.layers import Reshape,Dense, Dropout, Embedding, LSTM,Flatten,Conv2D,MaxPooling2D,Conv2DTranspose
from keras.optimizers import Adam
from keras import backend as KTF
import numpy
from util64 import conv_block, deconv_block, shrinkScr
from UnitNet import UnitNet
import os,sys

class DroneNet(UnitNet):
    def __init__(self,loading=False):
        super(DroneNet,self).__init__(loading)
        INP_CHANNEL=18
        OUT_CHANNEL=6
        self.session=KTF.get_session()
        self.graph=tf.get_default_graph()
        with self.session.as_default():
            with self.graph.as_default():       
                self.inp=Input((360,360,INP_CHANNEL),dtype='float32')
                #self.conv1=conv_block(self.inp,1,True)
                self.conv1=Conv2D(32,(1,1),activation='relu',padding='same')(self.inp)
                self.pool1=MaxPooling2D((2,2))(self.conv1)
                self.conv2=conv_block(self.pool1,1)
                self.pool2=MaxPooling2D((2,2))(self.conv2)
                self.conv3=conv_block(self.pool2,1)
                self.pool3=MaxPooling2D((2,2))(self.conv3)

                self.deconv1=deconv_block(self.pool3,2)
                self.up1=UpSampling2D((2,2))(self.deconv1)
                #print(self.up1.get_shape(),self.conv3.get_shape())
                self.deconv2=deconv_block(Concatenate(axis=3)([self.up1,self.conv3]),1)
                #self.deconv2=deconv_block(self.pool2,1)
                self.up2=UpSampling2D((2,2))(self.deconv2)
                self.deconv3=deconv_block(Concatenate(axis=3)([self.up2,self.conv2]),1)
                self.up3=UpSampling2D((2,2))(self.deconv3)
                #self.deconv4=deconv_block(Concatenate(axis=3)([self.up3,self.conv1]),1)
                self.deconv4=Conv2DTranspose(64,(3,3),activation='relu',padding='same')(self.up3)
                self.out=Conv2DTranspose(OUT_CHANNEL,(3,3),activation='softmax',padding='same')(self.deconv4)
                #self.deconv6.set_shape([None,360,360,OUT_CHANNEL])
                self.model=Model(inputs=self.inp,outputs=self.out)
                #print(self.model.summary())
                opt=Adam(lr=0.0001)
                self.model.compile(optimizer=opt,loss='categorical_crossentropy')
                if(os.path.isfile('DroneNet.h5') and loading==True):
                    self.model=load_model("DroneNet.h5")
    @staticmethod
    def msg2mask(disGame,msg):
        ans=numpy.zeros([360,360,6])
        ans[:,:,0]=1
        ans[:,:,5]=(msg[1][1] or msg[1][2])
        x=msg[0][0]
        y=msg[0][1]
        ax=max(0,180-x)
        ay=max(0,180-y)
        hei,wid=disGame.regions.shape
        ans[ax:min(360,hei-x+180),
            ay:min(360,wid-y+180),1]=disGame.hground[max(0,x-180):min(x+180,hei),max(0,y-180):min(y+180,hei)]

        ans[ax:min(360,hei-x+180),
            ay:min(360,wid-y+180),1]*=disGame.regions[max(0,x-180):min(x+180,wid),max(0,y-180):min(y+180,wid)]
        for i in msg[2]:
            ans[i[0][0]-x+180,i[0][1]-y+180,4]=1-i[2]
        for i in msg[3]:
            ans[i[0][0]-x+180,i[0][1]-y+180,4]=1-i[2]
        if(msg[1][3]==0):
            for i in msg[4]:
                ans[i[1][0]-x+180,i[1][1]-y+180,3]=i[0]
        for i in msg[5]:
            ans[i[1][0]-x+180,i[1][1]-y+180,3]=1
        return ans
    @staticmethod
    def y2state(ind):
        ans=numpy.zeros([360,360,6])
        if(ind[2] in [0,5]):
            ans[180,180,ind[2]]=1
        else:
            ans[shrinkScr(ind[0]),shrinkScr(ind[1]),ind[2]]=1
        return ans

    @staticmethod
    def msg2state(disGame,msg):
        ans=numpy.zeros([360,360,18])
        x,y=msg[0]
        ans[:,:,11]=msg[1][0]
        ans[:,:,16]=msg[1][1]
        ans[:,:,17]=msg[1][2]
        for u in msg[2]:
            nx=u[0][0]-x+180
            ny=u[0][1]-y+180
            if(u[2]):
                ans[nx,ny,5]=1
            elif(u[3]):
                ans[nx,ny,6]=1
            else:
                ans[nx,ny,4]=1
            ans[nx,ny,13]=u[1]
            ans[nx,ny,14]=u[4]
            ans[nx,ny,15]=u[5]
        for u in msg[3]:
            
            nx=u[0][0]-x+180
            ny=u[0][1]-y+180
            if(u[2]):
                ans[nx,ny,5]=1
            elif(u[3]):
                ans[nx,ny,6]=1
            else:
                ans[nx,ny,4]=1
            ans[nx,ny,12]=u[1]
        for u in msg[4]:
            
            nx=u[1][0]-x+180
            ny=u[1][1]-y+180
            if(u[0]):
                ans[nx,ny,7]=1
            else:
                ans[nx,ny,8]=1
        for u in msg[5]:

            nx=u[0]-x+180
            ny=u[1]-y+180
            #print(u,x,y,nx,ny)
            ans[nx,ny,9]=1
        ax=max(0,180-x)
        ay=max(0,180-y)
        X=disGame.hground.shape[0]
        Y=disGame.hground.shape[1]
        ans[ax:min(360,X-x+180),
            ay:min(360,Y-y+180),10]=disGame.hground[max(0,x-180):min(x+180,X),max(0,y-180):min(y+180,Y)]

        ans[ax:min(360,X-x+180),
            ay:min(360,Y-y+180),0]=disGame.regions[max(0,x-180):min(x+180,X),max(0,y-180):min(y+180,Y)]
        return ans
