import numpy
import keras
import scipy
import pickle
import socket
import struct
#import matplotlib.pyplot as plt
from keras.models import Sequential, Model, load_model
from keras.layers import Input,Concatenate,BatchNormalization,UpSampling2D,Layer,Add
from keras.layers import Reshape,Dense, Dropout, Embedding, LSTM,Flatten,Conv2D,MaxPooling2D,Conv2DTranspose
from keras.optimizers import Adam
from keras import backend as KTF
from consts import WINDOW_SIZE

def conv_block(inp,times,has_input=False):
    x=inp
    for i in range(times):
        conv1=Conv2D(8,(3,3),activation='relu',padding='same')(x)
        conv2=Conv2D(8,(5,5),activation='relu',padding='same')(x)
        conv3=Conv2D(8,(7,7),activation='relu',padding='same')(x)
        conv4=Conv2D(8,(9,9),activation='relu',padding='same')(x)
        x=Concatenate(axis=3)([conv1,conv2,conv3,conv4])
    #if(has_input):
    #    return x
    #return x
    short=Conv2D(32,(1,1),activation='linear',padding='same')(inp)
    return Add()([x,short])
def deconv_block(inp,times):
    x=inp
    for i in range(times):
        conv1=Conv2DTranspose(8,(3,3),activation='relu',padding='same')(x)
        conv2=Conv2DTranspose(8,(5,5),activation='relu',padding='same')(x)
        conv3=Conv2DTranspose(8,(7,7),activation='relu',padding='same')(x)
        conv4=Conv2DTranspose(8,(9,9),activation='relu',padding='same')(x)
        x=Concatenate(axis=3)([conv1,conv2,conv3,conv4])
    short=Conv2DTranspose(32,(1,1),activation='linear',padding='same')(inp)
    return Add()([x,short])

def recv_msg(sock):
    # Read message length and unpack it into an integer
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # Read the message data
    return recvall(sock, msglen)

def recvall(sock, n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

def shrinkScr(x):
    if(x<0):
        return 0
    if(x>WINDOW_SIZE-1):
        return WINDOW_SIZE-1
    return x


class gameInstance:
    def __init__(self,reg):
        self.regions=numpy.array(reg[1])


    
