import numpy
import struct
import struct
# import matplotlib.pyplot as plt
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Concatenate, BatchNormalization, UpSampling2D, Layer, Add
from keras.layers import Reshape, Dense, Dropout, Embedding, LSTM, Flatten, Conv2D, MaxPooling2D, Conv2DTranspose, Activation, Lambda
from keras.optimizers import Adam
from keras import backend as KTF
from keras.layers.advanced_activations import LeakyReLU
import keras
from consts import WINDOW_SIZE

def get_trainable_params(agent):
    return agent.model.trainable_weights


def conv_block(inp, times, has_input=False):
    x= Lambda(lambda i:i+0)(inp)
    for i in range(times):
        conv1 = Conv2D(32, (3, 3), activation=LeakyReLU(), padding='same')(x)
        conv2 = Conv2D(32, (5, 5), activation=LeakyReLU(), padding='same')(x)
        conv3 = Conv2D(32, (7, 7), activation=LeakyReLU(), padding='same')(x)
        conv4 = Conv2D(32, (9, 9), activation=LeakyReLU(), padding='same')(x)

        x = Concatenate(axis=3)([conv1, conv2, conv3, conv4])
    # if(has_input):
    #    return x
    # return x
    short = Conv2D(128, (1, 1),  padding='same')(inp)
    #short= inp
    return short
    #return Add()([x, short])


def deconv_block(inp, times):
    x = Lambda(lambda i:i+0)(inp)
    for i in range(times):
        conv1 = Conv2DTranspose(32, (3, 3), activation=LeakyReLU(), padding='same')(x)
        conv2 = Conv2DTranspose(32, (5, 5), activation=LeakyReLU(), padding='same')(x)
        conv3 = Conv2DTranspose(32, (7, 7), activation=LeakyReLU(), padding='same')(x)
        conv4 = Conv2DTranspose(32, (9, 9), activation=LeakyReLU(), padding='same')(x)
        x = Concatenate(axis=3)([conv1, conv2, conv3, conv4])
    short = Conv2DTranspose(128, (1, 1), padding='same')(inp)
    return Add()([x, short])


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
    if (x < 0):
        return 0
    if (x > WINDOW_SIZE - 1):
        return WINDOW_SIZE - 1
    return x

class gameMap:
    def __init__(self, reg, name):
        # print(reg,reg.shape)
        self.name=name
        self.regions = numpy.zeros([reg.ans.shape[0] * 8, reg.ans.shape[1] * 8])
        for i in range(reg.ans.shape[0]):
            for j in range(reg.ans.shape[1]):
                self.regions[i * 8:i * 8 + 8, j * 8:j * 8 + 8] = reg.ans[i, j]
        #self.regions=numpy.transpose(self.regions)

class Maps:
    def __init__(self):
        self.maps=[]
    def is_empty(self):
        return len(self.maps)==0
    def add_map(self,map):
        self.maps.append(map)
    def find_map(self,name):
        for i in self.maps:
            if(i.name==name):
                return i
        return None
def send_msg(sock, msg):
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)

def inReach(up, mp):
    if (abs(up[0] - mp[0]) < WINDOW_SIZE // 2 and abs(up[1] - mp[1]) < WINDOW_SIZE // 2):
        return True
    return False