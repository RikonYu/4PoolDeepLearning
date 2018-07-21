import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Input,Concatenate,BatchNormalization,UpSampling2D
from keras.layers import Reshape,Dense, Dropout, Embedding, LSTM,Flatten,Conv2D,MaxPooling2D,Conv2DTranspose
from keras.optimizers import Adam
import socket
import os
import copy
from scipy.spatial import distance
import matplotlib.pyplot as plt
import pickle
import numpy
import numpy.random
import subprocess

LEARNING=1
EPS=0
DQN_BATCH=16
soc=socket.socket(socket.AF_INET,socket.SOCK_STREAM)

host='127.0.0.1'
soc.bind((host,6666))
soc.listen(5)
scope_center=set(numpy.ndindex(360,360,1))
def get_scope(me,others,zind):
    ans=[]
    for i in others:
        if(distance.euclidean(i[0],me[0])>=180):
            continue
        rng=numpy.ndindex(i[4]-i[2],i[3]-i[1],1)
        rng=numpy.array(list(rng))
        rng[0]+=i[0][0]-me[0][0]
        rng[1]+=i[0][1]-me[0][1]
        ans.extend(scope_center.intersection(map(tuple,rng)))
    #print(ans)
    ras=numpy.array(list(zip(*ans)))
    return ras
#field:terrain,unit,mypos,hasmineral
#mask:gather,return,move,nothing
#data:others, me, minerals,frameCount,reward
def unravel(data):
    others=data[0]
    me=data[1]
    field=numpy.zeros([360,360,4])
    mask=numpy.zeros([360,360,4])
    field[:,:,0]=1
    field[get_scope(me,others,1)]=1
    field[180,180,2]=1
    field[:,:,3]=data[2]
    mask[get_scope(me,others,0)]=1
    mask[:,:,1]=data[2]
    mask[:,:,2]=1
    mask[:180-me[0][0],:,2]=0
    mask[:,:180-me[0][1],2]=0
    mask[4096-me[0][0]+180:,:,2]=0
    mask[:,4096-me[0][1]+180:,2]=0
    mask[:,:,3]=(data[5]!='PlayerGuard' and data[5]!='Nothing')
    return (field,mask,data[4],data[3])
def random_action(mask):
    places=numpy.nonzero(mask)
    pos=numpy.random.choice(len(places[0]))
    return [i[pos] for i in places]
mining=MiningNet()
miningt=MiningNet()
miningQ=DQN.DQN(DQN_BATCH,10000,miningt)
prv=None#[field,action,reward]
first_sar=0
train_every=50
last_train=0
replace_every=2400
ttttt=0
print('listening')
subprocess.Popen(['e:\python32bit\python.exe','sender.py'])
while True:
    c,addr=soc.accept()
    while True:
        data=c.recv(40960)
        if(data):
            ans=pickle.loads(data)
            ans=unravel(ans)
            action=0
            if(LEARNING and numpy.random.uniform()>EPS):
                action=random_action(ans[1])
            else:
                ttttt+=1
                action=mining.mask_pos(ans[0],ans[1])
            #print(action)
            c.send(pickle.dumps(action))
            #c.send(pickle.dumps([0,1,2]))
            #print('ans',ans[0].shape)
            if(prv!=None):
                miningQ.add_experience(prv[0],prv[1],ans[2]-prv[2],ans[0])
            prv=[ans[0],action,ans[2]]
            #print(ans[3])
            if(ans[3]%train_every==0 and ans[3]>DQN_BATCH and ans[3]!=last_train):
                exp=miningQ.experience_replay()
                print(numpy.mean(exp[2]))
                miningt.train_experience(exp)
                last_train=ans[3]
            if(ans[3]%replace_every==0):
                mining.set_weights(miningt.get_weights())
            if(ans[3]==1):
                mining.save_model()
        else:
            break
        
