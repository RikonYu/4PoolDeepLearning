import pickle
import keras.backend as KTF
import sys
import numpy
import keras
import random
import os
from util32 import WINDOW_SIZE
import util64
from DroneNet import DroneNet
X=[]
Y=[]
Xt=[]
Yt=[]
freq={}
allrep=[]

max_epoch=0
batch_size=0
    
reppath='../reps/'
for i in os.listdir(reppath):
    allrep.append(i)
#train: 90%
#test: 90%+
TRAIN_BATCHES=5000
valid_every=50
save_every=500
ngsl=[]
ngs=[]
ngtl=[]
ngt=[]
def find_place(x,arr):
    left=0
    right=len(arr)
    while(left+1<right):
        mid=(left+right)//2
        if(mid>x):
            right=mid
        elif(mid==x):
            return mid//2
        else:
            left=mid
    return left//2
def valid(model,size=128):
    ans=0
    ind=numpy.random.choice(len(Xt),size,replace=False)
    for i in ind:
        yt=numpy.zeros([WINDOW_SIZE,WINDOW_SIZE,6])
        #print(Yt[i])
        X=numpy.reshape(DroneNet.msg2state(ngt[find_place(i,ngt)],Xt[i]),[1,WINDOW_SIZE,WINDOW_SIZE,18])
        Y=numpy.reshape(DroneNet.y2state(Yt[i]),[1,WINDOW_SIZE,WINDOW_SIZE,6])
        ans+=model.evaluate(X,Y)
        print(model.predict_ans(X),end=' ')
    print('\n')
    return ans/size

if(__name__=='__main__'):
    tout=open('trainerr.txt','wb')
    vout=open('validerr.txt','wb')
    #for i in range(len(allrep)*9//10):
    for i in range(1):
        try:
            f=open(reppath+allrep[i],'rb')
            reg=pickle.load(f)
            ngs.append(util64.gameMap(reg, ''))
            ngsl.append(len(X))
            while(True):
                x=pickle.load(f)
                y=pickle.load(f)
                if(y[2]!=0): 
                    X.append(x)
                    Y.append(y)
        except EOFError:
            f.close()
            ngsl.append(len(X))
    #for i in range(len(allrep)*9//10,len(allrep)):
    for i in range(1,2):
        
        try:
            f=open(reppath+allrep[i],'rb')
            reg=pickle.load(f)
            ngt.append(util64.gameMap(reg, ''))
            ngtl.append(len(Xt))
            while(True):
                x=pickle.load(f)
                y=pickle.load(f)
                if(y[2]!=0): 
                    Xt.append(x)
                    Yt.append(y)
        except EOFError:

            ngtl.append(len(Xt))
            f.close()
            continue
    for i in range(len(Y)):
        if(Y[i][2] in freq):
           freq[Y[i][2]]+=1
        else:
            freq[Y[i][2]]=1
    #print(freq)
    X=numpy.array(X)
    Y=numpy.array(Y)
    max_epoch=int(sys.argv[1])
    batch_size=int(sys.argv[2])
    nX=len(X)
    agent=DroneNet(True)
    tk=0
    trainerr=[]
    validerr=[]
    #for epoch in range(max_epoch):
                        
    ind=list(range(nX))
    numpy.random.shuffle(ind)
    
    for epoch in range(TRAIN_BATCHES):
        picks=numpy.random.choice(nX,batch_size,False)
        X_=numpy.array([DroneNet.msg2state(ngs[find_place(i,ngsl)],X[i]) for i in picks])
        Y_=numpy.zeros([batch_size,WINDOW_SIZE,WINDOW_SIZE,6])
        for j in range(batch_size):
            Y_[j]=DroneNet.y2state(Y[picks[j]])
        print('epoch %d'%epoch, end=' ')
        history=agent.train(X_,Y_)
        trainerr.append(history.history['loss'])
        if(tk%valid_every==0):
            validerr.append(valid(agent))
        if(tk%save_every==0):
            agent.save()
        tk+=1
    agent.save()
    pickle.dump(trainerr,tout)
    pickle.dump(validerr,vout)
    tout.close()
    vout.close()
                
        
            
        
