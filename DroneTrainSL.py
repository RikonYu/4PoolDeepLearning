import pickle
import keras.backend as KTF
import sys
import numpy
import keras
import os
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
#train: 1~1900
#test: 1900+
#train_size=100
valid_every=50
def valid(model,size=128):
    ans=0
    ind=numpy.random.choice(len(Xt),size,replace=False)
    for i in ind:
        yt=numpy.zeros([360,360,6])
        ans+=model.evaluate(
            numpy.reshape(util64.msg2stateDrone(Xt[i]),[1,360,360,18]),
             numpy.reshape(util64.y2stateDrone(Yt[i]),[1,360,360,6]))
    return ans/size
if(__name__=='__main__'):
    tout=open('trainerr.txt','wb')
    vout=open('validerr.txt','wb')
    for i in range(len(allrep)*9//10):
        try:
            f=open(reppath+allrep[i],'rb')
            reg=pickle.load(f)
            util64.makeReg(reg)
            while(True):
                x=pickle.load(f)
                y=pickle.load(f)
                X.append(x)
                Y.append(y)
            f.close()
        except EOFError:
            continue

    for i in range(len(allrep)*9//10,len(allrep)):
        try:
            f=open(reppath+allrep[i],'rb')
            reg=pickle.load(f)
            util64.makeReg(reg)
            while(True):
                x=pickle.load(f)
                y=pickle.load(f)
                Xt.append(x)
                Yt.append(y)
            f.close()
        except EOFError:
            continue
    
    for i in range(len(Y)):
        if(Y[i][2] in freq):
           freq[Y[i][2]]+=1
        else:
            freq[Y[i][2]]=1
    print(freq)
    X=numpy.array(X)
    Y=numpy.array(Y)
    max_epoch=int(sys.argv[1])
    batch_size=int(sys.argv[2])
    nX=len(X)
    agent=DroneNet(False)
    tk=0
    trainerr=[]
    validerr=[]
    for epoch in range(max_epoch):
        ind=list(range(nX))
        numpy.random.shuffle(ind)
        for i in range(nX//batch_size):
            X_=numpy.array([util64.msg2stateDrone(x) for x in X[ind[i*batch_size:(i+1)*batch_size]]])
            Y_=numpy.zeros([batch_size,360,360,6])
            for j in range(batch_size):
                Y_[j]=util64.y2stateDrone(Y[ind[i*batch_size+j]])
            #print(X_.shape,Y_.shape,batch_size)
            history=agent.train(X_,Y_)
            trainerr.append(history.history['loss'])
            if(tk%valid_every==0):
                validerr.append(valid(agent))
                pass
            tk+=1
    agent.save()
    pickle.dump(trainerr,tout)
    pickle.dump(validerr,vout)
    tout.close()
    vout.close()
                
        
            
        
