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

freq={}
allrep=[]
reppath='../reps/'
for i in os.listdir(reppath):
    allrep.append(i)
#train: 1~1900
#test: 1900+
#train_size=100
valid_every=50
if(__name__=='__main__'):
    tout=open('trainerr.txt','wb')
    vout=open('validerr.txt','wb')
    for i in range(len(allrep)*0.9):
        try:
            f=open(allrep[i],'rb')
            reg=pickle.load(f)
            print('reg:',reg)
            print('\n\n\n')
            util64.makeReg(reg)
            while(True):
                x=pickle.load(f)
                y=pickle.load(f)
                
                #print(x,y)
                X.append(x)
                Y.append(y)
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
                #print(Y[ind[i*batch_size+j]])
                if(Y[ind[i*batch_size+j]][2]==5):
                    Y_[j,180,180,5]=1
                elif(Y[ind[i*batch_size+j]][2]==0):
                    Y_[j,180,180,0]=1
                else:
                    Y_[j][tuple(Y[ind[i*batch_size+j]])]=1
            #print(X_.shape,Y_.shape,batch_size)
            #history=agent.model.fit(numpy.zeros([2,360,360,18]),numpy.zeros([2,360,360,6]))
            history=agent.train(X_,Y_)
            trainerr.append(history.history['loss'])
            if(tk%valid_every==0):
                #validerr.append(agent.evaluate(X_,Y_))
                pass
            tk+=1
    pickle.dump(trainerr,tout)
    pickle.dump(validerr,vout)
    tout.close()
    vout.close()
                
        
            
        
