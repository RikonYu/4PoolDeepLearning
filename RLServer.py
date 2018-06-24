import util64
import pickle
import socket
import random
import numpy
import ReplayBuffer
from DroneNet import DroneNet
from DragoonNet import DragoonNet
from ClassConstr import getUnitClass
import threading
from consts import WINDOW_SIZE
#Deep Q Learning
batch_size=32
disGame=None
learning=threading.Semaphore(value=1)
buf=ReplayBuffer.ReplayBuffer(20000)
#drones=DroneNet(True)
#target=DroneNet(True)
targetType=''
dragoons=None
target=None

epsilon=0.3
discount=0.9
learn_epoch=0
def learner():
    global dragoons,buf,disGame,target,discount,learning,learn_epoch

    replace_every=500
    learning.aquire()
    samples=buf.sample(batch_size)
    if(samples==None):
        learning.release()
        return
    X=numpy.array([dragoons.msg2state(disGame,i) for i,_a,_sp,_r in samples])
    Y=drones.predict_all(X)
    aprime=target.predict_max(numpy.array(dragoons.msg2state(disGame,i) for _s,_a,i,_r in samples))
    Y_=[(samples[i][3]+discount*aprime[i]) for i in range(batch_size)]
    diff=numpy.copy(Y)
    for i in range(batch_size):
        diff[i,samples[i][1][0],samples[i][1][1],samples[i][1][2]]+=Y_[i]
    #new_agent.fit(X,diff)
    dragoons.train(X,diff)
    if(learn_epoch%replace_every==0):
        target.set_weights(dragoons.get_weights())
    learn_epoch+=1
    learning.release()
def unit_RL(con):
    global disGame,buf,dragoons,epsilon,targetType
    last_state=None
    last_action=None
    last_mineral=None
    while(True):
        try:
            data=util64.recv_msg(con)
            k=pickle.loads(data)
            #print(k)
            if(k[0]=='reg'):
                disGame=util64.gameInstance(k[1])
                targetType=k[2]
                dragoons = getUnitClass(targetType, True)
                target = getUnitClass(targetType, True)
                con.send(b'ok')

                break
            else:
                ans=0
                if(numpy.random.random()<epsilon):
                    places=dragoons.msg2mask(disGame,k[1])
                    ini,inj,ink=numpy.nonzero(places)
                    ind=numpy.random.choice(len(ini))
                    ans=places[ini[ind],inj[ind],ink[ind]]
                else:
                    #X=DroneNet.msg2state(disGame,k[1])
                    #mask=DroneNet.msg2mask(disGame,k[1])
                    X=dragoons.msg2state(disGame,k[1])
                    mask = dragoons.msg2mask(disGame, k[1])
                    ans=dragoons.predict_ans_masked(X,mask)
                #ans=[random.randint(0,359),random.randint(0,359),random.randint(0,5)]
                con.sendall(pickle.dumps(ans))
                if(last_state!=None):
                    buf.add(last_state,last_action,k[1],(last_mineral==1 and k[1][1][1]==0))
                    last_state=k[1]
                    last_action=ans
                    last_mineral=k[1][1][1]
                    if(numpy.random.randint(1,10)==3):
                        lx=threaing.Thread(target=learner,args=[])
                        lx.start()
        except EOFError:
            break
if(__name__=='__main__'):
    soc=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    host='linux.cs.uwaterloo.ca'
    soc.bind((host,12346))
    soc.listen(5)
    print('listening')
    while(True):
        con,addr=soc.accept()
        k=threading.Thread(target=unit_RL,args=[con])
        k.start()
