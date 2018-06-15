import util64
import pickle
import socket
import numpy
import ReplayBuffer
import DroneNet
import threading

#Deep Q Learning
batch_size=32
disGame=None
learning=threading.Semaphore(value=1)
buf=ReplayBuffer.ReplayBuffer(20000)
drones=DroneNet.DroneNet(True)
target=DroneNet.DroneNet(True)
epsilon=0.3
discount=0.9
def learner():
    global drones,buf,disgame,target,discount,learning
    learning.aquire()
    new_agent=DroneNet.DroneNet()
    new_agent.set_weights(drones.get_weights())
    samples=buf.sample(batch_size)
    if(sample==None):
        learning.release()
        return
    X=numpy.array([util64.game2stateDrone(i) for i in samples[0]])
    Y=drones.predict_all(X)
    maxes=target.predict_max(numpy.array([util64.game2stateDrone(i) for i in samples[2]]))
    Y_=[(samples[i][3]+discountmaxes[i]) for i in range(batch_size)]
    diff=numpy.zeros(Y.shape)
    for i in range(batch_size):
        diff[i,samples[i][1][0],samples[i][1][1],samples[i][1][2]]=Y_[i]
    diff+=Y
    new_agent.fix(X,diff)
    learning.release()
def unit_RL(con):
    global disGame,buf,drones,epsilon
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
                soc.send(b'ok')
                break
            else:
                ans=0
                if(random.random.random()<epsilon):
                    ans=[random.randint(0,359),random.randint(0,359),random.randint(0,5)]
                else:
                    X=disGame.msg2stateDrone(k[1])
                    mask=disGame.msg2maskDrone(k[1])
                    ans=drones.predict_ans_masked(X,mask)
                #ans=[random.randint(0,359),random.randint(0,359),random.randint(0,5)]
                soc.sendall(pickle.dumps(ans))
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
    while(True):
        con,addr=soc.accept()
        k=threading.Thread(target=unit_RL,args=[con])
        k.start()
