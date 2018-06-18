import numpy
from DroneNet import DroneNet
import util64
import socket
import pickle
import threading
import random
soc=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
host='linux.cs.uwaterloo.ca'
soc.bind((host,12346))
soc.listen(5)
disGame=None
drones=DroneNet(True)
def unit_control(soc):
    global disGame
    print('new thread')
    while(True):
        try:
            data=util64.recv_msg(soc)
            k=pickle.loads(data)
            #print(k)
            if(k[0]=='reg'):
                disGame=util64.gameInstance(k[1])
                soc.send(b'ok')
                break
            else:
                X=DroneNet.msg2state(disGame,k[1])
                mask=DroneNet.msg2mask(disGame,k[1])
                ans=drones.predict_ans_masked(X,mask)
                #ans=[random.randint(0,359),random.randint(0,359),random.randint(0,5)]
                soc.sendall(pickle.dumps(ans))
        except EOFError:
            break
while(True):
    con,addr=soc.accept()
    k=threading.Thread(target=unit_control,args=[con])
    k.start()
    '''
    while(True):
        alldata=b''
        data=util64.recv_msg(con)
        #print('server recv',len(data))
        k=pickle.loads(data)
        #print(k)
        if(k[0]=='reg'):
            disGame=util64.gameInstance(k[1])
            con.sendall(b'ok')
        else:
            X=disGame.msg2stateDrone(k[1])
            mask=disGame.msg2maskDrone(k[1])
            ans=drones.predict_ans_masked(X,mask)
            #ans=drones.predict_ans([k[1]])
            ans=[random.randint(0,359),random.randint(0,359),random.randint(0,5)]
            #print('server send',ans)
            con.sendall(pickle.dumps(ans))
    '''
