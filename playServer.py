import numpy
import DroneNet
import util64
import socket
import pickle
import random
soc=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
host='linux.cs.uwaterloo.ca'
soc.bind((host,123456))
soc.listen(5)
disGame=None
drones=DroneNet(Loading=False)
while(True):
    c,addr=soc.accept()
    while(True):
        data=c.recv(20480)
        if(data):
            k=pickle.loads(data)
            if(k[0]=='reg'):
                disGame=util64.gameInstance(k[0])
            else:
                ans=DroneNet.predict_ans(k)
                soc.send(pickle.dumps([random.randint()%360,random.randint()%360,random.randint()%6]))
            
