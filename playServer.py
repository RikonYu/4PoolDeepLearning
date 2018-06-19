import numpy
from DroneNet import DroneNet
import util64
import socket
import pickle
import threading
import random
import platform
import subprocess
soc=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
if(platform.system()=='Windows'):
    host='127.0.0.1'
    import matplotlib.pyplot as plt
else:
    host='linux.cs.uwaterloo.ca'
soc.bind((host,12346))
soc.listen(5)
disGame=None
plotting=0
drones=DroneNet()
def unit_control(soc):
    global disGame,plotting
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
                if(platform.system()=='Windows'):
                    if(plotting==0):
                        plotting=1
                        fig=plt.figure(figsize=(6,6))
                        for i in range(6):
                            fig.add_subplot(2,3,i+1)
                            plt.imshow(mask[:,:,i]*255.0,cmap=plt.cm.gray)
                        plt.show()
                #ans=[random.randint(0,),random.randint(0,359),random.randint(0,5)]
                soc.sendall(pickle.dumps(ans))
        except EOFError:
            break
if(platform.system()=='Windows'):
    subprocess.Popen(['e:\python32bit\python.exe','playClient.py'])
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
