import numpy
import pickle
import util32
import socket,os
from pybrood import BaseAI, run, game, Color
soc=None
def send(u,tp):
    msg=util32.game2msgDrone(u)
    soc.sendall(pickle.dumps([tp,msg]))
def send_reg():
    #print(len(pickle.dumps(['reg',util32.reg2msg()])))
    soc.send(pickle.dumps(['reg',util32.reg2msg()]))
def receive():
    k=soc.recv(10240)
    while(len(k)==0):
        k=soc.recv(10240)
        
        print(len(k))
    return pickle.loads(k)
class PlayAI(BaseAI):
    def prepare(self):
        self.playerMe=game.self()
        send_reg()
    def frame(self):
        if(game.getFrameCount()%10!=0):
            return
        for i in game.getAllUnits():
            if(i.getType().getName()=='Zerg_Drone' and i.getPlayer()==self.playerMe):
                send(i,'drone')
                k=receive()
                util32.command(i,k)
    def finished(self):
        pass

if(__name__=='__main__'):
    soc=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    soc.connect(('linux.cs.uwaterloo.ca',12346))
    run(PlayAI)
