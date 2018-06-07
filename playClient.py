import numpy
import pickle
import util32
import socket,os
from pybrood import BaseAI, run, game, Color
soc=None
def send(u,tp):
    msg=util32.game2msgDrone(u)
    soc.send([tp,pickle.dumps(msg)])
def send_reg():
    soc.send(['reg',util32.reg2msg()])
def receive():
    k=soc.receive(10240)
    return pickle.loads(k)
class PlayAI(BaseAI):
    def prepare(self):
        self.playerMe=game.self()
        send_reg()
    def frame(self):
        if(game.getFrameCount()%10!=0):
            continue
        for i in game.getAllUnits():
            if(i.getType().getName()=='Zerg_Drone' and i.getPlayer()==self.playerMe):
                send(i,'drone')
                k=receive()
                util32.command(i,k)
    def finished(self):
        pass

if(__name__=='__main__'):
    soc=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    soc.connect(('linux.cs.uwaterloo.ca',22))
    run(PlayAI)
