from time import sleep
from pybrood import BaseAI, run, game, Color
import pickle
from scipy import misc
import numpy
import socket,os

soc=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
soc.connect(('127.0.0.1',6666))
minerals=[]
def wrapup(units,regions,me):
    ans=numpy.zeros([256,256,4])
    ans[127,127,0]=1
    ans[:,:,1]=me.isCarryingMinerals()
    return pickle.dumps(ans)
def unravel(data):
    return pickle.loads(data)
def move(unit,cmd):
    global minerals
    if(cmd[0]=='gather'):
        unit.gather(minerals[cmd[1]])
    else:
        unit.returnCargo()
class HelloAI(BaseAI):
    def prepare(self):
        global minerals
        units=game.getAllUnits()
        for u in units:
            if(u.getType().getName()=='Resource_Mineral_Field'):
                minerals.append(u)
    def frame(self):
        units=game.getAllUnits()
        regions=game.getAllRegions()
        for u in units:
            if(u.getType().getName()=='Terran_SCV'):
                k=wrapup(units,regions,u)
                soc.send(k)
                k=soc.recv(10240)
                print('yo'+str(u.getPosition()))
                k=unravel(k)
                move(u,k)
        #sleep(0.5)

if __name__ == '__main__':
    run(HelloAI)
