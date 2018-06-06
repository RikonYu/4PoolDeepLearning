from time import sleep
from pybrood import BaseAI, run, game, Color, UnitTypes
from pybrood import CoordinateType as CType
import pickle
from scipy import misc
import numpy
import matplotlib.pyplot as plt
import socket,os
from scipy.spatial import distance


soc=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
soc.connect(('127.0.0.1',6666))
minerals=[]
cframe=0
playerMe=None
move_every=20
plt.ion()
#field:terrain,unit,mypos,hasmineral
#mask:gather,return,move,?

#units' coordinates,
def wrap_box(unit):
    return [unit.getPosition(),unit.getLeft(),unit.getTop(),unit.getRight(),unit.getBottom()]
def wrapup(units,regions,me,curFrame,reward,lastCmd):
    return pickle.dumps([[wrap_box(u) for u in units],wrap_box(me),me.isCarryingMinerals(),curFrame,reward,lastCmd])
def unravel(data):
    return pickle.loads(data)
def move(unit,cmd):
    global minerals
    center=unit.getPosition()
    #print('%s from %d,%d to %d,%d do %d prv %s'%(unit.getType().getName(),center[0],center[1],cmd[0]+center[0]-180,cmd[1]+center[1]-180,cmd[2],unit.getOrder().getName()))
    if(cmd[2]==0):#gather
        if(unit.getOrder().getName()!='MiningMinerals'):
            unit.gather(game.getClosestUnit(([cmd[0]+center[0]-180,cmd[1]+center[1]-180])))
    elif(cmd[2]==1):#return
        unit.returnCargo()
    elif(cmd[2]==2):#move
        unit.move([cmd[0]+center[0]-180,cmd[1]+center[1]-180])
    elif(cmd[2]==3):#do nothing
        pass
typeSCV=UnitTypes.Terran_SCV

trace=[]
class HelloAI(BaseAI):
    def prepare(self):
        self.playerMe=game.self()
        global minerals
        units=game.getAllUnits()
        minerals=[]
        for u in units:
            if(u.getType().getName()=='Resource_Mineral_Field'):
                minerals.append(u)
    def frame(self):
        #return
        curFrame=game.getFrameCount()
        units=game.getAllUnits()
        regions=game.getAllRegions()
        if(curFrame>=3600):
            trace.append(self.playerMe.gatheredMinerals()-50)
            print(trace)
            game.restartGame()
        if(curFrame%move_every!=0):
            return
        for u in units:
            if(u.getType().getName()=='Terran_SCV' and u.getPlayer().getID()==self.playerMe.getID()):
                k=wrapup(units,regions,u,curFrame//move_every,self.playerMe.gatheredMinerals()-50,u.getOrder().getName())
                soc.send(k)
                k=b''
                k=soc.recv(10240)
                k=unravel(k)
                move(u,k)
            elif(u.getType().getName()=='Terran_Command_Center'):
                #print('minerals:%d'%self.playerMe.minerals(),typeSCV)
                if(self.playerMe.minerals()>=50 and u.canTrain(typeSCV)):
                    u.train(typeSCV)
        #sleep(0.5)

if __name__ == '__main__':
    run(HelloAI)
