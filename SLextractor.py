import numpy
import pybrood
import os
import shutil
import glob
import util32
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import random
from pybrood import BaseAI, run, game, Color
#from DroneNet import DroneNet
newpath='E:\\sc\\1.16\\maps\\replays'
mypath='E:\\sc\\replays\\pros\\By.herO 2016'
playerMe=None
qqq=0
#dronemoves={}
lastorders={}
X=[]
Y=[]
allrep=[]
for i in os.walk(mypath):
    for j in i[2]:
        allrep.append([i[0],j])
#print(allrep)
repCount=0

def newOrder(u):
    if(u.getID() in lastorders):
        if(lastorders[u.getID()][0]==u.getOrder().getID() and lastorders[u.getID()][1]==u.getOrderTargetPosition()):
            return False
    lastorders[u.getID()]=[u.getOrder().getID(), u.getOrderTargetPosition()]
    return True
class FourPoolAISupervised(BaseAI):
    def prepare(self):
        global playerMe
        self.fout=open('Drone%s.txt'%game.mapFileName()[:-4],'wb')
        playerMe=game.self()
        game.setLocalSpeed(0)
        util32.init()
        no_zerg=1
        for i in game.getPlayers():
            if(i.getRace().getName()=='Zerg' and len(list(i.getUnits()))==9):
                no_zerg=0
        if(no_zerg==1):
            game.leaveGame()
        else:
            pickle.dump(util32.reg2msg(),self.fout)
    def frame(self):
        if(game.isReplay()==True):
            for i in game.getAllUnits():
                if(i.getType().getName()=='Zerg_Drone'):
                    #if(random.randint(1,5)!=1):
                    #    continue
                    if(newOrder(i)):
                        mz=util32.ord2cmd(i.getOrder().getID())
                        '''
                        if(mz==0):
                            if(random.randint(1,200)<=199):
                                continue
                        elif(mz==1 or mz==3 or mz==5):
                            if(random.randint(1,10)<=9):
                                continue
                        '''
                            
                        #x=util32.game2stateDrone(i)
                        x=util32.game2msgDrone(i)
                        y=[i.getOrderTargetPosition()[0]+180-i.getPosition()[0],
                           i.getOrderTargetPosition()[1]+180-i.getPosition()[1],mz]

                        pickle.dump(x,self.fout)
                        pickle.dump(y,self.fout)
        
    def finished(self):
        global X,Y,repCount
        print('finished')
        #pickle.dump([X,Y],fout)
        self.fout.close()
        X=[]
        Y=[]
        print("%s finished, opening next"%game.mapFileName())
        os.remove(newpath+'\\'+allrep[repCount][1])
        repCount+=1
        shutil.copyfile(allrep[repCount][0]+'\\'+allrep[repCount][1],newpath+'\\'+allrep[repCount][1])
        #game.setMap('E:\\sc\\1.16\\maps\\replays\\%d.rep'%repCount)
        #game.restartGame()
        util32.navigate()

if __name__ == '__main__':
    print(allrep[0])
    shutil.copyfile(allrep[0][0]+'\\'+allrep[0][1],newpath+'\\'+allrep[0][1])
    run(FourPoolAISupervised)
