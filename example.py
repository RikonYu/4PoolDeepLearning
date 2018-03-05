from time import sleep
from pybrood import BaseAI, run, game, Color
import keras
import numpy
import pickle
import theano
from scipy import misc
from keras.models import Sequential, Model, load_model
from keras.layers import Input,Concatenate
from keras.layers import Reshape,Dense, Dropout, Embedding, LSTM,Flatten,Conv2D,MaxPooling2D
from keras.optimizers import Adam
import matplotlib.pyplot as plt

theano.config.optimizer='None'
def inCenter(x,y,top,bot,left,right):
    if(bot<x-128):return False
    if(x+128<top):return False
    if(right<y-128):return False
    if(y+128<left):return False
    return True
minerals=[]
done=1
class NaiveMiningNet:
    
    def __init__(self,output_num):
        self.model=Sequential()
        self.model.add(Conv2D(32,(3,3),activation='relu',padding='same',input_shape=(256,256,4)))
        #self.model.add(Conv2D(128,(3,3),activation='relu',padding='same'))
        self.model.add(MaxPooling2D((32,32)))
        #self.model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
        #self.model.add(MaxPooling2D((2,2)))
        #self.model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
        #self.model.add(MaxPooling2D((4,4)))
        
        self.model.add(Flatten())
        self.model.add(Dense(32,activation='relu'))
        self.model.add(Dense(output_num,activation='softmax'))
        self.model.compile(loss='categorical_crossentropy',optimizer='adam')
    
    def predict(self,X):
        k=self.model.predict(X)
        return numpy.argmax(k,axis=1)
    
    @staticmethod
    def MakeInput(units,regions,unitMe):
        global minerals
        minerals=[]
        cx=unitMe.getPosition()[0]
        cy=unitMe.getPosition()[1]
        rmaps=numpy.zeros([256,256])        
        umaps=numpy.zeros([256,256])
        mmaps=numpy.zeros([256,256])
        ans=numpy.zeros([1,256,256,4])
        for u in units:
            '''
            top=u.getTop()
            bot=u.getBottom()
            left=u.getLeft()
            right=u.getRight()
            if(inCenter(cx,cy,top,bot,left,right)):
                #ans[]=(u.getType().getName()=='Terran_SCV')
                #mmaps[u.getTop():u.getBottom(),u.getLeft():u.getRight()]=(u.getType().getName()=='Resource_Mineral_Field')
            '''
            if(u.getType().getName()=='Resource_Mineral_Field'):
                minerals.append(u)
        #for r in regions:
        #    rmaps[r.getBoundsTop():r.getBoundsBottom(),r.getBoundsLeft():r.getBoundsRight()]=r.isAccessible()
        ans[0,127-cx+unitMe.getTop():127-cx+unitMe.getBottom(),127-cy+unitMe.getLeft():127-cy+unitMe.getRight(),3]=1
        
        return ans

class HelloAI(BaseAI):
    def prepare(self):
        force = game.getForce(0)
        print(force)
        print(force.getID())
        print(force.getName())
        units = game.getAllUnits()
        mcount=0
        for u in units:
            if(u.getType().getName()=='Resource_Mineral_Field'):
                mcount+=1
        #print(mcount)s
            
        self.MiningNet=NaiveMiningNet(mcount)
    def frame(self):
        global minerals
        units=game.getAllUnits()
        regions=game.getAllRegions()
        for u in units:
            if(u.getType().getName()=='Terran_SCV'):
                if(u.isCarryingMinerals()==1):
                    u.returnCargo()
                else:
                    
                    ans=self.MiningNet.predict(NaiveMiningNet.MakeInput(units,regions,u))[0]
                    print("infering",len(minerals))
                    u.gather(minerals[ans])
        sleep(0.05)
        #print(game.elapsedTime())
        #game.restartGame()


if __name__ == '__main__':
    run(HelloAI)
