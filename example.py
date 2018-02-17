from time import sleep
from pybrood import BaseAI, run, game, Color
import keras
import numpy
from scipy import misc
from keras.models import Sequential, Model, load_model
from keras.layers import Input,Concatenate
from keras.layers import Reshape,Dense, Dropout, Embedding, LSTM,Flatten,Conv2D,MaxPooling2D
from keras.optimizers import Adam
import matplotlib.pyplot as plt

class NaiveMiningNet:
    
    def __init__(self,output_num):
        self.model=Sequential()
        self.model.add(Conv2D(256,(3,3),activation='relu',padding='same',input_shape=(4096,4096,4)))
        self.model.add(Conv2D(256,(3,3),activation='relu',padding='same'))
        self.model.add(MaxPooling2D((4,4)))
        self.model.add(Conv2D(256,(3,3),activation='relu',padding='same'))
        self.model.add(MaxPooling2D((4,4)))
        self.model.add(Conv2D(256,(3,3),activation='relu',padding='same'))
        self.model.add(MaxPooling2D((4,4)))
        self.model.add(Flatten())
        print(self.model.output_shape)
        self.model.add(Dense(128,activation='relu'))
        self.model.add(Dense(output_num,activation='softmax'))
        self.model.compile(loss='categorical_crossentropy',optimizer='adam')
    
    def infer(self,X):
        k=self.model.predict(X)
        return numpy.argmax(k,axis=1)
    
    @staticmethod
    def MakeInput(units,regions,unitMe):
        rmaps=numpy.zeros([4097,4097])        
        umaps=numpy.zeros([4097,4097])
        mmaps=numpy.zeros([4097,4097])
        minerals=[]
        for u in units:
            umaps[u.getTop():u.getBottom(),u.getLeft():u.getRight()]=(u.getType().getName()=='Terran_SCV')
            mmaps[u.getTop():u.getBottom(),u.getLeft():u.getRight()]=(u.getType().getName()=='Resource_Mineral_Field')
            if(u.getType().getName()=='Resource_Mineral_Field'):
                minerals.append(u)
        for r in regions:
            rmaps[r.getBoundsTop():r.getBoundsBottom(),r.getBoundsLeft():r.getBoundsRight()]=r.isAccessible()
        ans=numpy.zeros([4096,4096,4])
        ans[:,:,0]=rmaps[1:,1:]
        ans[:,:,1]=umaps[1:,1:]
        ans[:,:,2]=mmaps[1:,1:]
        ans[unitMe.getTop():unitMe.getBottom(),unitMe.getLeft():unitMe.getRight(),3]=1
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
        print(mcount)
            
        self.MiningNet=NaiveMiningNet(mcount)
    def frame(self):
        units=game.getAllUnits()
        regions=game.getAllRegions()
        for u in units:
            if(u.getName()=='Terran_SCV'):
                if(u.isCarryingMinerals()==1):
                    u.returnCargo()
                else:
                    u.gather(self.MiningNet.predict(MiningNet.MakeInput(units,regions,u)))
        sleep(1)
        #print(game.elapsedTime())
        #game.restartGame()


if __name__ == '__main__':
    run(HelloAI)
