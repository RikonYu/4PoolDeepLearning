import util64
import pickle
import time
import numpy
from ClassConstr import getUnitClass
from consts import WINDOW_SIZE
from readerwriterlock import RWLock
from UnitNet import ValueNetwork

class A2C:
    def __init__(self, epsilon, discount, exploration_weight, batch_size):
        self.epsilon=epsilon
        self.discount=discount
        self.batch_size=batch_size
        self.exploration_weight=exploration_weight
        self.actor=None
        self.critic=None
        self.tempd=None
        self.units=None
        self.mapSet=util64.Maps()
        self.lock=RWLock.RWLockWrite()
        self.learn_epoch=0
        self.target_type=''
        self.agent=0
        self.memory=[]

    def init_episode(self, k):
        if (self.mapSet.is_empty()):
            self.mapSet.add_map(util64.gameMap(k.msg, k.mapName))
            self.targetType = k.unitType
            self.actor = getUnitClass(self.targetType, True,'softmax')
            self.tempd = getUnitClass(self.targetType, True,'softmax')
            self.critic=ValueNetwork(self.actor._in_channel)
        elif (self.mapSet.find_map(k.mapName) is None):
            self.mapSet.add_map(util64.gameMap(k.msg, k.mapName))
        self.mapName = k.mapName
        self.agent_no = 1
    def learner(self):
        wl=self.lock.genWlock()
        while(True):
            if(len(self.memory)==0):
                time.sleep(5)
                continue
            values=[]
            targets=[]
            for i in self.memory:
                values.append(self.critic.predict(self.actor.msg2state(i[0])))
            values.append(0)
            for i in range(len(self.memory)):
                advantages=numpy.zeros([1,WINDOW_SIZE,WINDOW_SIZE,self.actor._in_channel])
                advantages[0][tuple(self.memory[i][1])]=self.memory[i][3]+self.discount*values[i+1]-values[i]
                targets=numpy.zeros([1,1])
                targets[0][0]=self.memory[i][3]+self.discount*values[i+1]
                wl.acquire()
                self.actor.train([self.actor.msg2state(self.memory[i][1])],advantages)
                self.critic.train_batch([self.actor.msg2state(self.memory[i][1])], targets)
                wl.release()
            self.actor.save()
            self.critic.save()
    def controller(self, con, is_first):
        rl=self.lock.genRlock()
        last_state=None
        last_act=None
        memory=[]
        while(True):
            try:
                data=pickle.load(util64.recv_msg(con))
                if(data[0] ==  'reg'):
                    self.init_episode(data)
                    con.send(b'ok')
                    break
                else:
                    X=self.actor.msg2state(self.mapSet.find_map(self.mapName),data[1])
                    mask=self.actor.msg2mask(self.mapSet.find_map(self.mapName),data[1])
                    rl.acquire()
                    act=self.actor.sample_ans(X,mask)
                    rl.release()
                    util64.send_msg(con,pickle.dumps(act))
                    if(last_state is not None):
                        if(data[0]=='terminal'):
                            memory.append([last_state,last_act,last_state, 0, data[2]])
                        else:
                            memory.append([last_state,last_act,data[1], 1, data[2]])
            except EOFError:
                self.memory=memory
                break