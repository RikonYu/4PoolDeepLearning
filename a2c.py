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
        self.units=None
        self.mapSet=util64.Maps()
        self.lock=RWLock.RWLockWrite()
        self.learn_epoch=0
        self.target_type=''
        self.agent_no=0
        self.memory=[]
        self.memory_map=[]

    def init_episode(self, k):
        if (self.mapSet.is_empty()):
            self.mapSet.add_map(util64.gameMap(k.msg, k.mapName))
            self.target_type = k.unitType
            self.actor = getUnitClass(self.target_type, False,'softmax')
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
            tactor = getUnitClass(self.target_type, False, 'softmax')
            tactor.set_weights(self.actor.get_weights())
            tcritic = ValueNetwork(self.tactor._in_channel)
            tcritic.set_weights(self.critic.get_weights())
            values=[]
            cmem=self.memory[0]
            for i in cmem:
                values.append(self.critic.predict(self.actor.msg2state(i[0])))
            values.append(0)
            for i in range(len(cmem)):
                advantages=numpy.zeros([1,WINDOW_SIZE,WINDOW_SIZE,self.actor._in_channel])
                advantages[0][tuple(cmem[i][1])]=cmem[i][3]+self.discount*values[i+1]-values[i]
                targets=numpy.zeros([1,1])
                targets[0][0]=cmem[i][3]+self.discount*values[i+1]
                tactor.train([self.actor.msg2state(self.mapSet.find_map(self.memory_map[0]),cmem[i][0])],advantages)
                tcritic.train_batch([self.actor.msg2state(self.mapSet.find_map(self.memory_map[0]),cmem[i][0])], targets)
            tactor.save()
            tcritic.save()
            wl.acquire()
            self.actor.set_weights(tactor.get_weights())
            self.critic.set_weights(tcritic.get_weights())
            wl.release()
            self.memory.pop(0)
    def controller(self, con, is_first):
        rl=self.lock.genRlock()
        last_state=None
        last_act=None
        memory=[]
        fval=None
        frwd=None
        last_val=None
        if(is_first==1):
            fval=open('cVal.txt','a')
            frwd=open('reward.txt','a')
        while(True):
            try:
                data=pickle.loads(util64.recv_msg(con))
                if(data.type ==  'reg'):
                    self.init_episode(data)
                    con.send(b'ok')
                    break
                else:
                    X=self.actor.msg2state(self.mapSet.find_map(self.mapName),data.msg)
                    mask=self.actor.msg2mask(self.mapSet.find_map(self.mapName),data.msg)
                    rl.acquire()
                    act=self.actor.sample_ans_masked(X,mask)
                    if(is_first==1 and last_val is not None):
                        print(self.critic.predict([X]), data.value)
                        fval.write(str(self.critic.predict([X])[0,0])+'\n')
                        frwd.write(str(data.value-last_val)+'\n')
                    rl.release()
                    util64.send_msg(con,pickle.dumps(act))
                    if(last_state is not None):
                        if(data.type=='terminal'):
                            memory.append([last_state,last_act,last_state, 0, data.value])
                        else:
                            memory.append([last_state,last_act,data.msg, 1, data.value])
                    last_val=data.value
                    last_state=data.msg
                    last_act=act
            except:
                self.memory.append(memory)
                self.memory_map.append(self.mapName)
                break