import util64
import pickle
import time
import numpy
from ClassConstr import getUnitClass
from consts import WINDOW_SIZE
from readerwriterlock import RWLock
from UnitNet import ValueNetwork
import os
from Learners import Learner

class A2C(Learner):
    def __init__(self, *args):
        super(A2C, self).__init__(*args)
        self.actor=None
        self.critic=None
        self.units=None
        self.memory=[]
        self.memory_map=[]

    def init_episode(self, k):
        if (self.mapSet.is_empty()):
            self.mapSet.add_map(util64.gameMap(k.msg, k.mapName))
            self.target_type = k.unitType
            self.actor = getUnitClass(self.target_type, False,'softmax')
            self.tactor=getUnitClass(self.target_type, False, 'softmax')
            self.tcritic=ValueNetwork(self.actor._in_channel)
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
            print('training')
            self.tactor.set_weights(self.actor.get_weights())
            self.tcritic.set_weights(self.critic.get_weights())
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
                self.tactor.train([self.actor.msg2state(self.mapSet.find_map(self.memory_map[0]),cmem[i][0])],advantages)
                self.tcritic.train_batch([self.actor.msg2state(self.mapSet.find_map(self.memory_map[0]),cmem[i][0])], targets)
            self.tactor.save()
            self.tcritic.save()
            wl.acquire()
            self.actor.set_weights(self.tactor.get_weights())
            self.critic.set_weights(self.tcritic.get_weights())
            self.memory.pop(0)
            self.memory_map.pop(0)
            wl.release()

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
                    rl.release()
                    if(is_first==1 and last_val is not None):
                        #print(act, self.critic.predict([X])[0,0], data.value)
                        fval.write(str(self.critic.predict([X])[0,0])+'\n')

                        fval.flush()
                        os.fsync(fval.fileno())
                    util64.send_msg(con,pickle.dumps(act))
                    if(last_state is not None):
                        if(data.type=='terminal'):
                            memory.append([last_state,last_act,last_state, 0, data.value])
                            if(is_first==1):
                                frwd.write(str(data.value) + '\n')
                                frwd.flush()
                                os.fsync(frwd.fileno())
                            return
                        else:
                            memory.append([last_state,last_act,data.msg, 1, data.value])
                    last_val=data.value
                    last_state=data.msg
                    last_act=act
            except EOFError:
                rl.acquire()
                self.memory.append(memory)
                self.memory_map.append(self.mapName)
                rl.release()
                return