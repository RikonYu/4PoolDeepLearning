import util64
import pickle
import time
import numpy
import os
import ReplayBuffer
from keras import backend as KB
import tensorflow as tf
from ClassConstr import getUnitClass
import threading
from consts import WINDOW_SIZE
from readerwriterlock import RWLock

class A3C:
    def __init__(self, epsilon, discount, exploration_weight, batch_size):
        self.epsilon=epsilon
        self.discount=discount
        self.batch_size=batch_size
        self.exploration_weight=exploration_weight
        self.Q=None
        self.policy=None
        self.mapSet=util64.Maps()
        self.lock=RWLock.RWLockWrite()
        self.learn_epoch=0
        self.target_type=''
        self.agent=0

    def init_episode(self, k):
        if (self.mapSet.is_empty()):
            self.mapSet.add_map(util64.gameMap(k[1], k[3]))
            self.targetType = k[2]
            self.Q = getUnitClass(self.targetType, True)
            self.policy = getUnitClass(self.targetType, True)
            self.tempd.set_weights(self.units.get_weights())
        elif (self.mapSet.find_map(k[3]) is None):
            self.mapSet.add_map(util64.gameMap(k[1], k[3]))
            print('new map: ', k[3])
        self.mapName = k[3]
        self.agent_no = 1

    def controller(self, con, is_first):
        rl=self.lock.genRlock()
        wl=self.lock.genWlock()
        gradients=0
        step=0
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
                    X=self.policy.msg2state(self.mapSet.find_map(self.mapName),data[1])
                    mask=self.policy.msg2mask(self.mapSet.find_map(self.mapName),data[1])
                    rl.acquire()
                    act=self.policy.sample_ans(X,mask)
                    rl.release()
                    util64.send_msg(con,pickle.dumps(act))
                    if(last_state is not None):
                        if(data[0]=='terminal'):
                            memory.append([last_state,last_act,last_state, 0, data[2]])
                        else:
                            memory.append([last_state,last_act,data[1], 1, data[2]])
            except EOFError:
                #update gradient
                R=0
                for i in memory:
                    R=i[4]+self.discount*R
                    gradient=KB.gradients(self.policy.out,self.policy.model.trainable_weights)
                    gradients=0
                break