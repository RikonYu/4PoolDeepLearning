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
class Learner:
    def __init__(self, epsilon,discount, exploration_weight,batch_size):
        self.lock=RWLock.RWLockWrite()
        self.agent_no=0
        self.batch_size=batch_size
        self.mapSet=util64.Maps()
        self.mapName=''
        self.learn_epoch=0
        self.epsilon=epsilon
        self.discount=discount
        self.targetType=''
        self.exploration_weight=exploration_weight
    def learner(self):
        return
    def controller(self, con, is_first):
        return
    def init_episode(self, data):
        if (self.mapSet.is_empty()):
            self.mapSet.add_map(util64.gameMap(data.msg, data.mapName))
            self.targetType = data.unitType
            self.units = getUnitClass(self.targetType, True)
            self.target = getUnitClass(self.targetType, True)
            self.tempd = getUnitClass(self.targetType, True)
        elif (self.mapSet.find_map(data.mapName) is None):
            self.mapSet.add_map(util64.gameMap(data.msg, data.mapName))
        self.mapName = data.mapName
        self.agent_no = 1