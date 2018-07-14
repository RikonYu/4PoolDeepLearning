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

class A2C:
    def __init__(self, epsilon, discount, exploration_weight, batch_size):
        self.epsilon=epsilon
        self.discount=discount
        self.batch_size=batch_size
        self.exploration_weight=exploration_weight
        self.Q=None
        self.policy=None
        self.advantage=None
        self.learn_epoch=0
        self.target_type=''

    def controller(self, con, is_first):
        pass