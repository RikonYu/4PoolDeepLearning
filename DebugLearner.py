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
from Learners import Learner
class DebugLearner(Learner):
    def __init__(self, *args):
        super(DebugLearner, self).__init__(*args)
        self.train_err=[]

        self.ferr=open('trainerr.txt','w')
    def exploiter(self):
        return
    def controller(self, con, is_first):
        rl=self.lock.genRlock()
        wl=self.lock.genWlock()
        while(True):
            try:
                data = pickle.loads(util64.recv_msg(con))
                if (data.type == 'reg'):
                    self.init_episode(data)
                    con.send(b'ok')
                    break
                else:
                    msg = data.msg
                    pos=0
                    X=self.units.msg2state(self.mapSet.find_map(self.mapName), msg)
                    mask=self.units.msg2mask(self.mapSet.find_map(self.mapName), msg)
                    if(data.type=='terminal'):
                        self.units.predict_ans_masked(X,mask,True)
                        break
                    for i in msg.resources:
                        if(i.type=='Resource_Vespene_Geyser'):
                            pos=i.coord
                    pos[0]=pos[0]-msg.myInfo.coord[0]+WINDOW_SIZE//2
                    pos[1] = pos[1] - msg.myInfo.coord[1] + WINDOW_SIZE // 2
                    places=numpy.nonzero(mask)
                    #ans=numpy.random.choice(len(places))
                    #util64.send_msg(con, pickle.dumps([places[0][ans], places[1][ans], places[2][ans]]))
                    ans=[256,256,0]
                    util64.send_msg(con, pickle.dumps(ans))
                    if(is_first==1):
                        Y=numpy.zeros([WINDOW_SIZE,WINDOW_SIZE,self.units._out_channel])
                        for ind,_ in numpy.ndenumerate(Y[:,:,1]):

                            Y[ind[0],ind[1],1]=-numpy.linalg.norm(numpy.array(ind) - pos)/256.0
                        '''
                        ftarget=open('target.txt','wb')
                        pickle.dump(Y, ftarget)
                        ftarget.close()
                        '''
                        history=self.units.train(X.reshape([-1, WINDOW_SIZE, WINDOW_SIZE,self.units._in_channel]), Y.reshape([-1,WINDOW_SIZE,WINDOW_SIZE,self.units._out_channel]))
                        self.ferr.write(str(history.history['loss'][0])+'\n')
                        self.ferr.flush()
                        os.fsync(self.ferr.fileno())
            except ConnectionError:
                break