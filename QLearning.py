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
# Deep Q Learning
from Learners import Learner
class QLearning(Learner):
    def __init__(self, *args):
        super(QLearning, self).__init__(*args)
        self.units=None
        self.tempd=None
        self.target=None
        self.explore_count=0
        self.exploit_count=0
        self.async=0
        self.buflock=threading.Semaphore(1)
        self.buf = ReplayBuffer.PriortizedReplayBuffer(100000,self.buflock)
        self.learn_epoch=0

    def learner(self):
        replace_every = 10
        train_every = 64
        wl = self.lock.genWlock()
        while (True):
            if(self.async==1):
                return
            if (self.buf.count < train_every):
                # print('not enough samples')
                time.sleep(10)
                continue
            samples, indx, bias = self.buf.sample(self.batch_size)
            #samples = self.buf.sample(self.batch_sacquireize)
            print('training')
            self.tempd.set_weights(self.units.get_weights())
            X = numpy.array([self.units.msg2state(self.mapSet.find_map(map_name), i) for i, _act, _nextS, _reward, _is_terminal,map_name in samples])
            Y = self.units.predict_all(X)  # Q(s,a)
            aprime = [self.target.predict_max_masked(self.units.msg2state(self.mapSet.find_map(map_name), i), self.units.msg2mask(self.mapSet.find_map(map_name),i))
                      for _state, _act, i, _reward, _is_terminal,map_name in samples]  # max_aQ'(s',a')
            Y_ = [(samples[i][3] + self.discount * aprime[i] * (1 - samples[i][4])) for i in
                  range(self.batch_size)]  # r+discount*max_aq'(s',a')
            diff = numpy.copy(Y)
            self.buf.update(indx,
                       list(Y_[i] - Y[i, samples[i][1][0], samples[i][1][1], samples[i][1][2]] for i in range(self.batch_size)))


            for i in range(self.batch_size):
                if(samples[i][3]!=0):
                    print(Y_[i], aprime[i], samples[i][3])
                diff[i, samples[i][1][0], samples[i][1][1], samples[i][1][2]] += (Y_[i]-diff[i, samples[i][1][0], samples[i][1][1], samples[i][1][2]])

            self.tempd.train(X, diff)
            self.tempd.save()
            if (self.learn_epoch % replace_every == 0):
                self.target.set_weights(self.tempd.get_weights())
            wl.acquire()
            self.units.set_weights(self.tempd.get_weights())
            wl.release()
            self.learn_epoch += 1
    def drawer(self, con, is_first):
        while (True):
            try:
                data = util64.recv_msg(con)
                k = pickle.loads(data)
                if (k.type == 'reg'):
                    self.init_episode(k)
                    con.send(b'ok')
                    break
                else:
                    if(is_first!=1):
                        return
                    msg=k.msg
                    X = self.units.msg2state(self.mapSet.find_map(self.mapName), msg)
                    mask = self.units.msg2mask(self.mapSet.find_map(self.mapName), msg)
                    if(msg.myInfo.killCount==1):
                        time.sleep(100)
                    util64.send_msg(con, pickle.dumps([256,256,0]))

                    #ans = self.units.predict_ans_masked(X, mask, is_first == 1)
                    #util64.send_msg(con, pickle.dumps(ans))
            except EOFError:
                break
    def exploiter(self, con, is_first):
        while (True):
            try:
                data = util64.recv_msg(con)
                k = pickle.loads(data)
                if (k.type == 'reg'):
                    self.init_episode(k)
                    con.send(b'ok')
                    break
                else:
                    msg=k.msg
                    X = self.units.msg2state(self.mapSet.find_map(self.mapName), msg)
                    mask = self.units.msg2mask(self.mapSet.find_map(self.mapName), msg)
                    ans = self.units.predict_ans_masked(X, mask, is_first == 1)
                    if (is_first == 1):
                        print('exploiting', ans[0], ans[1])
                        ans=ans[0]
                    util64.send_msg(con, pickle.dumps(ans))
            except EOFError:
                break
    def controller(self, con, is_first):
        last_state = None
        last_action = None
        last_value = 0
        visited = numpy.zeros([1, 1])
        unvisited = 0
        rl = self.lock.genRlock()
        feval = 0
        fq = 0
        pSticky=0.2
        if (is_first == 1):
            feval = open('rewards.txt', 'a')
            fq = open('Qvals.txt', 'a')
        while (True):
            try:
                data = util64.recv_msg(con)
                k = pickle.loads(data)
                if (k.type == 'reg'):
                    print(self.explore_count,self.exploit_count)
                    self.init_episode(k)
                    con.send(b'ok')
                    break
                else:
                    msg=k.msg
                    X = self.units.msg2state(self.mapSet.find_map(self.mapName), msg)
                    if (k.type == 'terminal' and last_action is not None):
                        self.buf.add(last_state, last_action, last_state, (k.value - self.exploration_weight * unvisited - last_value),
                                1, self.mapName)
                        if (is_first == 1):
                            feval.write(str(k.value) + '\n')
                            feval.flush()
                            os.fsync(feval.fileno())
                            self.epsilon*=0.98
                            print(self.epsilon)
                        break
                    if (visited.shape[0] == 1):
                        visited = numpy.zeros(self.mapSet.find_map(self.mapName).regions.shape)
                        unvisited = visited.shape[0] * visited.shape[1]
                        last_value = -self.exploration_weight * unvisited
                    # print(k)
                    visited[msg.myInfo.coord[0], msg.myInfo.coord[1]] += 1
                    if (visited[msg.myInfo.coord[0], msg.myInfo.coord[1]] == 1):
                        unvisited -= 1
                    if (numpy.random.random() < self.epsilon):
                        self.explore_count += 1
                        if(numpy.random.random()<pSticky):
                            print('sticky')
                            ans=[256,256,-1]
                        else:
                            places = self.units.msg2mask(self.mapSet.find_map(self.mapName), msg)
                            _,_, ink = numpy.nonzero(places)
                            moveType=numpy.random.choice(numpy.unique(ink))
                            ini,inj=numpy.nonzero(places[:,:,moveType])
                            ind = numpy.random.choice(len(ini))
                            ans = [ini[ind], inj[ind], moveType]
                            if (is_first == 1):
                                print('exploring', ans)
                                fq.write('None\n')
                                fq.flush()
                                os.fsync(fq.fileno())
                        # print(ans)
                    else:
                        mask = self.units.msg2mask(self.mapSet.find_map(self.mapName), msg)
                        rl.acquire()
                        ans = self.units.predict_ans_masked(X, mask, is_first == 1)
                        rl.release()
                        if (is_first == 1):
                            print('exploiting', ans[0], ans[1])
                            self.exploit_count+=1
                            fq.write(str(ans[1]) + '\n')
                            fq.flush()
                            os.fsync(fq.fileno())
                            ans = ans[0]
                    util64.send_msg(con, pickle.dumps(ans))
                    if (last_action is not None):
                        self.buf.add(last_state, last_action, msg,
                                (k.value - self.exploration_weight * unvisited - last_value), 0, self.mapName)
                    last_state = msg
                    last_action = ans
                    last_value = k.value - self.exploration_weight * unvisited

            except EOFError:
                print('exception found')
                break
        if (is_first == 1):
            feval.close()
            fq.close()

    def asyncController(self, con, is_first):
        ETARGET=150
        EUPDATE=50
        epoch=0
        self.async=1
        last_state=None
        last_action=None
        last_value=0
        gradients=[]
        epsilon=self.epsilon*numpy.random.uniform(0.8,1.2)
        while(True):
            data=pickle.loads(util64.recv_msg(con))
            if(data.type=='reg'):
                self.init_episode(data)
                con.send(b'ok')
                break
            else:
                msg=data.msg
                if(data.type=='terminal'):
                    #apply gradients
                    pass
                X=self.units.msg2state(self.mapSet.find_map(self.mapName), msg)
                places = self.units.msg2mask(self.mapSet.find_map(self.mapName), msg)
                if(numpy.random.random()<epsilon):
                    ini, inj, ink = numpy.nonzero(places)
                    ind = numpy.random.choice(len(ini))
                    ans = [ini[ind], inj[ind], ink[ind]]
                else:
                    ans = self.units.predict_ans_masked(X, places)
                util64.send_msg(con,pickle.dumps(ans))
                maxNext=self.target.predict_max_masked(X, places)
                if(last_state is not None):
                    Y=self.units.predict_all(self.units.msg2state(self.mapSet.find_map(self.mapName), last_state))
                    Y_=numpy.copy(Y)
                    Y_[0,last_action[0],last_action[1], last_action[2]]=data.value-last_value+self.discount*maxNext
                    gradient=self.units.gradient(Y_, self.units.msg2state(self.mapSet.find_map(self.mapName), last_state))
                    print([i.eval() for i in gradient])
                    gradients+=[gradient]
                last_state=msg
                last_action=ans
                last_value=data.value
                self.learn_epoch+=1
                epoch+=1
                if(self.learn_epoch % ETARGET == 0):
                    self.target.set_weights(self.units.get_weights())
                if(epoch % EUPDATE == 0):
                    self.units.apply_gradient(gradients)
                    gradients=0




