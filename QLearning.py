import util64
import pickle
import time
import numpy
import os
import ReplayBuffer
from ClassConstr import getUnitClass
import threading
from consts import WINDOW_SIZE
from readerwriterlock import RWLock
# Deep Q Learning
class QLearning:
    def __init__(self, epsilon,discount, exploration_weight,batch_size):
        self.lock=RWLock.RWLockWrite()
        self.agent_no=0
        self.units=None
        self.tempd=None
        self.target=None
        self.batch_size=batch_size
        self.disGame = None
        self.mapName=''
        self.learn_epoch=0
        self.buflock=threading.Semaphore(1)
        self.buf = ReplayBuffer.PriortizedReplayBuffer(50000)
        self.epsilon=epsilon
        self.discount=discount
        self.targetType=''
        self.exploration_weight=exploration_weight

    def learner(self):
        replace_every = 50
        train_every = 64
        wl = self.lock.genWlock()
        while (True):
            if (self.buf.count < train_every):
                # print('not enough samples')
                time.sleep(10)
                continue
            self.buflock.acquire()
            samples, indx, bias = self.buf.sample(self.batch_size)
            self.buflock.release()
            print('training')
            self.tempd.set_weights(self.units.get_weights())
            X = numpy.array([self.units.msg2state(self.disGame, i) for i, _a, _sp, _r, _it in samples])
            Y = self.units.predict_all(X)  # Q(s,a)
            # print(numpy.array([units.msg2state(disGame, i) for _s, _a, i, _r, _it in samples]).shape)
            aprime = self.target.predict_max(
                [self.units.msg2state(self.disGame, i) for _s, _a, i, _r, _it in samples])  # max_aQ'(s',a')
            Y_ = [(samples[i][3] + self.discount * aprime[i] * (1 - samples[i][4])) for i in
                  range(self.batch_size)]  # r+discount*max_aq'(s',a')
            diff = numpy.copy(Y)
            for i in range(self.batch_size):
                diff[i, samples[i][1][0], samples[i][1][1], samples[i][1][2]] = Y_[i]

            # not using bias for now
            self.buflock.acquire()
            self.buf.update(indx,
                       list(Y_[i] - Y[i, samples[i][1][0], samples[i][1][1], samples[i][1][2]] for i in range(self.batch_size)))
            self.buflock.release()
            self.tempd.train(X, diff)
            self.tempd.save()
            if (self.learn_epoch % replace_every == 0):
                self.target.set_weights(self.tempd.get_weights())
            wl.acquire()
            # print('acquired')
            self.units.set_weights(self.tempd.get_weights())
            wl.release()
            # print('released')
            self.buflock.acquire()
            self.buf.count -= train_every
            self.buflock.release()
            self.learn_epoch += 1


    def controller(self, con, is_first):
        last_state = None
        last_action = None
        last_value = 0
        visited = numpy.zeros([1, 1])
        unvisited = 0
        rl = self.lock.genRlock()
        feval = 0
        fq = 0
        if (is_first == 1):
            feval = open('rewards.txt', 'a')
            fq = open('Qvals.txt', 'a')
            # print('appending')
        while (True):
            try:
                data = util64.recv_msg(con)
                k = pickle.loads(data)
                if (k[0] == 'reg'):
                    if(self.disGame is None):
                        self.disGame=util64.gameInstance(k[1])
                        self.targetType = k[2]
                        self.units = getUnitClass(self.targetType, True)
                        self.target = getUnitClass(self.targetType, True)
                        self.tempd = getUnitClass(self.targetType, True)
                        self.tempd.set_weights(self.units.get_weights())
                    elif(self.mapName!=k[3]):
                        self.disGame(k[1])
                        print('new map')
                    else:
                        con.send(b'ok')
                        break
                    self.mapName=k[3]
                    self.agent_no = 1
                    con.send(b'ok')
                    break
                else:
                    print(self.disGame.regions.shape)
                    X = self.units.msg2state(self.disGame, k[1])
                    if (k[0] == 'terminal' and last_action is not None):
                        self.buflock.acquire()
                        self.buf.add(last_state, last_action, last_state, (k[2] - self.exploration_weight * unvisited + last_value),
                                1)
                        self.buflock.release()
                        break
                    if (visited.shape[0] == 1):
                        visited = numpy.zeros(self.disGame.regions.shape)
                        unvisited = visited.shape[0] * visited.shape[1]
                        last_value = -self.exploration_weight * unvisited
                    # print(k)
                    visited[k[1][0][0], k[1][0][1]] += 1
                    if (visited[k[1][0][0], k[1][0][1]] == 1):
                        unvisited -= 1
                    if (numpy.random.random() < self.epsilon):
                        places = self.units.msg2mask(self.disGame, k[1])
                        '''
                        probs = numpy.zeros([WINDOW_SIZE, WINDOW_SIZE])
                        x = k[1][0][0]
                        y = k[1][0][1]
                        ax = max(0, WINDOW_SIZE // 2 - x)
                        ay = max(0, WINDOW_SIZE // 2 - y)
                        probs[ax:min(WINDOW_SIZE, visited.shape[0] - x + WINDOW_SIZE // 2),
                        ay:min(WINDOW_SIZE, visited.shape[1] - y + WINDOW_SIZE // 2)] = numpy.exp(-visited[
                                                                                                   max(0,
                                                                                                       x - WINDOW_SIZE // 2):min(
                                                                                                       x + WINDOW_SIZE // 2,
                                                                                                       visited.shape[0]),
                                                                                                   max(0,
                                                                                                       y - WINDOW_SIZE // 2):min(
                                                                                                       y + WINDOW_SIZE // 2,
                                                                                                       visited.shape[1])])
    
                        probsum = numpy.sum(probs[ini, inj])
                        '''
                        ini, inj, ink = numpy.nonzero(places)
                        # ind = numpy.random.choice(len(ini), p=probs[ini, inj] / probsum)
                        ind = numpy.random.choice(len(ini))
                        ans = [ini[ind], inj[ind], ink[ind]]
                        if (is_first == 1):
                            print('exploring', ans, places[tuple(ans)])
                        # print(ans)
                    else:
                        mask = self.units.msg2mask(self.disGame, k[1])
                        rl.acquire()
                        ans = self.units.predict_ans_masked(X, mask, is_first == 1)
                        rl.release()
                        if (is_first == 1):
                            print('exploiting', ans[0], mask[tuple(ans[0])])

                        if (is_first == 1):
                            fq.write(str(ans[1]) + '\n')
                            fq.flush()
                            os.fsync(fq.fileno())
                            ans = ans[0]
                    util64.send_msg(con, pickle.dumps(ans))
                    if (last_action is not None):
                        self.buflock.acquire()
                        self.buf.add(last_state, last_action, k[1],
                                (k[2] - self.exploration_weight * unvisited - last_value), 0)
                        self.buflock.release()
                    last_state = k[1]
                    last_action = ans
                    last_value = k[2] - self.exploration_weight * unvisited
                    if (is_first == 1):
                        feval.write(str(last_value) + '\n')
                        feval.flush()
                        os.fsync(feval.fileno())
            except EOFError:
                print('exception found')
                break
        if (is_first == 1):
            feval.close()
            fq.close()