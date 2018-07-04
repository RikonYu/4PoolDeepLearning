import util64
import pickle
import socket
import time
import numpy
import os
import ReplayBuffer
from ClassConstr import getUnitClass
import threading
from consts import WINDOW_SIZE
from readerwriterlock import RWLock

# Deep Q Learning
batch_size = 64
disGame = None
# buf = ReplayBuffer.ReplayBuffer(20000)
buflock = threading.Semaphore(1)
buf = ReplayBuffer.PriortizedReplayBuffer(50000)
targetType = ''
dragoons = None
target = None
tempd = None
# dragoon_weights=None

epsilon = 0.3
discount = 0.9
learn_epoch = 0
exploration_weight = 0.0001
lock = RWLock.RWLockWrite()


def Qlearner():
    global dragoons, buf, disGame, target, discount, learn_epoch, targetType, lock, tempd, batch_size
    global exploration_weight
    replace_every = 50
    train_every = 64
    wl = lock.genWlock()
    while (True):
        if (buf.count < train_every):
            # print('not enough samples')
            time.sleep(10)
            continue
        buflock.acquire()
        samples, indx, bias = buf.sample(batch_size)
        buflock.release()
        print('training')
        tempd.set_weights(dragoons.get_weights())
        X = numpy.array([dragoons.msg2state(disGame, i) for i, _a, _sp, _r, _it in samples])
        Y = dragoons.predict_all(X)  # Q(s,a)
        # print(numpy.array([dragoons.msg2state(disGame, i) for _s, _a, i, _r, _it in samples]).shape)
        aprime = target.predict_max(
            [dragoons.msg2state(disGame, i) for _s, _a, i, _r, _it in samples])  # max_aQ'(s',a')
        Y_ = [(samples[i][3] + discount * aprime[i] * (1 - samples[i][4])) for i in
              range(batch_size)]  # r+discount*max_aq'(s',a')
        diff = numpy.copy(Y)
        for i in range(batch_size):
            diff[i, samples[i][1][0], samples[i][1][1], samples[i][1][2]] = Y_[i]

        # not using bias for now
        buflock.acquire()
        buf.update(indx,
                   list(Y_[i] - Y[i, samples[i][1][0], samples[i][1][1], samples[i][1][2]] for i in range(batch_size)))
        buflock.release()
        tempd.train(X, diff)
        tempd.save()
        if (learn_epoch % replace_every == 0):
            target.set_weights(tempd.get_weights())
        wl.acquire()
        # print('acquired')
        dragoons.set_weights(tempd.get_weights())
        wl.release()
        # print('released')
        buflock.acquire()
        buf.count -= train_every
        buflock.release()
        learn_epoch += 1


def unit_RL(con, is_first):
    global disGame, buf, dragoons, epsilon, targetType, target, tempd, lock
    last_state = None
    last_action = None
    last_value = 0
    last_mine = 0
    visited = numpy.zeros([1, 1])
    unvisited = 0
    mask = numpy.zeros(1)
    rl = lock.genRlock()
    feval = 0
    fq = 0
    if (is_first == 1):
        feval = open('rewards.txt', 'w')
        fq = open('Qvals.txt', 'w')
    while (True):
        try:
            data = util64.recv_msg(con)
            k = pickle.loads(data)
            # print(k)
            if (k[0] == 'reg'):
                if (disGame is not None):
                    con.send(b'ok')
                    break
                disGame = util64.gameInstance(k[1])
                targetType = k[2]
                dragoons = getUnitClass(targetType, True)
                target = getUnitClass(targetType, True)
                tempd = getUnitClass(targetType, True)
                tempd.set_weights(dragoons.get_weights())
                con.send(b'ok')
                break
            else:
                ans = 0
                mine_count = k[1][1][5]
                X = dragoons.msg2state(disGame, k[1])
                if (k[0] == 'terminal' and last_action is not None):
                    buflock.acquire()
                    buf.add(last_state, last_action, last_state, (last_mine - mine_count) * 0.2, 1)
                    buflock.release()
                    break
                if (visited.shape[0] == 1):
                    visited = numpy.zeros(disGame.regions.shape)
                    unvisited = visited.shape[0] * visited.shape[1]
                    last_value = -exploration_weight * unvisited
                # print(k)
                visited[k[1][0][0], k[1][0][1]] += 1
                if (visited[k[1][0][0], k[1][0][1]] == 1):
                    unvisited -= 1
                if (numpy.random.random() < epsilon):
                    places = dragoons.msg2mask(disGame, k[1])
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
                    ini, inj, ink = numpy.nonzero(places)
                    # top5=numpy.argpartition(probs[ini,inj],-5)[-5:]
                    # print('options',numpy.unravel_index(top5,probs.shape),probs[ini,inj][top5],len(ini))
                    # ind=numpy.random.choice(5,p=probs[ini,inj][top5]/sum(probs[ini,inj][top5]))
                    # ind = numpy.argmax(probs[ini, inj])
                    probsum = numpy.sum(probs[ini, inj])
                    ind = numpy.random.choice(len(ini), p=probs[ini, inj] / probsum)
                    # ans=[ini[ind]-WINDOW_SIZE//2,inj[ind]-WINDOW_SIZE//2,ink[ind]]
                    ans = [ini[ind] - WINDOW_SIZE // 2, inj[ind] - WINDOW_SIZE // 2, ink[ind]]
                    if (is_first == 1):
                        print('exploring', ans)
                    # print(ans)
                else:
                    # temps = getUnitClass()
                    # temps.set_weights(dragoons.get_weights())
                    mask = dragoons.msg2mask(disGame, k[1])
                    if(is_first==1):
                        print('mask:',numpy.sum(mask))
                    '''
                    ftest=open('masks.txt','wb')
                    pickle.dump(mask,ftest)
                    ftest.close()
                    '''
                    # print('trying to acquire read %d'%threading.get_ident())
                    rl.acquire()
                    # print('read acquired %d'%threading.get_ident())
                    ans = dragoons.predict_ans_masked(X, mask, is_first == 1)
                    rl.release()
                    if (is_first == 1):
                        print('exploiting', ans)
                    # print('read released %d'%threading.get_ident())
                # con.sendall(pickle.dumps(ans))
                # util64.send_msg(con,pickle.dumps([ans,mask]))
                    if (is_first == 1):
                        fq.write(str(ans[1]) + '\n')
                        fq.flush()
                        os.fsync(fq.fileno())
                        ans=ans[0]
                util64.send_msg(con, pickle.dumps(ans))
                if (last_action is not None):
                    buflock.acquire()
                    buf.add(last_state, last_action, k[1], (
                            (k[1][1][1] - exploration_weight * unvisited - mine_count * 0.2) - (
                            last_value - last_mine * 0.2)), 0)
                    buflock.release()
                last_state = k[1]
                last_action = ans
                last_mine = mine_count
                last_value = k[1][1][1] - exploration_weight * unvisited
                if (is_first == 1):
                    feval.write(str(last_value - last_mine * 0.2) + '\n')
                    feval.flush()
                    os.fsync(feval.fileno())
        except EOFError:
            print('exception found')
            break
    if (is_first == 1):
        feval.close()
        fq.close()


if (__name__ == '__main__'):
    first_agent = 0
    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = 'linux.cs.uwaterloo.ca'
    soc.bind((host, 12346))
    soc.listen(5)
    lx = threading.Thread(target=Qlearner, args=[])
    time.sleep(1)
    lx.start()

    print('listening')

    while (True):
        con, addr = soc.accept()
        # print(addr)
        k = threading.Thread(target=unit_RL, args=[con, first_agent])
        first_agent += 1
        time.sleep(1)
        k.start()
