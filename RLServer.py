import util64
import pickle
import socket
import time
import numpy
import ReplayBuffer
from DroneNet import DroneNet
from DragoonNet import DragoonNet
from ClassConstr import getUnitClass
import threading
from consts import WINDOW_SIZE
from readerwriterlock import RWLock

# Deep Q Learning
batch_size = 64
disGame = None
buf = ReplayBuffer.ReplayBuffer(20000)
targetType = ''
dragoons = None
target = None
tempd=None
#dragoon_weights=None

epsilon = 0.3
discount = 0.9
learn_epoch = 0
lock = RWLock.RWLockWrite()


def learner():
    global dragoons, buf, disGame, target, discount, learn_epoch, targetType, lock,tempd, batch_size
    replace_every = 500
    train_every=256
    rl = lock.genRlock()
    wl = lock.genWlock()
    while (True):
        if(buf.count<train_every):
            #print('not enough samples')
            time.sleep(10)
            continue
        samples = buf.sample(batch_size)
        print('training')
        rl.acquire()
        tempd.set_weights(dragoons.get_weights())
        rl.release()
        X = numpy.array([dragoons.msg2state(disGame, i) for i, _a, _sp, _r, _it in samples])
        Y = dragoons.predict_all(X)
        print(numpy.array([dragoons.msg2state(disGame, i) for _s, _a, i, _r, _it in samples]).shape)
        aprime = target.predict_max([dragoons.msg2state(disGame, i) for _s, _a, i, _r, _it in samples])
        Y_ = [(samples[i][3] + discount * aprime[i] * (1 - samples[i][4])) for i in range(batch_size)]
        diff = numpy.copy(Y)
        for i in range(batch_size):
            diff[i, samples[i][1][0], samples[i][1][1], samples[i][1][2]] += Y_[i]
        # new_agent.fit(X,diff)

        tempd.train(X, diff)
        if (learn_epoch % replace_every == 0):
            tempd.save()
            target.set_weights(tempd.get_weights())

        wl.acquire()
        print('acquired')
        dragoons.set_weights(tempd.get_weights())
        wl.release()
        print('released')
        buf.count = 0
        learn_epoch += 1


def unit_RL(con):
    global disGame, buf, dragoons, epsilon, targetType, target,tempd, lock
    last_state = None
    last_action = None
    last_value = 0
    visited = numpy.zeros([1, 1])
    rl = lock.genRlock()
    while (True):
        try:
            data = util64.recv_msg(con)
            k = pickle.loads(data)
            # print(k)
            if (k[0] == 'reg'):
                disGame = util64.gameInstance(k[1])
                targetType = k[2]
                dragoons = getUnitClass(targetType, True)
                target = getUnitClass(targetType, True)
                tempd = getUnitClass(targetType,True)
                tempd.set_weights(dragoons.get_weights())
                con.send(b'ok')
                break
            else:
                ans = 0
                X = dragoons.msg2state(disGame, k[1])
                if (k[0] == 'terminal' and last_action != None):
                    buf.add(last_state, last_action, last_state, -1, 1)
                    break
                if (visited.shape[0] == 1):
                    visited = numpy.zeros(disGame.regions.shape)
                # print(k)
                visited[k[1][0][0], k[1][0][1]] += 1
                if (numpy.random.random() < epsilon):
                    places = dragoons.msg2mask(disGame, k[1])
                    # probs=numpy.exp(-visited[])
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
                    ind = numpy.argmax(probs[ini, inj])
                    # ans=[ini[ind]-WINDOW_SIZE//2,inj[ind]-WINDOW_SIZE//2,ink[ind]]
                    ans = [ini[ind] - WINDOW_SIZE // 2, inj[ind] - WINDOW_SIZE // 2, ink[ind]]
                    # print(ans)
                else:
                    # temps = getUnitClass()
                    # temps.set_weights(dragoons.get_weights())
                    mask = dragoons.msg2mask(disGame, k[1])
                    '''
                    ftest=open('masks.txt','wb')
                    pickle.dump(mask,ftest)
                    ftest.close()
                    '''
                    print('trying to acquire read')
                    rl.acquire()
                    print('read acquired')
                    ans = dragoons.predict_ans_masked(X, mask)
                    rl.release()
                    print('read released')
                con.sendall(pickle.dumps(ans))
                if (last_action != None):
                    if (k[1][1][1] != last_value):
                        print('reward', last_value, k[1][1][1])
                    buf.add(last_state, last_action, k[1], (k[1][1][1] - last_value), 0)
                last_state = k[1]
                last_action = ans
                last_value = k[1][1][1]
        except EOFError:
            break


if (__name__ == '__main__'):
    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = 'linux.cs.uwaterloo.ca'
    soc.bind((host, 12346))
    soc.listen(5)
    lx = threading.Thread(target=learner, args=[])
    time.sleep(1)
    lx.start()

    print('listening')

    while (True):
        con, addr = soc.accept()
        k = threading.Thread(target=unit_RL, args=[con])
        time.sleep(1)
        k.start()
