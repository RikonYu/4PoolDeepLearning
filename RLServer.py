import util64
import pickle
import socket
import time
import numpy
import os
import threading
from QLearning import QLearning
from consts import WINDOW_SIZE



QL=QLearning(0.3,0.9,0,64)
if (__name__ == '__main__'):
    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = 'linux.cs.uwaterloo.ca'
    soc.bind((host, 12346))
    soc.listen(5)
    #lx = threading.Thread(target=QL.learner, args=[])
    #time.sleep(1)
    #lx.start()
    try:
        os.remove('Qvals.txt')
        os.remove('rewards.txt')
    except:
        pass
    print('listening')

    while (True):
        con, addr = soc.accept()
        # print(addr)
        k = threading.Thread(target=QL.exploiter, args=[con, QL.agent_no])
        #print(agent_no)
        QL.agent_no += 1
        time.sleep(1)
        k.start()
