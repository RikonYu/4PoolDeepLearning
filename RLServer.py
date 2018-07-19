import util64
import pickle
import socket
import time
import numpy
import os
from a2c import A2C
import threading
from QLearning import QLearning
from consts import WINDOW_SIZE



#agent=QLearning(0.3,0.9,0,32)
agent=A2C(0.3,0.95,0,0)
if (__name__ == '__main__'):
    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = 'linux.cs.uwaterloo.ca'
    soc.bind((host, 12346))
    soc.listen(5)
    lx = threading.Thread(target=agent.learner, args=[])
    time.sleep(1)
    lx.start()
    try:
        os.remove('Qvals.txt')
        os.remove('rewards.txt')
    except:
        pass
    print('listening')

    while (True):
        con, addr = soc.accept()
        # print(addr)
        k = threading.Thread(target=agent.controller, args=[con, agent.agent_no])
        #print(agent_no)
        QL.agent_no += 1
        time.sleep(1)
        k.start()
