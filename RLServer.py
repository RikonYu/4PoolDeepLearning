import util64
import pickle
import socket
import time
import numpy
import os, sys
from a2c import A2C
import threading
from QLearning import QLearning
from consts import WINDOW_SIZE
from DebugLearner import DebugLearner


#agent=QLearning(0.3,0.9,0,64)
#agent=A2C(0.3,0.95,0,0)
if (__name__ == '__main__'):
    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #agent = DebugLearner(0.3, 0.9, 0, 64)
    agent=QLearning(0.3,0.9,0,64)
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
        if(sys.argv[1]=='train'):
            k = threading.Thread(target=agent.controller, args=[con, agent.agent_no])
        elif(sys.argv[1]=='test'):
            k=threading.Thread(target=agent.exploiter, args=[con,agent.agent_no])
        #print(agent_no)
        agent.agent_no += 1
        time.sleep(1)
        k.start()
