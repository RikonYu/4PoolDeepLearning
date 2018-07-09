import numpy
import pickle
import util32
import socket, os
import struct
import threading
import time
from PIL import Image
from pybrood import BaseAI, run, game, Color
import pybrood
from tasks import *

Socks = {}
address = 'linux.cs.uwaterloo.ca'
# address='127.0.0.1'
unitThreads = {}
targetType = 'Protoss_Dragoon'
first_time=0

def send_msg(sock, msg):
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)


def send(u, tp, sock):
    msg = util32.game2msg(u)
    send_msg(sock, pickle.dumps([tp, msg]))


def send_reg():
    global first_time
    if(first_time):
        return
    first_time=1
    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    soc.connect((address, 12346))
    send_msg(soc, pickle.dumps(['reg', util32.reg2msg(), targetType]))
    k = soc.recv(16)
    soc.close()



def dead_unit(ind):
    send(game.getUnit(ind), 'terminal', Socks[ind])


def unit_thread(ind):
    send(game.getUnit(ind), targetType, Socks[ind])
    k = pickle.loads(util32.recv_msg(Socks[ind]))
    util32.command(game.getUnit(ind), k)


class PlayAI(BaseAI):
    def prepare(self, task):
        self.playerMe = game.self()
        send_reg()
        self.task=task

    def frame(self):
        if (game.getFrameCount() % 10 != 0):
            return
        for i in game.getAllUnits():
            if (self.task.can_control(i,self.playerMe)):
                if (i.getID() in Socks):
                    continue
                Socks[i.getID()] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                Socks[i.getID()].connect((address, 12346))

        kys = list(Socks.keys())
        for i in kys:
            if (self.task.is_terminal(i)):
                unitThreads[i] = threading.Thread(target=dead_unit, args=[i])
                unitThreads[i].start()
            else:
                unitThreads[i] = threading.Thread(target=unit_thread, args=[i])
                unitThreads[i].start()
        for i in unitThreads.keys():
            unitThreads[i].join()
        for i in kys:
            if (self.task.is_terminal(i)):
                Socks[i].close()
                Socks.pop(i, None)
                unitThreads.pop(i, None)
        if(game.getFrameCount()>=self.task.maxFrame):
            ans=self.task.valueFunc()
            print('final Value', ans)
            game.leaveGame()
        #print(len(Socks.keys()))
    def finished(self):

        pass

def printer(k):
    pic=Image.fromarray(k[:,:,1]*255.0)
    pic.show()
    time.sleep(0.1)
    pic.close()

if (__name__ == '__main__'):
    # soc=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    # soc.connect(('linux.cs.uwaterloo.ca',12346))
    run(PlayAI, taskDragoonDefuse)
