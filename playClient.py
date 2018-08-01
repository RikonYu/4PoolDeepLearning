import numpy
import pickle
import util32
import socket
from gameMessage import *
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
first_time=0
curTask=taskDebug
regSoc=None

def send_msg(sock, msg):
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)


def send(u, tp, sock):
    msg=gameMessage(u)
    send_msg(sock, pickle.dumps(gameState(tp, msg, curTask.valueFunc(u))))


def send_reg():
    global regSoc
    regSoc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    regSoc.connect((address, 12346))
    send_msg(regSoc, pickle.dumps(mapState(mapMessage(), curTask.unitTypes[0].getName(), game.mapName())))
    _ = regSoc.recv(16)
    regSoc.close()



def dead_unit(ind):
    send(game.getUnit(ind), 'terminal', Socks[ind])


def unit_thread(ind):
    send(game.getUnit(ind), curTask.unitTypes[0].getName(), Socks[ind])
    k = pickle.loads(util32.recv_msg(Socks[ind]))
    util32.command(game.getUnit(ind), k)


class PlayAI(BaseAI):
    def prepare(self):
        self.playerMe = game.self()
        send_reg()

    def frame(self):
        for i in game.getAllUnits():
            if(i.getType().getName()=="Zerg_Drone"):
                game.drawBoxMap(i.getPosition()[0]-256, i.getPosition()[1]-256, i.getPosition()[0]+256, i.getPosition()[1]+256,pybrood.Colors.Red)
                return

    def finished(self):
        kys = list(Socks.keys())
        for i in kys:
            Socks[i].close()
            Socks.pop(i, None)
            unitThreads.pop(i, None)
        pass

def printer(k):
    pic=Image.fromarray(k[:,:,1]*255.0)
    pic.show()
    time.sleep(0.1)
    pic.close()

if (__name__ == '__main__'):
    # soc=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    # soc.connect(('linux.cs.uwaterloo.ca',12346))
    run(PlayAI)
