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
curTask=taskBaseScout

def send_msg(sock, msg):
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)


def send(u, tp, sock):
    msg=gameMessage(u)
    send_msg(sock, pickle.dumps(gameState(tp, msg, curTask.valueFunc(u))))


def send_reg():
    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    soc.connect((address, 12346))
    send_msg(soc, pickle.dumps(mapState(mapMessage(), curTask.unitTypes[0].getName(), game.mapName())))
    _ = soc.recv(16)
    soc.close()



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
        if (game.getFrameCount() % curTask.frameSkip != 0):
            for i in game.getAllUnits():
                if(i.getLastCommand().getType().getName()=='Move'):
                    game.drawLineMap(i.getPosition(),i.getLastCommand().getTargetPosition(),pybrood.Colors.Red)
            return
        for i in game.getAllUnits():
            if (curTask.can_control(i,self.playerMe)):
                if (i.getID() in Socks):
                    continue
                Socks[i.getID()] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                Socks[i.getID()].connect((address, 12346))

        kys = list(Socks.keys())
        for i in kys:
            if (curTask.is_terminal(game.getUnit(i))):
                unitThreads[i] = threading.Thread(target=dead_unit, args=[i])
                unitThreads[i].start()
            else:
                unitThreads[i] = threading.Thread(target=unit_thread, args=[i])
                unitThreads[i].start()
        for i in unitThreads.keys():
            unitThreads[i].join()
        for i in kys:
            if (curTask.is_terminal(game.getUnit(i))):
                Socks[i].close()
                Socks.pop(i, None)
                unitThreads.pop(i, None)
        if(game.getFrameCount()>=curTask.maxFrame):
            ans=curTask.finalValueFunc(self.playerMe)
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
    run(PlayAI)
