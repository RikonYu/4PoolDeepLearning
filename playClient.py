import numpy
import pickle
import util32
import socket, os
import struct
import threading
# import matplotlib.pyplot as plt
from pybrood import BaseAI, run, game, Color

Socks = {}
address = 'linux.cs.uwaterloo.ca'
# address='127.0.0.1'
unitThreads = {}
targetType = 'Protoss_Dragoon'


def send_msg(sock, msg):
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)


def send(u, tp, sock):
    msg = util32.game2msg(u)
    send_msg(sock, pickle.dumps([tp, msg]))


def send_reg():
    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    soc.connect((address, 12346))
    send_msg(soc, pickle.dumps(['reg', util32.reg2msg(), targetType]))
    k = soc.recv(16)
    soc.close()


def receive(soc):
    k = soc.recv(16384)
    while (len(k) == 0):
        k = soc.recv(16384)
    return pickle.loads(k)


def dead_unit(ind):
    send(game.getUnit(ind), 'terminal', Socks[ind])


def unit_thread(ind):
    send(game.getUnit(ind), targetType, Socks[ind])
    #print('%d sent at %d'%(ind,game.getFrameCount()))
    k = receive(Socks[ind])
    #print('%d recv at %d'%(ind,game.getFrameCount()))
    util32.command(game.getUnit(ind), k)


class PlayAI(BaseAI):
    def prepare(self):
        self.playerMe = game.self()
        send_reg()

    def frame(self):
        if (game.getFrameCount() % 10 != 0):
            return
        for i in game.getAllUnits():
            if (i.getType().getName() == targetType and i.getPlayer() == self.playerMe):
                if (i.getID() in Socks):
                    continue
                #print('dragoon %d'%i.getID(),i.getPosition())
                Socks[i.getID()] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                Socks[i.getID()].connect((address, 12346))

        kys = list(Socks.keys())
        for i in kys:
            if (game.getUnit(i).exists() == False):
                unitThreads[i] = threading.Thread(target=dead_unit, args=[i])
                unitThreads[i].start()
            else:
                unitThreads[i] = threading.Thread(target=unit_thread, args=[i])
                unitThreads[i].start()
        for i in unitThreads.keys():
            unitThreads[i].join()
        for i in kys:
            if (game.getUnit(i).exists() == False):
                Socks[i].close()
                Socks.pop(i, None)
                unitThreads.pop(i, None)
        #print(len(Socks.keys()))
    def finished(self):
        pass


if (__name__ == '__main__'):
    # soc=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    # soc.connect(('linux.cs.uwaterloo.ca',12346))
    run(PlayAI)
