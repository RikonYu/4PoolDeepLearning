import numpy
import pickle
import util32
import socket,os
import struct
import threading
from pybrood import BaseAI, run, game, Color
droneSocks={}
def send_msg(sock, msg):
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)
    
def send(u,tp,sock):
    msg=util32.game2msgDrone(u)
    #print('client send unit',len(msg))
    send_msg(sock,pickle.dumps([tp,msg]))
def send_reg():
    soc=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    soc.connect(('linux.cs.uwaterloo.ca',12346))
    #print('client send reg',len(pickle.dumps(['reg',util32.reg2msg()])))
    send_msg(soc,pickle.dumps(['reg',util32.reg2msg()]))
    soc.close()
def receive(soc):
    k=soc.recv(16384)
    #print('client recv',len(k))
    while(len(k)==0):
        k=soc.recv(16384)
        
        #print('client recv',len(k))
    return pickle.loads(k)
def unit_thread(ind):
    send(game.getUnit(ind),'drone',droneSocks[ind])
    print('send %d'%ind)
    k=receive(soc)
    print(k)
    util32.command(game.getUnit(ind),k)
class PlayAI(BaseAI):
    def prepare(self):
        self.playerMe=game.self()
        send_reg()
    def frame(self):
        if(game.getFrameCount()%10!=0):
            return
        for i in game.getAllUnits():
            if(i.getType().getName()=='Zerg_Drone' and i.getPlayer()==self.playerMe):
                if(i.getID() in droneSocks):
                    continue
                droneSocks[i.getID()]=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
                droneSocks[i.getID()].connect(('linux.cs.uwaterloo.ca',12346))
                #send(i,'drone')
                #k=receive()
                #util32.command(i,k)
        for i in droneSocks:
            if(game.getUnit(i).exists()==False):
                droneSocks[i].close()
                droneSocks.pop(i,None)
            else:
                threading.Thread(target=unit_thread,args=[i])
                
    def finished(self):
        pass

if(__name__=='__main__'):
    #soc=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    #soc.connect(('linux.cs.uwaterloo.ca',12346))
    run(PlayAI)
