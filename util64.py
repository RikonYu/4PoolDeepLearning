import numpy
import keras
import scipy
import pickle
import socket
import struct

def recv_msg(sock):
    # Read message length and unpack it into an integer
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # Read the message data
    return recvall(sock, msglen)

def recvall(sock, n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

def shrinkScr(x):
    if(x<0):
        return 0
    if(x>359):
        return 359
    return x
def y2stateDrone(ind):
    ans=numpy.zeros([360,360,6])
    #print(ind)
    if(ind[2] in [0,5]):
        ans[180,180,ind[2]]=1
    else:
        ans[shrinkScr(ind[0]),shrinkScr(ind[1]),ind[2]]=1
    return ans
class gameInstance:
    def __init__(self,reg):
        self.regions=numpy.zeros([reg[0][0][0]*32+1,reg[0][0][1]*32+1])
        self.hground=numpy.zeros([reg[0][0][0]*32+1,reg[0][0][1]*32+1])
        for i in reg[1]:
            self.regions[i[0]:i[1],i[2]:i[3]]=i[4]
            self.hground[i[0]:i[1],i[2]:i[3]]=i[5]

    def msg2maskDrone(self,msg):
        ans=numpy.zeros([360,360,6])
        ans[:,:,0]=1
        ans[:,:,5]=(msg[1][1] or msg[1][2])
        x=msg[0][0]
        y=msg[0][1]
        ax=max(0,180-x)
        ay=max(0,180-y)
        hei,wid=self.regions.shape
        ans[ax:min(360,hei-x+180),
            ay:min(360,wid-y+180),1]=self.hground[max(0,x-180):min(x+180,hei),max(0,y-180):min(y+180,hei)]

        ans[ax:min(360,hei-x+180),
            ay:min(360,wid-y+180),1]*=self.terrain[max(0,x-180):min(x+180,wid),max(0,y-180):min(y+180,wid)]
        for i in msg[2]:
            ans[i[0][0]-x+180,i[0][1]-y+180,4]=1-i[2]
        for i in msg[3]:
            ans[i[0][0]-x+180,i[0][1]-y+180,4]=1-i[2]
        for i in msg[4]:
            ans[i[1][0]-x+180,i[1][1]-y+180,3]=i[0]
        for i in msg[5]:
            ans[i[1][0]-x+180,i[1][1]-y+180,3]=1
        return ans
    def msg2stateDrone(self,msg):
        ans=numpy.zeros([360,360,18])
        x,y=msg[0]
        ans[:,:,11]=msg[1][0]
        ans[:,:,16]=msg[1][1]
        ans[:,:,17]=msg[1][2]
        for u in msg[2]:
            nx=u[0][0]-x+180
            ny=u[0][1]-y+180
            if(u[2]):
                ans[nx,ny,5]=1
            elif(u[3]):
                ans[nx,ny,6]=1
            else:
                ans[nx,ny,4]=1
            ans[nx,ny,13]=u[1]
            ans[nx,ny,14]=u[4]
            ans[nx,ny,15]=u[5]
        for u in msg[3]:
            
            nx=u[0][0]-x+180
            ny=u[0][1]-y+180
            if(u[2]):
                ans[nx,ny,5]=1
            elif(u[3]):
                ans[nx,ny,6]=1
            else:
                ans[nx,ny,4]=1
            ans[nx,ny,12]=u[1]
        for u in msg[4]:
            
            nx=u[1][0]-x+180
            ny=u[1][1]-y+180
            if(u[0]):
                ans[nx,ny,7]=1
            else:
                ans[nx,ny,8]=1
        for u in msg[5]:

            nx=u[0]-x+180
            ny=u[1]-y+180
            #print(u,x,y,nx,ny)
            ans[nx,ny,9]=1
        ax=max(0,180-x)
        ay=max(0,180-y)
        X=self.hground.shape[0]
        Y=self.hground.shape[1]
        ans[ax:min(360,X-x+180),
            ay:min(360,Y-y+180),10]=self.hground[max(0,x-180):min(x+180,X),max(0,y-180):min(y+180,Y)]

        ans[ax:min(360,X-x+180),
            ay:min(360,Y-y+180),0]=self.regions[max(0,x-180):min(x+180,X),max(0,y-180):min(y+180,Y)]
        return ans
