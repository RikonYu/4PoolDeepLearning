import numpy
import keras
import scipy
import pickle

regions=None
hground=None

def makeReg(reg):
    global regions, hground
    #print(reg)
    regions=numpy.zeros([reg[0][0][0]*32+1,reg[0][0][1]*32+1])
    hground=numpy.zeros([reg[0][0][0]*32+1,reg[0][0][1]*32+1])
    for i in reg[1]:
        regions[i[0]:i[1],i[2]:i[3]]=i[4]
        hground[i[0]:i[1],i[2]:i[3]]=i[5]
def y2stateDrone(ind):
    ans=numpy.zeros([360,360,6])
    if(ind[2] in [0,5]):
        ans[180,180,ind[2]]=1
    else:
        ans[ind[0],ind[1],ind[2]]=1
    return ans
def msg2stateDrone(msg):
    global regions, hground
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
    X=hground.shape[0]
    Y=hground.shape[1]
    ans[ax:min(360,X-x+180),
        ay:min(360,Y-y+180),10]=hground[max(0,x-180):min(x+180,X),max(0,y-180):min(y+180,Y)]

    ans[ax:min(360,X-x+180),
        ay:min(360,Y-y+180),0]=regions[max(0,x-180):min(x+180,X),max(0,y-180):min(y+180,Y)]
    return ans
