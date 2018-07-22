import matplotlib.pyplot as plt
import pickle
import numpy
'''
fin=open('realerr.txt','rb')
fin2=open('trainerr2.txt','rb')
fin3=open('trainerr.txt','rb')
fin4=open('realverr.txt','rb')

k=pickle.load(fin)
plt.plot(k,'r',label='deep')
k=pickle.load(fin2)
plt.plot(k,'b',label='shallow')
k=pickle.load(fin)
plt.plot(k,'r',label='real')
k=pickle.load(fin4)
#print(len(k))
plt.plot(range(0,2000,50),k,'g',label='valid')
plt.legend()
plt.show()
fin=open'masks(.txt','rb')
mask=pickle.load(fin)
fig = plt.figure(figsize=(6, 6))
for i in range(6):
    fig.add_subplot(2, 3, i + 1)
    plt.imshow(mask[:, :, i] * 255.0, cmap=plt.cm.gray)
plt.show()




fin=open('rewards.txt','r')
k=list(map(float,fin.read().splitlines()))
k=numpy.array(k)
sb=[sum(k[x:x+15]) for x in range(len(k)//15)]
plt.plot(k,'r',label='R')
plt.legend()
plt.show()
fin=open('Qvals.txt','r')
k=list(map(float,fin.read().splitlines()))
plt.plot(k,'r',label='Q')
plt.legend()
plt.show()
'''

ferr=open('trainerr.txt','r')
k=[float(i[:-1]) for i in ferr.readlines()]
plt.plot(k[:150], label='training MSE')
plt.legend()
plt.show()


fval=open('allval.txt','rb')
sb=pickle.load(fval)
mask=pickle.load(fval)
ssb=numpy.sum(sb,axis=2)
fig = plt.figure(figsize=(2, 3))
poss=numpy.nonzero(mask)
pos=numpy.argmax(sb[poss[0],poss[1],poss[2]])
print(numpy.amax(sb), numpy.amin(sb))
for i in range(2):
    fig.add_subplot(2, 3, i + 1)
    plt.imshow(numpy.rot90(sb[:, :, i], axes=(1,0)), cmap=plt.cm.gray)
    if(poss[2][pos]==i):
        plt.scatter(poss[0][pos]+15,poss[1][pos]+15, s=15, c='red', marker='o')
for i in range(2):
    fig.add_subplot(2, 3, i + 4)
    plt.imshow(mask[:, :, i].transpose(), cmap=plt.cm.gray)
fig.add_subplot(2, 3, 3)
plt.imshow(ssb.transpose(), cmap=plt.cm.gray)
plt.show()