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





'''
def plots(arr, rows=1):
    fig=plt.figure(figsize=(rows, arr.shape[2]//rows))
    for i in range(arr.shape[2]):
        fig.add_subplot(rows, arr.shape[2]//rows, i+1)
        plt.imshow(arr[:,:,i].transpose(),cmap=plt.cm.gray)
    plt.show()
def cvt(x):
    if(x=='None'):
        return 0
    return float(x)
if(__name__=='__main__'):
    plots(pickle.load(open('state.txt','rb')),1)
    plots(pickle.load(open('mask.txt', 'rb')), 1)

    fvv=open('results/vultureq.txt','r')
    k=fvv.readlines()
    k=k[::4]
    x=[int(i[12:]) for i in k]
    gamelen=[(i+1)//3 for i in x]

    fin = open('rewards.txt', 'r')
    k = list(map(cvt, fin.read().splitlines()))
    k = numpy.array(k)
    #plt.plot(k, 'r', label='Reward')
    plt.plot([sum(k[:i+1]/(i+1)) for i in range(len(k)-1)], 'r', label='average Reward')
    plt.legend()
    plt.show()
    sk=[]
    ct=0
    pt=0
    ss=0
    fin=open('Qvals.txt','r')
    k = list(map(cvt, fin.read().splitlines()))
    plt.plot([sum(k[i:i+10000])/10000 for i in range(10000,len(k)-10000)], 'r', label='Q')
    plt.show()
    for i in range(len(k)):
        if(pt>=len(gamelen)):
            break
        if(ct>=gamelen[pt]):
            ct=0
            pt+=1
            sk.append(ss)
            ss=0
        if(k[i]!='None'):
            ss+=float(k[i])
        ct+=1
    plt.plot(numpy.array(sk)/gamelen[:-1], 'r', label='average episodic Q')
    plt.legend()
    plt.show()

    ft=open('state.txt','rb')
    sb=pickle.load(ft)
    plots(sb[:,:,4:], 1)

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
        plt.imshow(numpy.rot90(sb[:, :, i], axes=(1,0)))
        if(poss[2][pos]==i):
            plt.scatter(poss[0][pos]+15,poss[1][pos]+15, s=15, c='red', marker='o')
    for i in range(2):
        fig.add_subplot(2, 3, i + 4)
        plt.imshow(mask[:, :, i].transpose())
    fig.add_subplot(2, 3, 3)
    plt.imshow(ssb.transpose(), cmap=plt.cm.gray)
    plt.show()