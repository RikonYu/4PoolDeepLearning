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
'''
fin=open('masks.txt','rb')
mask=pickle.load(fin)
fig = plt.figure(figsize=(6, 6))
for i in range(6):
    fig.add_subplot(2, 3, i + 1)
    plt.imshow(mask[:, :, i] * 255.0, cmap=plt.cm.gray)
plt.show()