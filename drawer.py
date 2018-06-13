import matplotlib.pyplot as plt
import pickle
import numpy

fin=open('trainerr3.txt','rb')
fin2=open('trainerr2.txt','rb')
fin3=open('trainerr.txt','rb')
fin4=open('trainerr4.txt','rb')


k=pickle.load(fin)
plt.plot(k,'r',label='deep')
k=pickle.load(fin2)
plt.plot(k,'b',label='shallow')
k=pickle.load(fin3)
plt.plot(k,'g',label='deeper')
k=pickle.load(fin4)
plt.plot(k,'y',label='current')
plt.legend()
plt.show()
