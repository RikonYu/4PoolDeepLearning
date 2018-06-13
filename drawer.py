import matplotlib.pyplot as plt
import pickle
import numpy

fin=open('trainerr3.txt','rb')
fin2=open('trainerr2.txt','rb')
fin3=open('trainerr.txt','rb')

k=pickle.load(fin)
plt.plot(k,'r',label='deep')
k=pickle.load(fin2)
plt.plot(k,'b',label='shallow')
k=pickle.load(fin3)
plt.plot(k,'g',label='deeper')
plt.legend()
plt.show()
