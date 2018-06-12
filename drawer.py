import matplotlib.pyplot as plt
import pickle
import numpy

fin=open('trainerr6.txt','rb')
fin2=open('trainerr.txt','rb')
k=pickle.load(fin)
plt.plot(k,'r',label='deep')
k=pickle.load(fin2)
plt.plot(k,'b',label='shallow')

plt.show()
