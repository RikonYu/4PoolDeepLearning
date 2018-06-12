import matplotlib.pyplot as plt
import pickle
import numpy

fin=open('validerr.txt','rb')
k=pickle.load(fin)
plt.plot(k)
plt.show()
