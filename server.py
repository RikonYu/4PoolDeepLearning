import keras
import socket
import os
import pickle
import numpy
import numpy.random

soc=socket.socket(socket.AF_INET,socket.SOCK_STREAM)


host='127.0.0.1'

soc.bind((host,6666))

soc.listen(5)

while True:
    c,addr=soc.accept()
    while True:
        data=c.recv(65536)
        if(data):
            c.send(pickle.dumps(["gather",numpy.random.randint(10)]))
        else:
            break
        
