import socket
import pickle
import threading


soc=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#soc.bind(('127,0,0,1',12345))

def thread_job():
    soc.send(b'23')
    return

soc.connect(('127.0.0.1',32132))
for i in range(10):
    threading.Thread(thread_job,None)
    
