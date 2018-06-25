import numpy

class ReplayBuffer:
    def __init__(self,length):
        self.buffer=[]
        self._ind=0
        self.maxlen=length
        self.count=0
    def getCount(self):
        return self.count
    def add(self,state,action,new_state,reward,is_terminal):
        self.count+=1
        if(len(self.buffer)>=self.maxlen):
            self.buffer[self._ind]=[state,action,new_state,reward,is_terminal]
            self._ind=(self._ind+1)%self.maxlen
        else:
            self.buffer.append([state,action,new_state,reward,is_terminal])
    def sample(self,batch_size):
        if(len(self.buffer)<batch_size):
            return []
        inds=numpy.random.choice(len(self.buffer),batch_size)
        return [self.buffer[i] for i in inds]
