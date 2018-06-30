import numpy

class ReplayBuffer:
    def __init__(self,length):
        self.buffers=[]
        self._ind=0
        self.maxlen=length
        self.count=0

    def add(self,state,action,new_state,reward,is_terminal):
        if(len(self.buffers)>=self.maxlen):
            self.buffers[self._ind]=[state,action,new_state,reward,is_terminal]
        else:
            self.buffers.append([state,action,new_state,reward,is_terminal])
        self._ind=(self._ind+1)%self.maxlen
        self.count+=1

    def sample(self,batch_size):
        if(len(self.buffers)<batch_size):
            return []
        inds=numpy.random.choice(len(self.buffers),batch_size)
        return [self.buffers[i] for i in inds]

class SegTree:
    def __init__(self,ln):
        self.a=numpy.zeros(ln*4+1)
        self.left=numpy.zeros(ln*4+1)
        self.right=numpy.zeros(ln*4+1)
        self.build(1,ln,1)
    def build(self,left,right,ind):
        self.left[ind]=left
        self.right[ind]=right
        if(left==right):
            return
        self.build(left,(left+right)//2,ind*2)
        self.build((left+right)//2+1,right,ind*2+1)

    def getmax(self):
        return self.a[1]

    def set(self,ind,val,pos):
        self.a[pos]=max(self.a[pos],val)
        if(self.left[pos]==ind and self.right[pos]==ind):
            self.a[pos]=val
            return
        if((self.left[pos]+self.right[pos])//2>ind):
            self.set(ind,val,pos*2)
        else:
            self.set(ind,val,pos*2+1)

class PriortizedReplayBuffer:
    def __init__(self,length):
        self.buffers=[]
        self._ind=0
        self.maxlen=length
        self.prts=SegTree(length)
        self.psum=0
        self.count=0
        self.beta=1
        self.alpha=1
    def add(self,state,action,new_state,reward,is_terminal):
        max_prt=self.prts.getmax()
        if(len(self.buffers)>=self.maxlen):
            self.psum-=numpy.exp(self.buffers[self._ind][1])
            self.buffers[self._ind]=[[state,action,new_state,reward,is_terminal],max_prt]
        else:
            self.buffers.append([[state,action,new_state,reward,is_terminal],max_prt])
        self.psum+=numpy.exp(max_prt)
        self.prts.set(self._ind,max_prt,1)
        self._ind=(self._ind+1)%self.maxlen
        self.count+=1

    def sample(self,batch_size):
        if(len(self.buffers)<batch_size):
            return []
        probs=numpy.exp([i[1] for i in self.buffers])/self.psum
        inds=numpy.random.choice(len(self.buffers),batch_size,p=probs)
        bias=numpy.power((self.count*probs[inds]),self.beta)
        return [self.buffers[i][0] for i in inds],inds,bias

    def update(self,ind,p):
        for i in range(ind):
            self.psum-=numpy.exp(self.buffers[ind[i]][1])
            self.prts.set(ind[i],abs(p[i]),1)
            self.psum+=numpy.exp(abs(p[i]))
            self.buffers[ind[i]][1]=abs(p[i])
