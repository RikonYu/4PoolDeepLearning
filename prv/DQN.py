import numpy
import random
import copy

class DQN:
    def __init__(self,batch_size,memory_len,model,learning_rate=0.1,discount=0.9 ):
        self.experience=[]
        self.memory_len=memory_len
        self.model=model
        self.learning_rate=learning_rate
        self.discount=discount
        self.batch_size=batch_size
    def add_experience(self,state,action,reward,next_state):
        #exp:[state,action,reward]
        self.experience.append([state,action,reward,next_state])
        if(len(self.experience)>=self.memory_len):
            self.experience=self.experience[1:]
    def get_data(self,states,actions,new_states,rewards):
        #print(self.model.predict(numpy.array(new_states)).shape)
        targets=rewards+self.learning_rate*numpy.amax(self.model.predict(numpy.array(new_states)),axis=(1,2,3))
        return (states,actions,targets)
    def experience_replay(self):
        ind=numpy.random.choice(len(self.experience),self.batch_size)
        states=[self.experience[i][0] for i in ind]
        actions=[self.experience[i][1] for i in ind]
        rewards=numpy.array([self.experience[i][2] for i in ind])
        nstates=[self.experience[i][3] for i in ind]
        return self.get_data(states,actions,nstates,rewards)
        
    
