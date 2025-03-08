# This code simulates the deep q-network for single-cell cellualr-connected UAV
# to find the best action/optimal trajectory for uav, resulting less interference.

import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque 

import os
import torch
import torch.nn as nn                  # neural network
import torch.optim as optim            # optimization function
import torch.nn.functional as F        # convlutional function

## simulation parameters
# discount factor
gamma=0.99
# learning rate(0.002)0.00005
learning_rate=1e-5
# the soft parameter for target update
tau=0.001
# update target weight
update_weight=10
# length of reply buffer
experience_memory=int(1e5)
# batch size
batch_size=64
# number of actions
num_action=48

# designing the main and target networks
class neural_network(nn.Module):
    # a fully connected neural network
    def __init__(self, state_size, action_size, seed):  # figure out the role of seed?
        super(neural_network, self).__init__()
        
        print("Input of net is: ",state_size)
        print("Action size is: ",action_size)
        
        # number of neurons in hidden layers
        self.Num_neurons=[64,64,128]
        self.seed = random.seed(seed)
        # define hidden layers
        # state_zise= dimension of a state 
        self.hidden1 = nn.Linear(state_size, self.Num_neurons[0])
        #self.bn1=nn.BatchNorm1d(self.Num_neurons[0])
        self.hidden2 = nn.Linear(self.Num_neurons[0],self.Num_neurons[1])
        #self.bn2=nn.BatchNorm1d(self.Num_neurons[1])
        self.hidden3 = nn.Linear(self.Num_neurons[1],self.Num_neurons[2])
        #self.bn3=nn.BatchNorm1d(self.Num_neurons[2])
        #self.hidden4 = nn.Linear(self.Num_neurons[2], self.Num_neurons[3])
        #self.apply(self._init_weights)
        self.output = nn.Linear(self.Num_neurons[2], action_size)
        
        # define optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        # define loss function  nn.SmoothL1Loss()
        self.loss=torch.nn.MSELoss()
        
        # set cpu or gpu
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    #initializes all weights from a Normal Distribution with mean 0 and standard deviation 1, all bias with zero
    #def _init_weights(self, module):
     #   if isinstance(module, nn.Linear):
      #      module.weight.data.normal_(mean=0.0, std=1)
       #     if module.bias is not None:
        #        module.bias.data.zero_()
    
    def forward (self,state_size):   # input state at each step
        #state=torch.flatten(state)
        #print("Input state shape is: ",state_size.shape)
        Out1 = F.relu(self.hidden1(state_size))
        Out2 = F.relu(self.hidden2(Out1))
        Out3 = F.relu(self.hidden3(Out2))
        #Out4 = F.relu(self.hidden4(Out3))
        Q_values = self.output(Out3)
        
        return Q_values
    
    
    
# define the dqn agent
class dqn_agent_T1:
    
    def __init__(self,state_size, action_size,seed):
        
        self.state=state_size
        self.action_size=action_size
        self.step=0
        
        self.seed=random.seed(seed)    # random seed (int)
        self.memory = deque(maxlen=experience_memory) 
        
        # main network
        self.main_network=neural_network(state_size, action_size,seed)        
        # target network
        self.target_network=neural_network(state_size, action_size,seed)
        self.target_network.load_state_dict(self.main_network.state_dict())
        self.target_network.eval()
   

    def reply_buffer(self,state,action,reward,next_state,done):
        # add transition to memory
        Transition=namedtuple('Transition',['state','action','reward','next_state','done'])
        transition=Transition(state,action,reward,next_state,done)
        if len(self.memory)>=experience_memory:
            self.memory.popleft()    # remove the first element of memory
            self.memory.append(transition)
        else:
            self.memory.append(transition)
         
        if len(self.memory)>=batch_size:
            loss=self.sample_buffer()
            return loss
        else:
            return None
        
        #print("Memory contains:",self.memory)

        
    #def epsilon_decay(self):
        ## decrease epsilon value
        #epsilon=max((Max_epsilon-Min_epsilon)*epsilon_decay+Min_epsilon,Min_epsilon)
        #Current_epsilon=0
        #if Max_epsilon > Min_epsilon:
        #    epsilon=(Max_epsilon-Min_epsilon)*epsilon_decay+Min_epsilon
        #    Max_epsilon=epsilon
        #else:
        #    epsilon=Min_epsilon
            
        #return epsilon 
    
    # action selection using epsilon strategy
    def choose_action(self,state,epsilon):
        
        #convert state from numpy to a float tensor
        state = torch.from_numpy(state).float().unsqueeze(0)
        
        # genertae random number
        random_number=random.random()
        if random_number < epsilon:
            action=random.randrange(self.action_size)
        else:
            # convert state to tensor
            self.main_network.eval()
            with torch.no_grad():
                action = self.main_network(state).argmax(dim=1).item()  #exploit
                self.main_network.train()
            #action_values=self.main_network(state)
            #action=np.argmax(action_values.cpu().data.numpy())
            
        return action
    

    # sample data from memory for training
    def sample_buffer(self):
        # sampling
        samples=random.sample(self.memory,batch_size)
        self.step+=1
            
        states=torch.from_numpy(np.vstack([i.state for i in samples if i is not None])).float()
        actions=torch.from_numpy(np.vstack([i.action for i in samples if i is not None])).type(torch.int64)
        rewards=torch.from_numpy(np.vstack([i.reward for i in samples if i is not None])).float()
        next_states=torch.from_numpy(np.vstack([i.next_state for i in samples if i is not None])).float()
        done=torch.from_numpy(np.vstack([i.done for i in samples if i is not None]).astype(np.uint8)).float()
        #return (states,actions,rewards,next_states,done)
            
        # main network will be in training mode
        self.main_network.train(mode=True)
            
        # predition for each transition in replay buffer
        Predicted_q_value=self.main_network.forward(states).gather(1, actions)
        #print ("Predicted_value:", Predicted_q_value)
        
        with torch.no_grad():
            Predicted_target_value=self.target_network(next_states).detach().max(1)[0].unsqueeze(1)
        
        # (1-done)=1 if done is False, otherwise is 0 for terminal state.
        Target_value=rewards+(gamma* Predicted_target_value*(1-done))
        #print ("Target_value:", Target_value)

            
        # loss 
        loss=self.main_network.loss(Predicted_q_value,Target_value).to(self.main_network.device)

        # set gradient to zero
        self.main_network.optimizer.zero_grad()
        # backpropagation
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), max_norm=1.0, norm_type=2.0)
        # update parameters
        self.main_network.optimizer.step()
            
        if self.step % update_weight==0: 
        # update weights of target network [theta_target = τ*theta_main + (1 - τ)*theta_target)]
            #self.target_network.load_state_dict(self.main_network.state_dict())
            for theta_main, theta_target in zip (self.main_network.parameters(),self.target_network.parameters()):
                    theta_target.data.copy_(tau*theta_main.data+(1-tau)*theta_target.data)

        return loss
             
    

                


                






        


        
        
        
        
        
        
        
        
        
        
        
        
        
        