# This class generates the DQN-environment for the UAV mobility with altitude 100 to 140 meters

import numpy as np
from numpy import random
#%matplotlib inline
import matplotlib.pyplot as plt
from matplotlib.path import Path
from mpl_toolkits import mplot3d
import gym
from gym import spaces
from gym.utils import seeding
from gym.spaces import Discrete, MultiDiscrete
from gym import Env
import math
from sklearn.preprocessing import normalize


# Description
# This calss generates a 3d grid environment for a single-cell network with an uav and two ground users.
# Each ground user is assigned one resourece block. NOMA is used to manage the generate
# uav-interference on its co-channel ground users using finding optimal path, RB and power optimization for uav.
# ground users assigned the constant and equal power. 

# Action pairs: (direction,power,RB):
# totally there are 6 directions represent mobility of uav in a 3D environment:
    # move up in z-axis
    # move down in z-axis
    # move west in y-axis
    # move east in y-axis
    # move north in x-axis
    # move south in x-axis

# ground users are mobile and move randomly
# Observations space is the locations of uav and ground users
# the uav starts from an initial location and ends the episode at a terminal location.
# The environment is designed like a grid world which is splitted in squares with size x*y*z.

    
    
class DQNUAVenv_T2(gym.Env):
    
    def __init__ (self):
        
        # simulation parameters
        self.Num_GUE=2
        self.Bandwidth=50e6                          # 50MHz       
        self.f_c=2e9                                 # Carrier frequency equal to 2GHz
        self.N_0=-164                                # Noise power density equal to -174dBm/Hz
        self.Lambda=(3e8)/self.f_c                   # wavelength (m)
        self.GUE_power=20                            # GUE transmission power in dBm (100mw=0.1w)    
        self.BS_hight=15                             # BS hight 
        self.alpha=3.5
        self.rho=1.4e-4                              # channel gain at reference distance 1 meter (lambda/4*pi)^2
        self.SINR_threshold=15                        # SINR threshold in dB 
        self.UAV_max_power=20                        # maximum transmission power of the uav in dbm (100mw=0.1w)
        
        self.UAV_speed=20                            # speed of the uav 20m/s
        self.UAV_timestep=1                          # every second making uplink communication
        
        # energy parameters of uav
        #self.max_energy=100                          # uav energy(100KJ)
        #self.sum_energy=0                            # total energy usage of uav
        #self.energy_usage=0
        #self.current_state=0
        
        self.done=False
        self.reward=0
        self.Num_steps=200                           # the number of steps in a episode
        self.Data_Rate={"Rate":0, "Distance":0, "Rate_OMA":0}
        self.sum_distance=0
        self.seed()
        
        # action space in a 3D area consists of:
        # uav mobility (up,down for z-axis, east,west for y-axis, south,north for x-axis),uav power and resource blocks
        self.uav_power= np.array([14,16,18,20])   # dBm
        self.uav_move=['up','down','east','west','south','north']
        self.user=np.array([1,2])
        
        # action_pair generation array
        #self.power_vector=np.repeat(self.uav_power,len(self.uav_move))   # repeat each element of power
        #self.Move_vector=np.tile(self.uav_move,len(self.uav_power))      # repeat array of uav_move
        
        # generating action pairs (power,direction,RB)
        self.action_pairs=[]
        for p in range(len(self.uav_power)):
            for m in range(len(self.uav_move)):
                for u in range(len(self.user)):
                    self.action_pairs.append([self.uav_power[p],self.uav_move[m],self.user[u]])
        
        #self.action_space=Discrete(len(self.uav_move)*len(self.uav_power))
        self.action_space=Discrete(len(self.action_pairs))
        
        # map each action pair to a code (totally there are 48 action pairs)
        """
            0:  (5,up,RB1)                   9: (5,south,RB2)                
            1:  (5,up,RB2)                   10:(5,north,RB1)              
            2:  (5,down,RB1)                 11:(5,north,RB2)                
            3:  (5,down,RB2)                 12:(10,up,RB1)                 
            4:  (5,east,RB1)                 13:(10,up,RB2)               
            5:  (5,east,RB2)                 14:(10,down,RB1)                
            6:  (5,west,RB1)                      ...
            7:  (5,west,RB2)               
            8:  (5,south,RB1)                 
        """
        
        # environment dimensions (m)
        self.Min_x=0
        self.Max_x=1500
        self.Min_y=0
        self.Max_y=1500
        self.Min_z=0
        self.Max_z=140
        self.Min_altitude=80
        self.Max_altitude=120
        
        # Ground BS location with hight=15m
        self.BS_location=np.array([self.Max_x/2,self.Max_y/2,self.BS_hight])
        
        # ground users configuration: there are 2 GUEs which follow random waypoint (RWP) mobility in a 2D area.
        # two ground users start from two different locations
        # generating random mobility model 
        random.seed(0)
        
        # speed of ground users (m/s)
        self.gue_speed=np.array([5,6])
        
        self.Initial_loc1=np.array([random.randint(self.Min_x,self.Max_x),random.randint(self.Min_y,self.Max_y)])
        self.Initial_loc2=np.array([random.randint(self.Min_x,self.Max_x),random.randint(self.Min_y,self.Max_y)])
        self.GUE1_loc=np.zeros((self.Num_steps,3),dtype=np.float64)    
        self.GUE2_loc=np.zeros((self.Num_steps,3),dtype=np.float64)
        # considering height 1m for two ground users
        self.GUE1_loc[:,2]=1
        self.GUE2_loc[:,2]=1

        for i in range(self.Num_steps):
            if i==0:
                self.GUE1_loc[i,0]=self.Initial_loc1[0]
                self.GUE1_loc[i,1]=self.Initial_loc1[1]
                self.GUE2_loc[i,0]=self.Initial_loc2[0]
                self.GUE2_loc[i,1]=self.Initial_loc2[1]
            else:
                speed_gue1=random.choice(self.gue_speed)
                direction_gue1=random.randint(0,360)
                self.GUE1_loc[i,0]=self.GUE1_loc[i-1,0]+(speed_gue1*(math.cos(direction_gue1)))
                self.GUE1_loc[i,1]=self.GUE1_loc[i-1,1]+(speed_gue1*(math.sin(direction_gue1)))
                if self.GUE1_loc[i,0]>=self.Max_x:
                    self.GUE1_loc[i,0]=(self.Max_x)-1
                elif self.GUE1_loc[i,0]<=(self.Min_x):
                    self.GUE1_loc[i,0]=self.Min_x+1
                if self.GUE1_loc[i,1]>=self.Max_y:
                    self.GUE1_loc[i,1]=(self.Max_y)-1
                elif self.GUE1_loc[i,1]<=self.Min_y:
                    self.GUE1_loc[i,1]=(self.Min_y+1)
            
                speed_gue2=random.choice(self.gue_speed)
                direction_gue2=random.randint(0,360)
                self.GUE2_loc[i,0]=self.GUE2_loc[i-1,0]+(speed_gue2*(math.cos(direction_gue2)))
                self.GUE2_loc[i,1]=self.GUE2_loc[i-1,1]+(speed_gue2*(math.sin(direction_gue2)))
                if self.GUE2_loc[i,0]>=self.Max_x:
                    self.GUE2_loc[i,0]=(self.Max_x)-1
                elif self.GUE2_loc[i,0]<=self.Min_x:
                    self.GUE2_loc[i,0]=(self.Min_x+1)
                if self.GUE2_loc[i,1]>=(self.Max_y):
                    self.GUE2_loc[i,1]=(self.Max_y)-1
                elif self.GUE2_loc[i,1]<=self.Min_y:
                    self.GUE2_loc[i,1]=(self.Min_y+1)
         
        # round float values to integer for gue locations
        self.GUE1_loc=np.asarray((np.ceil(self.GUE1_loc)),dtype='int')
        self.GUE2_loc=np.asarray((np.ceil(self.GUE2_loc)),dtype='int')
        self.GUE1_loc[self.GUE1_loc==(self.Max_x)]=(self.Max_x)-1
        self.GUE2_loc[self.GUE2_loc==(self.Max_x)]=(self.Max_x)-1
        #print(self.GUE1_loc)
        #print(self.GUE2_loc)
        
        # normalizing their location for state space
        #self.GUE1_loc=(self.GUE1_loc-self.GUE1_loc.min(axis=0))/(self.GUE1_loc.max(axis=0)-self.GUE1_loc.min(axis=0))
        #self.GUE2_loc=(self.GUE2_loc-self.GUE2_loc.min(axis=0))/(self.GUE2_loc.max(axis=0)-self.GUE2_loc.min(axis=0))
        
        # calculating channel gain of ground users to the BS
        # ue-BS channel is modeled using effect of pathloss, shadowing and  Rayleigh fading parameter
        """
        pathloss_ue(dB)=37.6 log10(d_i) − 18 log10(h_BS) + 21 log10(f_c) + 80dB + X_c
        
        d_i: distance from BS to user (km)
        h_BS: BS height (m)
        f_c: carrier frequency (MHz)
        X_c: shadowing effect (standard deviation of 10dB)
        
        h_ue=g_i(10^(−PL(dB)/10)) 
        
        """
        # saving information
        self.distance_gue1_BS=np.zeros((len(self.GUE1_loc),1),dtype=np.float64)
        self.distance_gue2_BS=np.zeros((len(self.GUE2_loc),1),dtype=np.float64)
        self.pathloss_gue1=np.zeros((len(self.GUE1_loc),1),dtype=np.float64)
        self.pathloss_gue2=np.zeros((len(self.GUE2_loc),1),dtype=np.float64)
        self.channel_gue1=np.zeros((len(self.GUE1_loc),1),dtype =np.float64)
        self.channel_gue2=np.zeros((len(self.GUE2_loc),1),dtype =np.float64)
        np.random.seed(0)
        #random.seed(0)
        for j in range(self.Num_steps):
            # distance to the BS
            # user1
            self.distance_gue1_BS[j]=(np.sqrt((self.GUE1_loc[j,0]-self.BS_location[0])**2+
                                              (self.GUE1_loc[j,1]-self.BS_location[1])**2+(self.GUE1_loc[j,2]-self.BS_location[2])**2))
            # user2
            self.distance_gue2_BS[j]=(np.sqrt((self.GUE2_loc[j,0]-self.BS_location[0])**2+
                                              (self.GUE2_loc[j,1]-self.BS_location[1])**2+(self.GUE2_loc[j,2]-self.BS_location[2])**2))
            
            # channel gain calculation
            #https://electronics.stackexchange.com/questions/311821/how-to-include-rayleigh-fading-into-path-loss-model
            # user1
            #self.pathloss_gue1[j]=((37.6*math.log10(self.distance_gue1_BS[j]/1e3))-(18*math.log10(self.BS_hight))+
            #                       (21*math.log10(self.f_c/1e6))+80+np.random.normal(0,10)) 
            self.pathloss_gue1[j]=((20*math.log10((4*math.pi)/(self.Lambda)))+(10*self.alpha*math.log10(self.distance_gue1_BS[j]))+
                                  np.random.normal(0,1))
            #self.channel_gue1[j]=(np.random.randn()+np.random.randn()*1j)*(10**(-self.pathloss_gue1[j]/10))
            self.channel_gue1[j]=self.pathloss_gue1[j]+np.random.rayleigh(1,1)[0]
            
            # user2
            #self.pathloss_gue2[j]=((37.6*math.log10(self.distance_gue2_BS[j]/1e3))-(18*math.log10(self.BS_hight))+
            #                       (21*math.log10(self.f_c/1e6))+80+np.random.normal(0,10))
            self.pathloss_gue2[j]=((20*math.log10((4*math.pi)/(self.Lambda)))+(10*self.alpha*math.log10(self.distance_gue2_BS[j]))+
                                  np.random.normal(0,1))
            self.channel_gue2[j]=self.pathloss_gue2[j]+np.random.rayleigh(1,1)[0]
            #self.channel_gue2[j]=(np.random.randn()+np.random.randn()*1j)*(10**(-self.pathloss_gue2[j]/10))
        
        
        # initial and terminal location of the uav: [(-100,-100),(100,100)]
        self.uav_initial_loc=np.array([220,200,self.Min_altitude])
        self.uav_terminal_loc=np.array([200,1100,self.Max_altitude])
        
        # observation space includes users locations (aerial and ground users).
        self.x_uav_loc=np.arange(self.Min_x,self.uav_initial_loc[0]+1)
        self.y_uav_loc=np.arange(self.Min_y,self.uav_terminal_loc[1]+1)
        self.z_uav_loc=np.arange(self.Min_z,self.Max_altitude+1)
        
        self.x_gue_loc=np.arange(self.Min_x,self.Max_x+1)  # area from 100 to 250 
        self.y_gue_loc=np.arange(self.Min_y,self.Max_y+1)
        self.z_gue_loc=np.arange(0,2)        # height 1m for gue
        
        self.observation_space=spaces.MultiDiscrete([len(self.x_uav_loc),len(self.y_uav_loc),len(self.z_uav_loc),   # 23*23*3
                                                    len(self.x_gue_loc),len(self.y_gue_loc),len(self.z_gue_loc),
                                                    len(self.x_gue_loc),len(self.y_gue_loc),len(self.z_gue_loc)])
        
        
        print(self.observation_space.shape)
        
        self.state=np.array(np.concatenate((self.uav_initial_loc,self.GUE1_loc[0],self.GUE2_loc[0])))
        self.last_state=np.array(np.concatenate((self.uav_terminal_loc,self.GUE1_loc[-1],self.GUE2_loc[-1])))
        #self.low_observation=np.array([self.Min_x,self.Min_y,1])
        #self.high_observation=np.array([self.Max_x,self.Max_y,self.Max_altitude])
        #self.observation_space=spaces.Box(low=self.Min_x, high=self.Max_x, shape=(9,), dtype=int)
        
        self.total_distance=(np.sqrt((self.uav_initial_loc[0]-self.uav_terminal_loc[0])**2+
                                     (self.uav_initial_loc[1]-self.uav_terminal_loc[1])**2+(self.uav_initial_loc[2]-
                                                                                           self.uav_terminal_loc[2])**2))
       
    
    # generator to return a random number based on generated seed
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    # select the action and update the state
    def step(self,action):

        # initial state
        #x_uav,y_uav,z_uav=self.state
        state=self.state
        if (state==np.array(np.concatenate((self.uav_initial_loc,self.GUE1_loc[0],self.GUE2_loc[0])))).all():
            #self.sum_energy=0
            self.sum_distance=0
            #print(f"Energy is: {self.sum_energy}")
        
        x_uav,y_uav,z_uav=state[0:3]
        # return the index of that location in array
        index_GUE1=np.where((self.GUE1_loc == (state[3:6])).all(axis=1))
        index_GUE2=np.where((self.GUE2_loc == (state[6:9])).all(axis=1))
        
        power_uav,direction_uav,user=self.action_pairs[action]
        #power_uav,direction_uav=self.action_pairs[action]
        
        if direction_uav=='up':
        # (up z-axis)
            new_x_uav=x_uav
            new_y_uav=y_uav
            new_z_uav=min(z_uav+(self.UAV_speed*self.UAV_timestep), self.Max_altitude)
            
        # (down z-axis)
        if direction_uav=='down':
            new_x_uav=x_uav
            new_y_uav=y_uav
            new_z_uav=max(self.Min_altitude,z_uav-(self.UAV_speed*self.UAV_timestep))
          
        #(west y-axis)
        if direction_uav=='west':
            new_x_uav=x_uav
            new_y_uav=max(y_uav-(self.UAV_speed*self.UAV_timestep),np.min(self.y_uav_loc))
            new_z_uav=z_uav
        
        # (east y-axis)
        if direction_uav=='east':
            new_x_uav=x_uav
            new_y_uav=min(y_uav+(self.UAV_speed*self.UAV_timestep),np.max(self.y_uav_loc))
            new_z_uav=z_uav

        #(south x-axis)
        if direction_uav=='south':
            new_x_uav=min(x_uav+(self.UAV_speed*self.UAV_timestep),np.max(self.x_uav_loc))
            new_y_uav=y_uav
            new_z_uav=z_uav

        #(north x-axis)
        if direction_uav=='north':
            new_x_uav=max(x_uav-(self.UAV_speed*self.UAV_timestep),np.min(self.x_uav_loc))
            new_y_uav=y_uav
            new_z_uav=z_uav
        
        # next uav location
        next_uav_loc=np.array([new_x_uav,new_y_uav,new_z_uav])
        
        # next ground users locations
        next_GUE1_loc=self.GUE1_loc[index_GUE1[0]+1]
        next_GUE2_loc=self.GUE2_loc[index_GUE2[0]+1]
       
        # update state to next one
        #self.state=np.concatenate((next_uav_loc,next_GUE1_loc,next_GUE2_loc)).reshape((3,3))
        self.state=np.array(np.concatenate((next_uav_loc,next_GUE1_loc[0],next_GUE2_loc[0])),dtype='int')
        
        # check the output for this
        #next_state=int(np.asarray(np.where((self.states == (next_uav_loc)).all(axis=1))))
        
        ## check whether uav is in a terminal location or not
        #if next_state==int(np.asarray(np.where((self.states == (self.terminal_uav_loc)).all(axis=1)))):
        #    self.done=True
        
        # terminal state is the last location of the uav
        #if (self.state[0:3]==self.last_state[0:3]).all():
        #   self.done=True
        
        """
        h_uav= ρ_0(d^(-2))=
        
        ρ_0: channel power gain at reference distance d_0=1m
        d: 3d distance between BS and uav (m)     
        """
        
        #Channel_uav= ((self.rho)*((np.sqrt((next_uav_loc[0]-self.BS_location[0])**2+(next_uav_loc[1]-self.BS_location[1])**2+
        #                               (next_uav_loc[2]-self.BS_location[2])**2))**-2))
        uav_BS_distance=np.sqrt((next_uav_loc[0]-self.BS_location[0])**2+(next_uav_loc[1]-self.BS_location[1])**2+
                                      (next_uav_loc[2]-self.BS_location[2])**2)
        
        Ch_uav=20*math.log10((4*math.pi*uav_BS_distance)/(self.Lambda))     # in dB
        
        # SINR calculation ===> [[]] the reason why i used [0] at front of gue channels==> []
        #Ch_users=[Ch_uav,self.channel_gue1[index_GUE1[0]+1][0],self.channel_gue2[index_GUE2[0]+1][0]]
        #Power_users=[power_uav,self.GUE_power,self.GUE_power]
        if user==1:
            Ch_users=[Ch_uav,self.channel_gue1[index_GUE1[0]+1][0]]
            Power_users=[power_uav,self.GUE_power]
            
            # rate of non-shared user
            SINR_OMA=((self.GUE_power-abs(self.channel_gue2[index_GUE2[0]+1][0]))-
                      (self.N_0+(10*math.log10(self.Bandwidth/len(self.user)))))
            Rate_OMA=((self.Bandwidth/len(self.user))*math.log2(1+(10**(SINR_OMA/10))))/1e6
            
        elif user==2:
            Ch_users=[Ch_uav,self.channel_gue2[index_GUE2[0]+1][0]]
            Power_users=[power_uav,self.GUE_power]
            
            # rate of non-shared user
            SINR_OMA=((self.GUE_power-abs(self.channel_gue1[index_GUE1[0]+1][0]))-
                      (self.N_0+(10*math.log10(self.Bandwidth/len(self.user)))))
            Rate_OMA=((self.Bandwidth/len(self.user))*math.log2(1+(10**(SINR_OMA/10))))/1e6
            
        
        info=list(zip(Ch_users,Power_users))
        # sort info based on channel gain Magnitude (largest to smallest)
        info.sort(reverse=True)
        
        #self.energy_usage,self.sum_energy=self.energy(state,next_uav_loc)
        Rate,self.reward,self.done=self.reward_fun(info,state,next_uav_loc)
        self.Data_Rate.update({"Rate":Rate,"Distance":self.sum_distance, "Rate_OMA":Rate_OMA})
        
        return self.state,self.reward,self.done,self.Data_Rate
    
    
    #def energy(self,state,next_uav_loc):
        # parameters
    #    m=4         # kg
    #    g=9.81      # m/s^2
    #    rho=0.18    # m^2
    #    a=1.225     # kg/m^3
    #    landa=0.08
    #    E_blade=(1/8)*landa*rho*a*(math.pow(self.UAV_speed,3))
    #    E_hov=((m*g)**(3/2))/(np.sqrt(2*rho*a))
        # energy for hovering
    #    if (next_uav_loc[0]==state[0] and next_uav_loc[1]==state[1] and next_uav_loc[2]==state[2]):
    #        self.energy_usage=E_hov
            
        # energy for going up 
     #   elif (next_uav_loc[0]==state[0]) and (next_uav_loc[1]==state[1]) and (next_uav_loc[2]>state[2]):
     #       self.energy_usage=(m*g*self.UAV_speed)+E_blade

        # energy for going down        
    #    elif (next_uav_loc[0]==state[0]) and (next_uav_loc[1]==state[1]) and (next_uav_loc[2]<state[2]):
    #        self.energy_usage=(-m*g*self.UAV_speed)+E_blade
        
        # energy for horizontal move
    #    else:
    #        self.energy_usage=(E_blade+((((m*g)**2)/(np.sqrt(2)*rho*a))*
    #                                    (1/(np.sqrt(math.pow(self.UAV_speed,2)+np.sqrt(math.pow(self.UAV_speed,4)+4*
    #                                                                                   math.pow(E_hov,4)))))))
    
    #    self.sum_energy+=(self.energy_usage)*(1e-3)
        
    #    return self.energy_usage,self.sum_energy
    
    
    # restart the episode and return the uav to the initial state (location)
    def reset(self,seed=None,options=None):
        #self.current_state=0
        self.state=np.array(np.concatenate((self.uav_initial_loc,self.GUE1_loc[0],self.GUE2_loc[0])))
        self.done=False
        self.Data_Rate.update({"Rate":0,"Distance":0,"Rate_OMA":0})
        self.sum_distance=0
        #self.sum_energy=0
        #self.energy_usage=0
        self.reward=0
        return self.state
    
    # calculate reward for each transition 
    def reward_fun(self,info,state,next_uav_loc):
        
        #constant_reward=100
        #sigma=80/100
        SINR=np.zeros((len(info),1),dtype=np.float64)                        
        Rate=0
        reward=0
        done=False
        w_1=0.7
        w_2=0.3
        
        # distance to terminal location
        dis=(np.sqrt((self.uav_terminal_loc[0]-next_uav_loc[0])**2+(self.uav_terminal_loc[1]-next_uav_loc[1])**2)
             +(self.uav_terminal_loc[2]-next_uav_loc[2])**2)
        # calculate SINR
        for r in range(len(info)):
            Noise=self.N_0+(10*math.log10(self.Bandwidth/len(self.user)))
            #SINR[r]=((abs(info[r][0])**2)*(10**((info[r][1]-30)/10))/(((self.Bandwidth/len(self.user))*(10**((self.N_0-30)/10)))+
            #    (np.sum(np.fromiter(((abs(info[s][0])**2)*(10**((info[s][1]-30)/10)) for s in range(r+1,len(info))),float)))))
            if r==len(info)-1:
                SINR[r]=(info[r][1]-abs(info[r][0]))-((Noise))
            else:
                SINR[r]=(info[r][1]-abs(info[r][0]))-((Noise)+(info[r+1][1]-abs(info[r+1][0])))
                #SINR[r]=((info[r][1]-abs(info[r][0])**2)-((Noise)+
                #                                      (np.sum((info[s][1]-abs(info[s][0])**2) for s in range (r+1,len(info))))))
       
        #print(SINR) 
        Distance=np.sqrt((next_uav_loc[0]-state[0])**2+(next_uav_loc[1]-state[1])**2+(next_uav_loc[2]-state[2])**2)
        self.sum_distance+=Distance
        #print(self.sum_distance)
        
        if dis>self.total_distance:
            reward=0
            done=False
            
        # check the condition
        #if (dis<=self.total_distance and dis!=0):
        elif (dis<=self.total_distance and dis!=0):
            if  all(x >= self.SINR_threshold for x in SINR):
                Rate=((self.Bandwidth/len(self.user))*(np.sum((math.log2(1+(10**(SINR[d]/10)))) for d in range(len(SINR))))/1e6)
                
                #reward=math.log10((Rate)+(1/self.sum_distance)+(1/dis))
                #reward=Rate-0.5*(dis)-0.5*(self.sum_distance)
                
                reward=(w_1*(Rate/self.sum_distance))+(w_2*(self.total_distance/dis))
                #print(f"reward is:{reward}")
                #rate/max(rate)
                #reward=math.log10(Rate+(1-(dis/self.total_distance))+(1-(self.sum_energy/self.max_energy)))  
                done=False
            else:
                reward=0
                done=False
                      
        elif dis==0:
            if  all(x >= self.SINR_threshold for x in SINR):
                Rate=((self.Bandwidth/len(self.user))*(np.sum((math.log2(1+(10**(SINR[d]/10)))) for d in range(len(SINR))))/1e6)
                #print(f"Rate is:{Rate}")
                #reward=Rate-(self.sum_distance)
                reward=(w_1*(Rate/self.sum_distance))
                #print(f"reward is:{reward}")
                #reward=math.log10(Rate+1+(1-(self.sum_energy/self.max_energy)))  
                done=True
            else:
                reward=0
                done=True
   
        return Rate,reward,done
    
    
    # visalization of uav trajectory
    def path_plot(self,x_array,y_array,z_array):
        
        #,x_array,y_array,z_array
        fig1 = plt.figure(figsize=(8,8))
        ax = plt.axes(projection='3d')
        
        # scatter initial and terminal points
        ini_uav_x, ini_uav_y, ini_uav_z=self.uav_initial_loc
        ter_uav_x, ter_uav_y, ter_uav_z=self.uav_terminal_loc
        ax.scatter3D(ini_uav_x, ini_uav_y, ini_uav_z, s=60, c='red',marker='*',label='Initial location')
        ax.scatter3D(ter_uav_x, ter_uav_y, ter_uav_z, s=60, c='red',marker="X",label='Terminal location')
        
        # scatter BS location and ground users
        ax.scatter3D(self.BS_location[0],self.BS_location[1],self.BS_location[2],s=100, c='black',marker="1",label='BS location')
        
        # plot random trajectory for ground users
        #ax.scatter3D(self.GUE1_loc[0,0],self.GUE1_loc[0,1],self.GUE1_loc[0,2],s=20, c='blue',marker="o")
        #ax.scatter3D(self.GUE2_loc[0,0],self.GUE2_loc[0,1],self.GUE2_loc[0,2],s=20, c='orange',marker="o")
        
        #ax.scatter3D(self.GUE1_loc[-1,0],self.GUE1_loc[-1,1],self.GUE1_loc[-1,2],s=20, c='blue',marker="o")
        #ax.scatter3D(self.GUE2_loc[-1,0],self.GUE2_loc[-1,1],self.GUE2_loc[-1,2],s=20, c='orange',marker="o")
        
        plt.plot(self.GUE1_loc[:,0],self.GUE1_loc[:,1],self.GUE1_loc[:,2],c='blue')
        plt.plot(self.GUE2_loc[:,0],self.GUE2_loc[:,1],self.GUE2_loc[:,2],c='orange')

        #ax.scatter3D(self.ue1_loc[0],self.ue1_loc[1],0,s=40, c='black',marker='o')
        #ax.scatter3D(self.ue2_loc[0],self.ue2_loc[1],0,s=40, c='black',marker='o')
        
        # plot the trajectory
        x_array.insert(0, ini_uav_x)
        y_array.insert(0, ini_uav_y)
        z_array.insert(0, ini_uav_z)
        x_array.append(ter_uav_x)
        y_array.append(ter_uav_y)
        z_array.append(ter_uav_z)
        
        line = plt.plot(x_array, y_array, z_array, lw=2, c='green',marker='*')[0] 
        
        plt.legend(('Initial location','Terminal location','BS location','Mobility user1','Mobility user2','UAV-trajectory')
                   ,loc='upper right',fontsize="8",ncol=2)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        #ax.set_ticklabel_format(axis='y', style='sci')
        #ax.set_title('Generated UAV-trajectory with less interference')
        ax.set_xlim(self.Min_x, self.Max_x)
        ax.set_ylim(self.Min_y, self.Max_y)
        ax.set_zlim(0, self.Max_altitude+50)
        ax.view_init(20, 8)
        #plt.show()
        plt.savefig(r'C:\Users\student\OneDrive - Carleton University (1)\Erricson\Third paper\Simulation_DRL\Plots\Trajectory_T2.pdf')

        fig2 = plt.figure()
        ax = plt.axes(projection='3d')
        plt.plot(self.GUE1_loc[:,0],self.GUE1_loc[:,1],self.GUE1_loc[:,2],c='blue')
        plt.savefig(r'C:\Users\student\OneDrive - Carleton University (1)\Erricson\Third paper\Simulation_DRL\Plots\U1.pdf')
        #plt.plot(self.GUE2_loc[:,0],self.GUE2_loc[:,1],self.GUE2_loc[:,2])
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.view_init(20, 8)
        ax.set_xlim(min(self.GUE1_loc[:,0]-50),max(self.GUE1_loc[:,0]+50))
        ax.set_ylim(min(self.GUE1_loc[:,1]-50),max(self.GUE1_loc[:,1]+50))
        ax.set_zlim(0, 10)
       
        
        fig3 = plt.figure()
        ax = plt.axes(projection='3d')
        #plt.plot(self.GUE1_loc[:,0],self.GUE1_loc[:,1],self.GUE1_loc[:,2])
        plt.plot(self.GUE2_loc[:,0],self.GUE2_loc[:,1],self.GUE2_loc[:,2],c='orange')
        plt.savefig(r'C:\Users\student\OneDrive - Carleton University (1)\Erricson\Third paper\Simulation_DRL\Plots\U2.pdf')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.view_init(20, 8)
        ax.set_xlim(min(self.GUE2_loc[:,0]-50),max(self.GUE2_loc[:,0]+50))
        ax.set_ylim(min(self.GUE2_loc[:,1]-50),max(self.GUE2_loc[:,1]+50))
        ax.set_zlim(0, 10)
        
        
        
        
        
        
        
        
        
        
        
  