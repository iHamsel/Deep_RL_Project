from dataclasses import dataclass
from http.client import NETWORK_AUTHENTICATION_REQUIRED
from re import A
from DecayValue import ExponentialDecay
from DQN import DQN, SingleHead
import gym
import torch

@dataclass
class AgentConfiguration:
   network:                DQN
   env:                    gym.Env 

   memory_size:            int   = 5000   #Size of the memory
   learningSize:           int   = 32     #How many frames are sampled in one learning iteration
   gamma:                  float = 0.95   #Factor how important the long-term reward is
   learnrate:              float = 0.25e-4


   repeatAction:           int   = 4      #Repeat a selected action n times in one play iteration
   learnInterval:          int   = 4      #Learn after n play iterations
   targetUpdateInterval:   int   = 1e4    #Update the target network n learning iterations

   epsilon:                int   = ExponentialDecay(1, 1e-2, 1e6)    #Selecting algorithm for epsilon-greedy