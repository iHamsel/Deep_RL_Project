from numbers import Number
import torch
import random
import numpy as np
from torch.nn.functional import mse_loss, smooth_l1_loss
from DQN.DQN import DQN

from ReplayMemory import Memory, Transition
from DQN import DQN
from DecayValue import *
from Agent import Agent
from gym import Env








class DoubleAgent(Agent):
   def __init__(
      self,
      Model: DQN,
      env: Env,
      memory_length=5000,
      replay_batchsize=64,
      gamma=0.95,
      eps: DecayValue=ExponentialDecay(0.95, 0.01, 1e5)
   ):
      super().__init__(Model, env, memory_length, replay_batchsize, gamma, eps)


   def calcLoss(self, batch):
      states, actions, next_states, rewards = batch
      
      states = torch.stack(states).to(self.device)
      actions  = torch.LongTensor(actions).to(self.device)
      rewards  = torch.FloatTensor(rewards).to(self.device)
      non_final_mask    =  torch.BoolTensor( [s is not None for s in next_states]).to(self.device)
      non_final_states  =  torch.stack( [s for s in next_states if s is not None]).to(self.device)


      curr_Q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
      
      next_Q = self.policy_net(non_final_states)
      _, next_actions = next_Q.max(1)

      target_next_Q = self.target_net(non_final_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)

      expected_Q = rewards
      expected_Q[non_final_mask] += self.gamma * target_next_Q
      loss = smooth_l1_loss(curr_Q, expected_Q)
      return loss



