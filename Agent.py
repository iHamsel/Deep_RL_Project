from numbers import Number
from os import stat
import torch
import random
import numpy as np
from torch.nn.functional import mse_loss, smooth_l1_loss
from DQN.DQN import DQN
from gym import Env

from ReplayMemory import Memory, Transition
from DQN import DQN
from DecayValue import *


class Agent():
   def __init__(
      self,
      Model: DQN,
      env: Env,
      memory_length=5000,
      replay_batchsize=64,
      gamma=0.95,
      eps: DecayValue=ExponentialDecay(0.95, 0.01, 1e5)
   ):
      assert issubclass(Model, DQN), "Parameter 'Model' is not a subclass of 'DQN'!"
      assert isinstance(eps, DecayValue), "Parameter 'eps' is not an instance of 'DecayValue' or a subclass of that!"
      assert gamma >= 0 and gamma <= 1, "Parameter 'gamma' have to be 0 <= gamma <= 1!"


      self.memory       = Memory(memory_length)
      self.batch_size   = replay_batchsize
      self.n_actions    = env.action_space.n
      self.eps          = eps
      self.gamma        = gamma


      self.device       = "cuda" if torch.cuda.is_available() else "cpu"
      self.policy_net   = Model(env.observation_space, env.action_space).to(self.device)
      self.target_net   = Model(env.observation_space, env.action_space).to(self.device)

      self.updateTargetNetwork()
      self.target_net.eval()
      
      self.optimizer    = torch.optim.RMSprop(self.policy_net.parameters(), lr=1e-4)


   def calcLoss(self, batch) -> torch.Tensor:
      """
         Calculate the loss of a given batch
      """
      states, actions, next_states, rewards = batch
      
      states = torch.stack(states).to(self.device)
      actions  = torch.LongTensor(actions).to(self.device)
      rewards  = torch.FloatTensor(rewards).to(self.device)
      non_final_mask    =  torch.BoolTensor( [s is not None for s in next_states]).to(self.device)
      non_final_states  =  torch.stack( [s for s in next_states if s is not None]).to(self.device)
      
      curr_Q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
      next_Q = self.target_net(non_final_states)
      max_next_Q = next_Q.max(1)[0]

      expected_Q = rewards
      expected_Q[non_final_mask] += self.gamma * max_next_Q

      loss = smooth_l1_loss(curr_Q, expected_Q)
      return loss

   def learn(self):
      """
         Learn with experience replay
      """
      if len(self.memory) < self.batch_size:
         return
      
      transitions = self.memory.sample(self.batch_size)
      batch = Transition(*zip(*transitions))

      loss = self.calcLoss(batch)
      
      self.optimizer.zero_grad()
      loss.backward()
      # torch.nn.utils.clip_grad.clip_grad_norm_(self.policy_net.parameters(), 5)
      self.optimizer.step()
      self.eps.step()

   def updateTargetNetwork(self):
      """
         Load the current state of the policy network into the target network
      """
      self.target_net.load_state_dict(self.policy_net.state_dict())      


   def sample_action(self, state) -> Number:
      """
         Sample a action
      """
      # Explore
      if random.random() <=self.eps.getValue():
         return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)

      state = torch.reshape(state, (1, *state.shape)).to(self.device)
      # Exploit
      with torch.no_grad():
         return self.policy_net(state).max(1)[1].view(1, 1)
         

