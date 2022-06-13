from numbers import Number
from os import stat
from more_itertools import sample
import torch
import random
import numpy as np
from torch.nn.functional import mse_loss, smooth_l1_loss
from DQN.DQN import DQN
from gym import Env

from ReplayMemory import ReplayMemory
from DQN import DQN
from DecayValue import *
from AgentConfiguration import AgentConfiguration


class Agent():

   def __init__(self, config: AgentConfiguration):
      self.config       = config
      self.memory       = ReplayMemory(config.memory_size)
      self.batch_size   = config.learningSize
      self.n_actions    = config.env.action_space.n
      self.eps          = config.epsilon
      self.gamma        = config.gamma
      self.env          = config.env

      self.device       = "cuda" if torch.cuda.is_available() else "cpu"
      self.policy_net   = config.network(config.env.observation_space, config.env.action_space).to(self.device)
      self.target_net   = config.network(config.env.observation_space, config.env.action_space).to(self.device)

      self.updateTargetNetwork()
      self.target_net.eval()
      self.optimizer    = torch.optim.RMSprop(self.policy_net.parameters(), lr=0.25e-4, alpha=0.95, momentum=0.95, eps=0.01)

      self.trainingRewards    = []
      self.evaluationRewards  = []

      self.mode = "train"


   def old__init__(
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


      self.memory       = ReplayMemory(memory_length)
      self.batch_size   = replay_batchsize
      self.n_actions    = env.action_space.n
      self.eps          = eps
      self.gamma        = gamma
      self.env          = env


      self.device       = "cuda" if torch.cuda.is_available() else "cpu"
      self.policy_net   = Model(env.observation_space, env.action_space).to(self.device)
      self.target_net   = Model(env.observation_space, env.action_space).to(self.device)

      self.updateTargetNetwork()
      self.target_net.eval()
      
      self.optimizer    = torch.optim.RMSprop(self.policy_net.parameters(), lr=0.25e-4, alpha=0.95, momentum=0.95, eps=0.01)


   def calcLoss(self, batch) -> torch.Tensor:
      """
         Calculate the loss of a given batch
      """
      states, actions, next_states, rewards = batch
      
      states = torch.stack(states).to(self.device)
      actions  = torch.stack(actions).to(self.device)
      rewards  = torch.stack(rewards).to(self.device)

      non_final_mask    =  torch.BoolTensor( [s is not None for s in next_states])
      non_final_states  =  torch.stack( [s for s in next_states if s is not None]).to(self.device)
      
      curr_Q = self.policy_net(states).gather(1, actions.squeeze(1)).squeeze(1)
      next_Q = self.target_net(non_final_states)
      max_next_Q = next_Q.max(1)[0]

      expected_Q = rewards
      expected_Q[non_final_mask] += self.gamma * max_next_Q

      loss = mse_loss(curr_Q, expected_Q)
      return loss

   def learn(self):
      """
         Learn with experience replay
      """
      if len(self.memory) < self.batch_size:
         return
      
      transitions = self.memory.sample(self.batch_size)
      batch = zip(*transitions)

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
      if self.mode == "train" and random.random() <=self.eps.getValue():
         return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)

      state = torch.reshape(state, (1, *state.shape)).to(self.device)
      # Exploit
      with torch.no_grad():
         return self.policy_net(state).max(1)[1].view(1, 1)


   def train(self, episodes):
      self.mode = "train"
      for _ in range(episodes):
         state = self.env.reset()
         episodeReward = 0

         while True:
            prev_state = state
            state, done, reward, action = self.play(state)

            episodeReward += reward
            next_state = state if done == False else None

            self.memory.append(prev_state, action, next_state, reward)
            self.learn()

            if done == True:
               break

         self.trainingRewards.append(episodeReward)
         print(f"Reward for training episode {len(self.trainingRewards)}: {episodeReward}")

   def eval(self, episodes):
      self.mode = "eval"
      for _ in range(episodes):
         state = self.env.reset()
         episodeReward = 0

         while True:
            state, done, reward, _ = self.play(state)
            episodeReward += reward

            if done == True:
               break

         self.evaluationRewards.append(episodeReward)
         print(f"Reward for evaluation episode {len(self.evaluationRewards)}: {episodeReward}")
  

   def play(self, currentState):
      reward = 0
      done = False
      state = None

      action = self.sample_action(currentState)

      for _ in range(4):
         state, r, done, _ = self.env.step(action)
         reward += r

      return state, done, reward, action

