from math import sqrt
from numbers import Number
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
from TrainingEpisodeLogEntry import TrainingEpisodeLogEntry

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
      self.optimizer    = torch.optim.RMSprop(self.policy_net.parameters(), lr=config.learnrate, alpha=0.95, momentum=0.95, eps=0.01)

      self.log = []
      self.trainingRewards    = []
      self.evaluationRewards  = []
      self.training_playSteps = 0
      self.learned = 0
      self.updated = 0

      self.mode = "prefill"

      self.fillMemory()


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
         return 0
      
      transitions = self.memory.sample(self.batch_size)
      batch = zip(*transitions)

      loss = self.calcLoss(batch)
      self.optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
      self.optimizer.step()
      self.eps.step()
      self.learned += 1
      if self.learned % self.config.targetUpdateInterval == 0:
         self.updateTargetNetwork()
         self.updated += 1    
      return float(loss)

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
      if self.mode == "prefill" or (self.mode == "train" and random.random() <=self.eps.getValue()):
         return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)

      state = torch.reshape(state, (1, *state.shape)).to(self.device)
      # Exploit
      with torch.no_grad():
         return self.policy_net(state).max(1)[1].view(1, 1)


   def train(self, episodes):
      self.mode = "train"
      self.policy_net.train()
      for _ in range(episodes):
         state = self.env.reset()
         episodeReward = 0
         losses = []
         repeated = 0
         last_action = -1
         while True:
            prev_state = state
            state, done, reward, action = self.play(state)

            episodeReward += reward
            next_state = state if done == False else None

            if action == last_action:
               repeated += 1
            else:
               repeated = 0
            
            last_action = action

            self.memory.append(prev_state, action, next_state, reward)
            self.training_playSteps += 1
            if self.training_playSteps % self.config.learnInterval == 0:
               losses.append(self.learn())

            if done == True:
               break
         avgLoss = sum(losses)/len(losses) if len(losses) > 0 else 0
         self.log.append(TrainingEpisodeLogEntry(episodeReward, avgLoss, self.training_playSteps, self.learned, self.updated))
         print(f"Reward for training episode {len(self.log)}: {episodeReward} | Avg loss: {self.log[-1].avgLoss} |  Epsilon value: {self.eps.getValue()}")

   def eval(self, episodes):
      self.mode = "eval"
      self.policy_net.eval()
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
   
   def fillMemory(self):
      self.mode = "prefill"
      prefill = self.config.memory_size if self.config.prefill_size > self.config.memory_size else self.config.prefill_size
      added = 0
      state = self.env.reset()
      for _ in range(prefill):
         prev_state = state
         state, done, reward, action = self.play(state)
         next_state = state
         self.memory.append(prev_state, action, next_state, reward)
         if done:
            state = self.env.reset()



   def play(self, currentState):
      reward = 0
      done = False
      state = None

      action = self.sample_action(currentState)

      for _ in range(self.config.repeatAction):
         state, r, done, _ = self.env.step(action)
         reward += r

      return state, done, reward, action

