from numbers import Number
import torch
import random
import numpy as np
from torch.nn.functional import smooth_l1_loss
from DQN.DQN import DQN

from ReplayMemory import Memory, Transition
from DQN import DQN
from DecayValue import *








class Agent():
   def __init__(self, Model: DQN, env, mem_len=5000, replay_batchsize=64, gamma=0.95, eps: DecayValue=ExponentialDecay(0.95, 0.01, 1e5)):
      super().__init__()

      assert issubclass(Model, DQN), "Parameter 'Model' is not a subclass of 'DQN'"

      self.memory       = Memory(mem_len)
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

   def replay(self):
      #only replay if there enough samples in memory
      if len(self.memory) < self.batch_size:
         return

      #Generate a batch out of the transitions in the memory
      transitions = self.memory.sample(self.batch_size)
      batch = Transition(*zip(*transitions))

      non_final_mask          = torch.tensor(tuple(map(lambda s: s is not None,
                                       batch.next_state)), device=self.device, dtype=torch.bool)

      non_final_next_states   = torch.stack(tuple(s for s in batch.next_state if s is not None)).to(self.device)
      states                  = torch.stack(batch.state).to(self.device)
      actions                 = torch.cat([torch.tensor([a]) for a in batch.action]).reshape((self.batch_size, 1)).to(self.device)
      rewards                 = torch.cat([torch.tensor([r]) for r in batch.reward]).to(self.device)



      #Calculate 
      policy_rewards = self.policy_net(states).gather(1, actions)

      #calculate expected rewards based on the target network
      next_rewards = torch.zeros(len(states), device=self.device)
      next_rewards[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
      expected_rewards = (self.gamma * next_rewards) + rewards

      #Updating parameters of the policy_net
      loss = smooth_l1_loss(policy_rewards, expected_rewards.unsqueeze(1))
      self.optimizer.zero_grad()
      loss.backward()
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
         


