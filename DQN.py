import torch
import random
from collections import namedtuple, deque
from torch.nn.functional import smooth_l1_loss


#subclassing namedtuple
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward') )




class Memory():
   def __init__(self, len):
      assert len > 0, "len must be > 0"
      self.memory = deque([], maxlen=len)
   
   def __lshift__(self, other):
      assert type(other) == Transition, "Cannot append other types than Transition"
      self.memory.append(other)
      return self

   def sample(self, n):
      assert n > 0, "n must be > 0"
      return random.sample(self.memory, n)

   def __len__(self):
      return len(self.memory)




class Agent():
   def __init__(self, Model, inDim, n_actions, mem_len=5000, replay_batchsize=64, gamma=0.95, eps=1, eps_decay=0.9, eps_end=0.01):
      super().__init__()

      self.memory       = Memory(mem_len)
      self.batch_size   = replay_batchsize 
      self.n_actions    = n_actions
      self.eps          = eps
      self.eps_decay    = eps_decay
      self.eps_end      = eps_end
      self.gamma        = gamma
      
      self.device       = "cuda" if torch.cuda.is_available() else "cpu"
      self.policy_net   = Model(inDim, n_actions).to(self.device)
      self.target_net   = Model(inDim, n_actions).to(self.device)
      self.target_net.load_state_dict(self.policy_net.state_dict())
      self.target_net.eval()

      self.optimizer    = torch.optim.Adam(self.policy_net.parameters(), lr=0.001, eps=1e-7)


   def replay(self):
      #only replay if there enough samples in memory
      if len(self.memory) < self.batch_size:
         return

      #Generate a batch out of the transitions in the memory
      transitions = self.memory.sample(self.batch_size)
      batch = Transition(*zip(*transitions))

      non_final_mask          = torch.tensor(tuple(map(lambda s: s is not None,
                                       batch.next_state)), device=self.device, dtype=torch.bool)

      non_final_next_states   = torch.stack(tuple(s for s in batch.next_state if s is not None))
      states                  = torch.stack(batch.state)
      actions                 = torch.cat(batch.action)
      rewards                 = torch.cat(batch.reward)

      #calculate rewards based on the actual policy network
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

      if self.eps > self.eps_end:
            self.eps *= self.eps_decay


   def updateTargetNetwork(self):
      self.target_net.load_state_dict(self.policy_net.state_dict())      


   def sample_action(self, state):
      state = state.to(self.device)
      if random.random() > self.eps:
         with torch.no_grad():
            return self.policy_net(state).max(1)[1].view(1, 1)
      else:
         return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)


