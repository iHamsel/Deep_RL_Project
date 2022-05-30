from collections import namedtuple, deque
import random

import torch

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


class ReplayMemory():
   def __init__(self, length):
      assert length > 0, "Parameter 'length' must be > 0"
      self.memory = deque([], maxlen=length)

   def append(self, state: torch.Tensor, action, next_state: torch.Tensor, reward):
      device = "cuda" if torch.cuda.is_available() else "cpu"
      state = state.to(device)
      if next_state is not None:
         next_state = next_state.to(device)
      reward = torch.tensor(reward, device=device)
      self.memory.append((state, action, next_state, reward))
      pass

   def sample(self, n):
      assert n > 0, "n must be > 0"
      return random.sample(self.memory, n)

   def __len__(self):
      return len(self.memory)