from collections import namedtuple, deque
import random

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