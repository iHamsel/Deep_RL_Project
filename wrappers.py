import gym
from gym import spaces
import numpy as np
import torch

class NumpyWrapper(gym.ObservationWrapper):
   def __init__(self, env, stacked):
      super(NumpyWrapper, self).__init__(env)
      
      self.stacked = stacked
      observation_space = self.observation_space
      
      if stacked == True:
         assert len(observation_space.shape) <= 4, "Not implemented for bigger shapes"
         self.k = observation_space.shape[0]
         if len(observation_space.shape) == 4:
            self.k *= observation_space.shape[-1]


         self.w = observation_space.shape[1]
         self.h = observation_space.shape[2]

      else:
         assert len(observation_space.shape) <= 3, "Not implemented for bigger shapes"
         self.k = 1
         if len(observation_space.shape) == 3:
            self.k *= observation_space.shape[-1]

         self.w = observation_space.shape[0]
         self.h = observation_space.shape[1]

      self.observation_space = spaces.Box(low=0, high=255, shape=(self.w, self.h, self.k))

   def observation(self, observation):
      observation = np.array(observation)
      if self.stacked == True:
         observation = np.moveaxis(observation, 0, -1)
      observation = np.reshape(observation, (self.w, self.h, self.k))
      return observation




class PyTorchWrapper(gym.ObservationWrapper):
   def __init__(self, env):
      super(PyTorchWrapper, self).__init__(env)
      observation_space = self.observation_space
      self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(observation_space.shape[-1], observation_space.shape[0], observation_space.shape[1]))

   def observation(self, observation):
      observation = np.moveaxis(observation, -1, 0)
      observation = observation / 255.0
      observation = torch.from_numpy(observation).float()
      return observation
