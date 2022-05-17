from gym.envs.atari.environment import AtariEnv
from collections import deque
from itertools import count
from collections import deque
import numpy
from sqlalchemy import true
import torch
import torchvision.transforms.functional as F
from Agent import Transition
from IPython import display
from PIL import Image



def train(env, agent, actions, nEpisodes, save_path):
   rewards = []

   for i_episode in range(nEpisodes):
      # Initialize the environment and state
      state = env.reset()
      #state = preprocessState(state)

      rewards.append(0)

      for t in count():
         # Select and perform an action
         action = agent.sample_action(state)
         prev_state = state
         reward = 0
         done = False
         for _ in range(1):
            state, r, done, _ = env.step(actions[action.item()])
            reward += r

         rewards[i_episode] += reward
         
         # Observe new state
         next_state = state
         if done:
            next_state = None

         # Store the transition in memory of DQN
         agent.memory << Transition(prev_state, action, next_state, reward)
         agent.replay()
         if done == True:
            break
         
         if t > 0 and t % 100 == 0:
            agent.updateTargetNetwork()

      print(f"\n\n{i_episode}. Episode finished: {rewards[i_episode]}\n\n")
      agent.updateTargetNetwork()
      print(agent.eps.getValue())

      if i_episode % 25 == 0:
         torch.save(agent.policy_net.state_dict(), f"{save_path}/Episode_{i_episode}.model")
   
   env.close()
   return rewards

