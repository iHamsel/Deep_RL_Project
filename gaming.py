from gym.envs.atari.environment import AtariEnv
from collections import deque
from itertools import count
from collections import deque
from sqlalchemy import true
import torch
import torchvision.transforms.functional as F
from DQN import Transition
from IPython import display
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"


def generateFrame(deque):
   return torch.vstack(list(deque))


def preprocessState(x):
   x = Image.fromarray(x)
   x = F.resize(x, (110, 84))
   x = F.crop(x, 0, 13, 84, 84)
   x = F.to_grayscale(x)
   x = F.to_tensor(x)
   return x.to(device)



def train(env, agent, actions, nEpisodes, save_path):
   rewards = []
   device = "cuda" if torch.cuda.is_available() else "cpu"


   for i_episode in range(nEpisodes):
      # Initialize the environment and state
      state = preprocessState(env.reset())
      frame_stack = deque([state] * 4, maxlen=4)

      rewards.append(0)

      badCounter = 0       #counter for counting bad rewards in a row
      for t in count():
         # Select and perform an action
         sequence = generateFrame(frame_stack)
         action = agent.sample_action(sequence.reshape((1, *sequence.shape)))

         reward = 0

         for _ in range(2):
            s, r, done, _ = env.step(actions[action.item()])
            state = preprocessState(s)
            frame_stack.append(state)
            reward += r

         rewards[i_episode] += reward
         
         reward = torch.tensor([reward], device=device)

         # Observe new state
         next_sequence = generateFrame(frame_stack)
         if done:
            next_sequence = None

         # Store the transition in memory of DQN
         agent.memory << Transition(sequence, action.to(device), next_sequence, reward.to(device))
         agent.replay()
         if done == True or badCounter >= 25:
            break
         
         if t > 0 and t % 100 == 0:
            agent.updateTargetNetwork()


      display.clear_output(wait=True)
      print(f"\n\n{i_episode}. Episode finished: {rewards[i_episode]}\n\n")
      agent.updateTargetNetwork()
      print(agent.eps)

      if i_episode % 25 == 0:
         torch.save(agent.policy_net.state_dict(), f"{save_path}/Episode_{i_episode}.model")
   
   env.close()
   return rewards




def play(agent, actions, nEpisodes):
   env = AtariEnv(render_mode="human")
   rewards = []


   for i_episode in range(nEpisodes):
      # Initialize the environment and state
      state = preprocessState(env.reset())

      frame_stack = deque([state] * 3, maxlen=3)

      badCounter = 0       #counter for counting bad rewards in a row
      for t in count():
         # Select and perform an action
         sequence = generateFrame(frame_stack)
         action = agent.sample_action(sequence.reshape((1, *sequence.shape)))

         reward = 0

         for _ in range(2):
            _, r, done, _ = env.step(actions[action.item()])
            env.render()
            reward += r

         # Observe new state
         state = preprocessState(env.state)
         frame_stack.append(state)

         badCounter = badCounter + 1 if t > 100 and reward < 0 else 0
         if done or int(env.score_label.text) < 0 or badCounter >= 25:
            break
      
      rewards.append(int(env.score_label.text))

      display.clear_output(wait=True)
      print(f"\n\n{i_episode}. Episode finished: {rewards[i_episode]}\n\n")
      
   env.close()
   return rewards

