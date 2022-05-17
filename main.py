from gym.envs.atari.environment import AtariEnv

from Agent import Agent
from gaming import train

import gym
import wrappers
from DQN import SingleHead
import matplotlib.pyplot as plt

env = gym.make("ALE/Pong-v5") #AtariEnv(game="PongNoFrameskip-v4")
env = gym.wrappers.ResizeObservation(env, (84, 84))
env = gym.wrappers.GrayScaleObservation(env)
env = gym.wrappers.FrameStack(env, 4)
env = wrappers.NumpyWrapper(env, True)
env = wrappers.PyTorchWrapper(env)


agent = Agent(SingleHead, env, mem_len=5000, replay_batchsize=32, gamma=0.99)

rewards = train(env, agent, [x for x in range(env.action_space.n)], 200, "savepath")

f = plt.figure()
plt.plot(rewards)
plt.show()