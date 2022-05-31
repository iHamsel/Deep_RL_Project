from gym.envs.atari.environment import AtariEnv

from Agent import Agent
from DoubleAgent import DoubleAgent
from DecayValue import ExponentialDecay
from gaming import train

import gym
import wrappers
from DQN import SingleHead, DoubleHead
import matplotlib.pyplot as plt
import pickle

env = gym.make("PongNoFrameskip-v4")
env = gym.wrappers.ResizeObservation(env, (84, 84))
env = gym.wrappers.GrayScaleObservation(env)
# env = gym.wrappers.AtariPreprocessing(env)
env = gym.wrappers.FrameStack(env, 4)
env = wrappers.NumpyWrapper(env, True)
env = wrappers.PyTorchWrapper(env)

combinations = [
   {"agent": Agent(SingleHead, env, memory_length=5000, replay_batchsize=32, gamma=0.99), "savename": "SingleNormal"},
   {"agent": Agent(DoubleHead, env, memory_length=5000, replay_batchsize=32, gamma=0.99), "savename": "DuelNormal"},
   {"agent": DoubleAgent(SingleHead, env, memory_length=5000, replay_batchsize=32, gamma=0.99), "savename": "SingleDouble"},
   {"agent": DoubleAgent(DoubleHead, env, memory_length=5000, replay_batchsize=32, gamma=0.99), "savename": "DuelDouble"}
]



for combination in combinations:
   rewards = train(env, combination["agent"], [x for x in range(env.action_space.n)], 200, "savepath")
   savename = combination["savename"]
   with open(f"{savename}.pickle", "wb") as f:
      pickle.dump(rewards, f)

