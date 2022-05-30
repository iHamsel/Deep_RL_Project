from gym.envs.atari.environment import AtariEnv

from Agent import Agent
from DecayValue import ExponentialDecay
from gaming import train

import gym
import wrappers
from DQN import SingleHead, DoubleHead
import matplotlib.pyplot as plt

# aenv = AtariEnv(repeat_action_probability=0, full_action_space=true)

env = gym.make("PongNoFrameskip-v4") #AtariEnv(game="PongNoFrameskip-v4")
env = gym.wrappers.ResizeObservation(env, (84, 84))
env = gym.wrappers.GrayScaleObservation(env)
# env = gym.wrappers.AtariPreprocessing(env)
env = gym.wrappers.FrameStack(env, 4)
env = wrappers.NumpyWrapper(env, True)
env = wrappers.PyTorchWrapper(env)

agent = Agent(SingleHead, env, memory_length=5000, replay_batchsize=32, gamma=0.99)

rewards = train(env, agent, [x for x in range(env.action_space.n)], 200, "savepath")