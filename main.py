from gym.envs.atari.environment import AtariEnv

from Agent import Agent
from AgentConfiguration import AgentConfiguration
from DoubleAgent import DoubleAgent
from DecayValue import ExponentialDecay, LinearDecay
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


config = AgentConfiguration(
   network=SingleHead,
   env=env,
   memory_size=int(1e6),
   learningSize=32,
   gamma=0.99,
   learnrate=0.25e-4,
   epsilon=LinearDecay(1, 0.1, 2e4),
   repeatAction=4,
   learnInterval=4,
   targetUpdateInterval=1000)

combinations = [
   {"agent": Agent(config), "name": "SingleDQN"}
   # {"agent": Agent(SingleHead, env, memory_length=int(1e6), replay_batchsize=32, gamma=0.99, eps=LinearDecay(1, 0.1, 2e4)), "savename": "SingleNormal"},
   # {"agent": Agent(DoubleHead, env, memory_length=int(1e6), replay_batchsize=32, gamma=0.99, eps=LinearDecay(1, 0.1, 2e4)), "savename": "DuelNormal"}
]


for combination in combinations:
   name = combination["name"]
   agent: Agent = combination["agent"]
   for _ in range(50):
      agent.train(10)
      savename = f"{name}_training.pickle"
      with open(f"{savename}.pickle", "wb") as f:
         pickle.dump(agent.trainingRewards, f)
      
      agent.eval(10)
      savename = f"{name}_evaluation.pickle"
      with open(f"{savename}.pickle", "wb") as f:
         pickle.dump(agent.evaluationRewards, f)

      agent.policy_net
   del agent

