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
import torch

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
   memory_size=int(5e3),
   learningSize=32,
   gamma=0.99,
   learnrate=1e-4,
   epsilon=LinearDecay(1, 0.001, 5e4),
   repeatAction=4,
   learnInterval=4,
   targetUpdateInterval=1000)

combinations = [
   {"agent": Agent(config), "name": "SingleDQN"}
]


for combination in combinations:
   name = combination["name"]
   agent: Agent = combination["agent"]
   agent.fillMemory()
   for i in range(150):
      agent.train(10)
      savename = f"{name}_training.pickle"
      with open(f"{savename}", "wb") as f:
         pickle.dump(agent.trainingRewards, f)
      with open(f"Losses.pickle", "wb") as f:
         pickle.dump(agent.losses, f)

      print(f"Played: {agent.training_playSteps} | Learned: {agent.learned} | Target updated: {agent.updated}")
      
      agent.eval(1)
      savename = f"{name}_evaluation.pickle"
      with open(f"{savename}", "wb") as f:
         pickle.dump(agent.evaluationRewards, f)
      if i % 5 == 4:
         torch.save(agent.policy_net.state_dict(), f"{name}_{(i+1)*10}.model")

   del agent

