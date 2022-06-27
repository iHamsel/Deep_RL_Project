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


gamename = "Atlantis"

def rewardTransform(r):
   if r > 0:
      return 1.0
   elif r < 0:
      return -1.0
   else:
      return 0.0

env = gym.make(f"{gamename}NoFrameskip-v4")
env = gym.wrappers.ResizeObservation(env, (84, 84))
env = gym.wrappers.GrayScaleObservation(env)
# env = gym.wrappers.TransformReward(env, rewardTransform)
# env = gym.wrappers.AtariPreprocessing(env)
env = gym.wrappers.FrameStack(env, 4)
env = wrappers.NumpyWrapper(env, True)
env = wrappers.PyTorchWrapper(env)


singleConfig = AgentConfiguration(
   network=SingleHead,
   env=env,
   memory_size=int(5e3),
   prefill_size=int(2.5e3),
   learningSize=32,
   gamma=0.99,
   learnrate=1e-4,
   epsilon=LinearDecay(1, 0.001, 5e4),
   repeatAction=4,
   learnInterval=4,
   targetUpdateInterval=1000)


duelConfig = AgentConfiguration(
   network=DoubleHead,
   env=env,
   memory_size=int(5e3),
   prefill_size=int(2.5e3),
   learningSize=32,
   gamma=0.99,
   learnrate=1e-4,
   epsilon=LinearDecay(1, 0.001, 5e4),
   repeatAction=4,
   learnInterval=4,
   targetUpdateInterval=1000)


combinations = [
   {"agent": Agent(singleConfig), "name": "SingleDQN"},
   {"agent": DoubleAgent(singleConfig), "name": "SingleDDQN"},
   {"agent": Agent(duelConfig), "name": "DuelDQN"},
   {"agent": DoubleAgent(duelConfig), "name": "DuelDDQN"},
]


for combination in combinations:
   agentname = combination["name"]
   agent: Agent = combination["agent"]
   prefix = f"{gamename}_{agentname}"
   for _ in agent.train():
      savename = f"{prefix}_training.pickle"
      with open(f"{savename}", "wb") as f:
         pickle.dump(agent.log, f)

      print(f"Played: {agent.training_playSteps} | Learned: {agent.learned} | Target updated: {agent.updated}")
      
      agent.eval(1)
      savename = f"{prefix}_evaluation.pickle"
      with open(f"{savename}", "wb") as f:
         pickle.dump(agent.evaluationRewards, f)

   savename = f"{prefix}_training.pickle"
   with open(f"{savename}", "wb") as f:
      pickle.dump(agent.log, f)

   savename = f"{prefix}_evaluation.pickle"
   with open(f"{savename}", "wb") as f:
      pickle.dump(agent.evaluationRewards, f)
   
   torch.save(agent.policy_net.state_dict(), f"{prefix}.model")

   del agent

