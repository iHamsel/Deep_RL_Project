from Agent import Agent
from AgentConfiguration import AgentConfiguration
from DoubleAgent import DoubleAgent
from DecayValue import LinearDecay

import gym
import wrappers
from DQN import SingleHead, DoubleHead
import pickle
import torch


gamename = "Breakout"

env = gym.make(f"{gamename}NoFrameskip-v4")
env = gym.wrappers.ResizeObservation(env, (84, 84))
env = gym.wrappers.GrayScaleObservation(env)
env = gym.wrappers.FrameStack(env, 4)
env = wrappers.NumpyWrapper(env, True)
env = wrappers.PyTorchWrapper(env)
env = wrappers.NoopResetEnv(env, 30)
env = wrappers.LiveLostContinueEnv(env)
env = wrappers.FireResetEnv(env)


singleConfig = AgentConfiguration(
   network=SingleHead,
   env=env,
   memory_size=int(5e3),
   prefill_size=int(2.5e3),
   learningSize=32,
   gamma=0.99,
   learnrate=2e-4,
   epsilon=LinearDecay(1, 0.1, 1e5),
   repeatAction=4,
   learnInterval=4,
   targetUpdateInterval=500,
   trainingLength=1e6,
   epochLength=5e3
   )


duelConfig = AgentConfiguration(
   network=DoubleHead,
   env=env,
   memory_size=int(5e3),
   prefill_size=int(2.5e3),
   learningSize=32,
   gamma=0.99,
   learnrate=2e-4,
   epsilon=LinearDecay(1, 0.1, 1e5),
   repeatAction=4,
   learnInterval=4,
   targetUpdateInterval=500,
   trainingLength=1e6,
   epochLength=5e3
   )


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
   agent.fillMemory()
   bestReward = 0
   for _ in agent.train():
      savename = f"{prefix}_training.pickle"
      with open(f"{savename}", "wb") as f:
         pickle.dump(agent.trainLog, f)

      print(f"Played: {agent.training_playSteps} | Learned: {agent.learned} | Target updated: {agent.updated} | Epsilon value: {agent.eps.getValue()}")
      
      agent.eval(10)
      savename = f"{prefix}_evaluation.pickle"
      with open(f"{savename}", "wb") as f:
         pickle.dump(agent.evaluationRewards, f)
      
      if bestReward < agent.evaluationRewards[-1]:
         torch.save(agent.policy_net.state_dict(), f"{prefix}_best.model")
         bestReward = agent.evaluationRewards[-1]
         print("Saved best model")
      # elif 0.8 * bestReward > agent.evaluationRewards[-1]:
      #    agent.policy_net.load_state_dict(torch.load(f"{prefix}_best.model"))
      #    agent.updateTargetNetwork()
      #    print("Reloaded best model")

   savename = f"{prefix}_training.pickle"
   with open(f"{savename}", "wb") as f:
      pickle.dump(agent.trainLog, f)

   savename = f"{prefix}_evaluation.pickle"
   with open(f"{savename}", "wb") as f:
      pickle.dump(agent.evaluationRewards, f)
   
   torch.save(agent.policy_net.state_dict(), f"{prefix}.model")

   del agent

