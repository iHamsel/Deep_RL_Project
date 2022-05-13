from gym.envs.atari.environment import AtariEnv
from paramiko import Agent

from DQN_Atari import DQN_AtariModel
from DQN import Agent
from gaming import train



env = AtariEnv(game="Atlantis")
env.ale.setBool("sound", False)


agent = Agent(DQN_AtariModel, 4, len(env._action_set), eps_decay=0.9999, eps_end=0.01, mem_len=10000)

train(env, agent, [x for x in range(len(env._action_set))], 100, "savepath")
