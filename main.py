from gym.envs.atari.environment import AtariEnv
from paramiko import Agent

from DQN_Atari import DQN_AtariModel
from DQN import Agent
from gaming import train



env = AtariEnv(game="Atlantis", render_mode="human")


agent = Agent(DQN_AtariModel, 3, len(env._action_set))

train(env, agent, [x for x in range(len(env._action_set))], 10, "savepath")

print([x for x in range(len(env._action_set))])

   