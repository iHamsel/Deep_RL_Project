from gym.envs.atari.environment import AtariEnv



env = AtariEnv(render_mode="human")

env.reset()
for _ in range(1000):
   env.step(env.action_space.sample()) # take a random action
env.close()

   