from itertools import count



def train(env, agent, actions, nEpisodes, save_path):
   rewards = []

   counter = 0

   for i_episode in range(nEpisodes):
      # Initialize the environment and state
      state = env.reset()

      rewards.append(0)

      for t in count():
         # Select and perform an action
         action = agent.sample_action(state)
         prev_state = state
         reward = 0
         done = False
         for _ in range(4):
            state, r, done, _ = env.step(actions[action.item()])
            reward += r

         rewards[i_episode] += reward
         
         # Observe new state
         next_state = None if done else state

         # Store the transition in memory of DQN
         agent.memory.append(prev_state, action, next_state, reward)
         if t > 0 and t % 4 == 0:
            agent.learn()
         
         
         if counter > 0 and counter % 4e4 == 0:
            print("Target updated")
            agent.updateTargetNetwork()
         
         counter += 1

         if done == True:
            break

      print(f"\n\n{i_episode}. Episode finished: {rewards[i_episode]}")
      print(agent.eps.getValue())
   
   env.close()
   return rewards

