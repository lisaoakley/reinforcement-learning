# Q learning tutorial on gym FrozenLake environment
# modified from: https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0

import gym
import numpy as np

env = gym.make('FrozenLake-v0')
a_space_size = env.action_space.n
o_space_size = env.observation_space.n
alpha = .8
gamma = .95
num_episodes = 2000
reward_list = []
# Q-table: 16 rows, 4 columns
Q = np.zeros([o_space_size, a_space_size])

for i in range(num_episodes):
    total_reward = 0
    observation = env.reset()
    for j in range(100):
        prev_observation = observation
        
        
        # greedily with noise choose from Q table. 
        # NOTE: noise here is to add "exploration", greedily choosing is "exploitation"
        action = np.argmax(Q[observation,:] + np.random.randn(1,a_space_size) * (1./(i+1)))
        
        # step
        observation, reward, done, _ = env.step(action)
        print(observation)

        # Bellman algorithm
        Q[prev_observation,action] = Q[prev_observation,action] + alpha*(reward + gamma*np.max(Q[observation,:]) - Q[prev_observation,action])
        
        # reward is 1 if win, 0 otherwise
        total_reward += reward
        if done:
            print('Episode {} completed with reward {}.'.format(i,total_reward))
            break
    reward_list.append(total_reward)
print('After {} episodes, average win rate is {}.'.format(num_episodes,sum(reward_list)/len(reward_list)))
print('Final Q-Table Values:\n{}'.format(Q))