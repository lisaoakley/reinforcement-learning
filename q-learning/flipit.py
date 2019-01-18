# Q learning tutorial on gym FrozenLake environment
# modified from: https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0

import gym
import gym_flipit
import numpy as np

def q(debug):
    env = gym.make('Flipit-v0')
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
            if debug:
                print(observation)

            # Bellman algorithm
            Q[prev_observation,action] = Q[prev_observation,action] + alpha*(reward + gamma*np.max(Q[observation,:]) - Q[prev_observation,action])
            
            # reward is 1 if win, 0 otherwise
            total_reward += reward
            if done:
                if debug:
                    print('Episode {} completed with reward {}.'.format(i,total_reward))
                break
        reward_list.append(total_reward)
    reward_list = reward_list
    win_average = sum(reward_list)/len(reward_list)
    print('After {} episodes, average reward is {}.'.format(num_episodes,win_average))
    if debug:
        print('Final Q-Table Values:\n{}'.format(Q))
    return reward_list

a = 0
a2 = 0
for i in range(20):
    res = q(False)
    a += sum(res)/len(res)
    a2 += sum(res[500:])/len(res[500:])
    

print(a/20)
print(a2/20)