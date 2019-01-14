# Example of running gym 'CartPole-v0' environment with a purely random strategy
# modified from http://gym.openai.com/docs/

# STEPS FOR INSTALLATION:
# `pip install gym`

import gym
env = gym.make('CartPole-v0') # generate instance of environment
for i_episode in range(10):
    observation = env.reset() # re-initialize environment
    print(observation)
    for t in range(100):
        env.render() # render game

        ### ACTION
        action = env.action_space.sample() # choose a random move within the action space of this environment
        # in 'CartPole-v0', the action space is Discrete(2), meaning actions are in {0,1}
        # 0 applies force from left, 1 applies force from right

        ### STEP
        observation, reward, done, info = env.step(action) # run the chosen action and collect results
        # results can be used to train learning algorithm

        ### RESULTS BREAKDOWN
        # observation (object): state of environment after action. In 'CartPole-v0', this is a 4-D array indicating state of cartpole
        print(observation)

        # reward (float): value of reward achieved by action. In 'CartPole-v0', this is +1 if pole remains upright after action
        print(reward)

        # done (bool): indicates the "episode" has completed (i.e. game over)
        print(done)

        # info (dict): additional info for debugging, not to be used by algorithms
        print(info)


        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break