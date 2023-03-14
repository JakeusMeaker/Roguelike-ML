# Loads and runs the most recently trained model. The file name is updated manually once training is complete
import RogueEnvironment

import gym
from stable_baselines3 import DQN
import numpy as np

env = gym.make('RogueLearning-v0')
model = DQN.load("models/DQN_TensorRogueDeterministic_v16", env)

print("running latest model eval...")
steps = 0
for _ in range(10):
    dead = False
    observation = env.reset()
    states_ = None
    episode_starts = np.ones(1, dtype=bool)
    while not dead:
        action, states_ = model.predict(observation, state=states_, episode_start=episode_starts, deterministic=True)
        # print(observation)
        observation, reward, terminated, info = env.step(action)
        print(reward)
        steps += 1
        episode_starts[0] = terminated
        env.render()
        if terminated:
            observation = env.reset()
            dead = True
            print("died after: {} steps".format(steps))
            steps = 0
env.close()
