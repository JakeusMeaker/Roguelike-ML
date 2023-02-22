#! /usr/bin/env python3
import RogueEnvironment

# it should be in the registry now
import pandas as pd
import gym
from stable_baselines3 import PPO,DQN
print(gym.envs.registry)
env = gym.make('RogueLearning-v0')

#print("training...")
#model = DQN("MlpPolicy", env, verbose=1)
#model.learn(total_timesteps=100000)
#model.save("models/DQN_rogue_v3")

model = DQN.load("models/DQN_rogue_v3.zip")
print("running eval...")
steps = 0

for _ in range(10):
    dead = False
    observation = env.reset()
    while not dead:
        action, states_ = model.predict(observation)
        observation, reward, terminated, info = env.step(action)
        steps += 1
        #env.render()
        if terminated:
            observation = env.reset()
            dead = True
            print("died after: {} steps".format(steps))
            steps = 0
#env.close()

# observations_df = pd.DataFrame(model.)
