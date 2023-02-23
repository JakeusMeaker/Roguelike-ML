#! /usr/bin/env python3
import RogueEnvironment

# it should be in the registry now
import pandas as pd
import gym
from stable_baselines3 import PPO, DQN, A2C
print(gym.envs.registry)
env = gym.make('RogueLearning-v0')

#print("training...")
#model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="./models/tensorlogs/")
#model.learn(total_timesteps=2000000, tb_log_name="DQN")
#model.save("models/DQN_TensorRogue_v4")

model = DQN.load("models/DQN_TensorRogue_v4")
print("running eval...")
steps = 0

for _ in range(10):
    dead = False
    observation = env.reset()
    while not dead:
        action, states_ = model.predict(observation)
        #print(observation)
        observation, reward, terminated, info = env.step(action)
        steps += 1
        env.render()
        if terminated:
            observation = env.reset()
            dead = True
            print("died after: {} steps".format(steps))
            steps = 0
env.close()

