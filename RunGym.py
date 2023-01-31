#! /usr/bin/env python3
import RogueEnvironment

# it should be in the registry now
import gym
from stable_baselines3 import PPO,DQN
print(gym.envs.registry)
env = gym.make('RogueLearning-v0')

print("training...")
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

print("running eval...")
steps = 0
observation = env.reset()
for _ in range(100000):
    action, states_ = model.predict(observation)
    observation, reward, terminated, info = env.step(action[0])
    steps += 1
    env.render()
    if terminated:
        observation = env.reset()
        print("died after: {} steps".format(steps))
        steps = 0
env.close()
