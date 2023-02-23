#! /usr/bin/env python3
import RogueEnvironment

# it should be in the registry now
import pandas as pd
import gym
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.callbacks import CheckpointCallback
print(gym.envs.registry)
env = gym.make('RogueLearning-v0')

# Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(
    save_freq=1000,
    save_path="./checkpointlogs",
    name_prefix="rl_model",
    save_replay_buffer=True,
)

print("training...")
model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="./models/tensorlogs/", exploration_fraction=1, exploration_initial_eps=0.8, learning_starts=100)
model.learn(total_timesteps=50000, tb_log_name="DQN", callback=checkpoint_callback)
model.save("models/DQN_TensorRogueDeterministic_v2")

#model = DQN.load("models/DQN_TensorRogue_v4")
print("running eval...")
steps = 0

for _ in range(10):
    dead = False
    observation = env.reset()
    while not dead:
        action, states_ = model.predict(observation, deterministic=True)
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

