#! /usr/bin/env python3
import RogueEnvironment

# it should be in the registry now
import pandas as pd
import gym
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, ProgressBarCallback
from stable_baselines3.common.monitor import Monitor
import numpy as np

print(gym.envs.registry)
env = gym.make('RogueLearning-v0')
eval_env = gym.make('RogueLearning-v0')

eval_callback = EvalCallback(Monitor(eval_env), best_model_save_path="./models/tensorlogs", log_path="./models/tensorlogs",
                             eval_freq=5000, deterministic=True, render=True)

# Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path="./models/checkpointlogs",
    name_prefix="rl_model",
    save_replay_buffer=False,
)

print("training...")
model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="./models/tensorlogs/", exploration_fraction=0.8, exploration_initial_eps=0.8, exploration_final_eps=0.2,  learning_starts=1000,)
# model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./models/tensorlogs/", learning_rate=0.8, gamma=0.5, n_steps=1000, n_epochs=20, ent_coef=100, vf_coef=100)
# model = A2C("MlpPolicy", env, verbose=1, tensorboard_log="./models/tensorlogs/")
model.learn(total_timesteps=100000, tb_log_name="DQN", callback=eval_callback, progress_bar=True)
model.save("models/DQN_TensorRogueDeterministic_v16")

#model = DQN.load("models/DQN_TensorRogueDeterministic_v13.zip", env)
#model = DQN.load("models/tensorlogs/best_model.zip", env)
print("running eval...")
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
