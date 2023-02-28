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
    save_freq=10000,
    save_path="./models/checkpointlogs",
    name_prefix="rl_model",
    save_replay_buffer=False,
)

print("training...")
#model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="./models/tensorlogs/", exploration_fraction=10, exploration_initial_eps=0.2, learning_starts=100)
#model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./models/tensorlogs/", learning_rate=0.8, gamma=0.5, n_steps=1000, n_epochs=20, ent_coef=100, vf_coef=100)
#model = A2C("MlpPolicy", env, verbose=1, tensorboard_log="./models/tensorlogs/")
#model.learn(total_timesteps=50000, tb_log_name="A2C", callback=checkpoint_callback)
#model.save("models/A2C_TensorRogueDeterministic_v2")

model = A2C.load("models/A2C_TensorRogueDeterministic_v2.zip")
print("running eval...")
steps = 0

for _ in range(10):
    dead = False
    observation = env.reset()
    while not dead:
        action, states_ = model.predict(observation, deterministic=True)
        #print(observation)
        observation, reward, terminated, info = env.step(action)
        print(reward)
        steps += 1
        env.render()
        model.set_env(env)
        if terminated:
            observation = env.reset()
            dead = True
            print("died after: {} steps".format(steps))
            steps = 0
env.close()

