import RogueEnvironment

import gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

# Create the environment for training the model on. Also creating one for evaluating the model against the last best model
print(gym.envs.registry)
env = gym.make('RogueLearning-v0')
eval_env = gym.make('RogueLearning-v0')

# The callback called for comparing the current model against the last best model. Compares every 5000 steps
eval_callback = EvalCallback(Monitor(eval_env), best_model_save_path="./models/tensorlogs", log_path="./models/tensorlogs",
                             eval_freq=5000, deterministic=True, render=True)

print("training...")
model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="./models/tensorlogs/", exploration_fraction=0.8, exploration_initial_eps=0.8, exploration_final_eps=0.2,  learning_starts=1000,)
model.learn(total_timesteps=100000, tb_log_name="DQN", callback=eval_callback)
model.save("models/DQN_TensorRogueDeterministic_v17")
print("training complete.")
