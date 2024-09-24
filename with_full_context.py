import os
import numpy as np
import random
import torch
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from carl.envs import CARLPendulum
from stable_baselines3.common.evaluation import evaluate_policy

LENGTH_RANGE = [0.5, 2]
TRAIN_LENGTHS = np.linspace(LENGTH_RANGE[0], LENGTH_RANGE[1], 10)
TEST_LENGTHS_INTR = np.linspace(LENGTH_RANGE[0], LENGTH_RANGE[1], 20)
TEST_LENGTHS_EXTR = np.linspace(2.1, 3, 10)

default = CARLPendulum().get_default_context()

contexts = {}
for i, m in enumerate(TRAIN_LENGTHS):
    contexts[i] = default.copy()
    contexts[i]["m"] = m

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def train_model(algo, algo_name, log_dir, contexts=contexts, hide_context=False, seed=None):
    print(f"Training {algo_name} {'with' if hide_context else 'without'} context using seed {seed}")

    seed_everything(seed)

    algo_best_model_dir = os.path.join(log_dir, "best_model")
    os.makedirs(algo_best_model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    env = CARLPendulum(contexts=contexts, hide_context=hide_context, dict_observation_space=False)
    env = FlattenObservation(env)

    eval_env = CARLPendulum(contexts=contexts, hide_context=hide_context, dict_observation_space=False)
    eval_env = FlattenObservation(eval_env)

    eval_callback = EvalCallback(eval_env,
                                 log_path=log_dir,
                                 best_model_save_path=algo_best_model_dir,
                                 eval_freq=2000,
                                 n_eval_episodes=len(TRAIN_LENGTHS),
                                 deterministic=True,
                                 render=False)

    model = algo("MlpPolicy", env, gradient_steps=5, gamma = 0.9, ent_coef= 0.2, batch_size=128, learning_rate=0.001,
                 tau=0.001, learning_starts=5000, train_freq=4,
                 verbose=1, tensorboard_log=f"{log_dir}/{algo_name}_seed_{seed}")

    total_timesteps = 100000

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    best_model_path = os.path.join(algo_best_model_dir, "best_model.zip")
    best_model = algo.load(best_model_path)

    return best_model


def test_model(log_dir, algo, test_lengths):
    algo_best_model_dir = os.path.join(log_dir, "best_model")
    best_model_path = os.path.join(algo_best_model_dir, "best_model.zip")

    if os.path.exists(best_model_path):
        model = algo.load(best_model_path)
    else:
        raise FileNotFoundError(f"No best model found at {best_model_path}")

    test_context = {}
    for i, m in enumerate(test_lengths):
        test_context[i] = default.copy()
        test_context[i]["m"] = m

    test_env = CARLPendulum(contexts=test_context, hide_context=False, dict_observation_space=False)
    test_env = FlattenObservation(test_env)

    mean_reward, std_reward = evaluate_policy(model, test_env, deterministic=True, n_eval_episodes=len(test_lengths))

    return mean_reward, std_reward

sac_log_dir = "logs"
seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
all_mean_rewards_intr = []
all_std_rewards_intr = []
all_mean_rewards_extr = []
all_std_rewards_extr= []

for i, seed in enumerate(seeds, start=0):
    log_dir = f"{sac_log_dir}/sac_with_full_context_seed_{i}"
    train_model(SAC, "sac_with_full_context", log_dir, contexts, seed=seed)

    mean_reward, std_reward= test_model(log_dir, SAC, TEST_LENGTHS_INTR)
    all_mean_rewards_intr.append(mean_reward)
    all_std_rewards_intr.append(std_reward)

    mean_reward, std_reward= test_model(log_dir, SAC, TEST_LENGTHS_EXTR)
    all_mean_rewards_extr.append(mean_reward)
    all_std_rewards_extr.append(std_reward)

print("Results across all seeds (INTERPOLATION):")
for i, (mean_reward, std_reward) in enumerate(zip(all_mean_rewards_intr, all_std_rewards_intr)):
    print(f"  Seed {seeds[i]}: Mean Reward = {mean_reward}, Std Reward = {std_reward}")

mean_reward_across_seeds_intr=np.mean(all_mean_rewards_intr)
std_reward_across_seeds_intr=np.std(all_mean_rewards_intr)
median_reward_across_seeds_intr=np.median(all_mean_rewards_intr)

print(f"Mean Reward Across Seeds = {mean_reward_across_seeds_intr}, Std Reward Across Seeds = {std_reward_across_seeds_intr},"
      f" Median Reward Across Seeds= {median_reward_across_seeds_intr}")

print("Results across all seeds (EXTRAPOLATION):")
for i, (mean_reward, std_reward) in enumerate(zip(all_mean_rewards_extr, all_std_rewards_extr)):
    print(f"  Seed {seeds[i]}: Mean Reward = {mean_reward}, Std Reward = {std_reward}")

mean_reward_across_seeds_extr=np.mean(all_mean_rewards_extr)
std_reward_across_seeds_extr=np.std(all_mean_rewards_extr)
median_reward_across_seeds_extr=np.median(all_mean_rewards_extr)

print(f"Mean Reward Across Seeds = {mean_reward_across_seeds_extr}, Std Reward Across Seeds = {std_reward_across_seeds_extr},"
      f" Median Reward Across Seeds= {median_reward_across_seeds_extr}")