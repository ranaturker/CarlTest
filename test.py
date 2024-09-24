from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from carl.envs import CARLPendulum
import os

lengths = [0.6, 1.6, 2.6, 3.6, 4.6]

default = CARLPendulum().get_default_context()

contexts = {}
for i, l in enumerate(lengths):
    contexts[i] = default.copy()
    contexts[i]["l"] = l

eval_env = CARLPendulum(contexts=contexts, hide_context=False, dict_observation_space=False,
                        state_context_features="changing_context_features")
eval_env = FlattenObservation(eval_env)

def train_model(algo, algo_name, model_dir, log_dir, eval_env, contexts):
    print(f"Training {algo_name}")

    algo_best_model_dir = f"best_model/{algo_name}"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    env = CARLPendulum(contexts=contexts, hide_context=False, dict_observation_space=False,
                       state_context_features="changing_context_features")
    env = FlattenObservation(env)

    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path=algo_best_model_dir,
                                 log_path=log_dir,
                                 eval_freq=500,
                                 n_eval_episodes=len(lengths),
                                 deterministic=True,
                                 render=False)
    model = algo("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
    model.learn(total_timesteps=int(200000), callback=eval_callback)

    model.save(f"{model_dir}/best_model_{algo_name}.zip")

    mean_reward, std_dev = evaluate_policy(model, model.get_env(), n_eval_episodes=len(lengths))
    print(f"{algo_name} - Mean Dev: {mean_reward}, Std Dev: {std_dev}")

    return model

ppo_model_dir = "models"
ppo_log_dir = "logs"
ppo_model = train_model(PPO, "ppo", ppo_model_dir, ppo_log_dir, eval_env, contexts)

sac_model_dir = "models"
sac_log_dir = "logs"
sac_model = train_model(SAC, "sac", sac_model_dir, sac_log_dir, eval_env, contexts)

def test_model(l, model, algo_name):
    context = default.copy()
    context["l"] = l
    test_context = {0: context}

    test_env = CARLPendulum(contexts=test_context, hide_context=False, dict_observation_space=False,
                            state_context_features="changing_context_features")
    test_env = FlattenObservation(test_env)

    mean_reward, std_reward = evaluate_policy(model, test_env, deterministic=True)

    print(f"Testing {algo_name} on length={l}: Mean Reward: {mean_reward}, Std Reward: {std_reward}")

    return mean_reward, std_reward

test_model(0.8, ppo_model, "PPO")

test_model(0.8, sac_model, "SAC")
