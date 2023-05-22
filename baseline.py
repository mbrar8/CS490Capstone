import gymnasium as gym
import sumo_rl
import os
import sys
import ray
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.tune.trainable.trainable import Trainable
from ray.tune.registry import register_env
from ray import air, tune
from ray.tune.logger import pretty_print

#env = gym.make('sumo-rl-v0',
#                net_file='sumo_inter2/lankershim_intersect2.net.xml',
#                route_file='sumo_inter2/lankershim_intersect2.rou.xml',
#                out_csv_name='inter2.csv',
#                use_gui=True,
#                num_seconds=1800)

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

def create_sumo(env_config={}):
    return sumo_rl.environment.env.SumoEnvironment(net_file='lankershim_intersect2.net.xml',
                    route_file='lankershim_intersect2_mod.rou.xml',
                    out_csv_name='result_baseline_constant_generalized',
                    use_gui=True,
                    num_seconds=2800,
                    single_agent=True)


env = create_sumo()

print(env.ts_ids)

for episode in range(1,2):
    obs, info = env.reset()
    done = False
    while not done:
        next_obs, reward, terminated, truncated, info = env.step(None)
        done = terminated or truncated

    env.save_csv('result_baseline_constant_generalized', episode)
