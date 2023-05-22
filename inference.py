import gymnasium as gym
import sumo_rl
import os
import sys
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.utils.framework import try_import_tf
import tensorflow as tf
from ray.tune.registry import register_env
from ray import tune

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare env var SUMO_HOME")


tf1, tf, tfv = try_import_tf()


print("Loaded algo")

def create_sumo(env_config={}):
    return sumo_rl.environment.env.SumoEnvironment(net_file='lankershim_intersect2.net.xml',
                    route_file='lankershim_intersect2_mod.rou.xml',
                    out_csv_name='sac_gen',
                    use_gui=True,
                    num_seconds=2800,
                    single_agent=True)

print("Registering env")

env = create_sumo()
register_env("sumo", create_sumo)

algo = Algorithm.from_checkpoint('C:\\Users\mahee\\ray_results\\SAC_sumo_2023-04-24_20-54-44vqlomqik\\checkpoint_000071')
algo.evaluate()


#pol = algo.get_policy()

#obs, info = env.reset()
#done = False
#while not done:
#    next_obs, reward, terminated, truncated, info = env.step(pol.compute_single_action(obs))
#    obs = next_obs
#    done = terminated or truncated

