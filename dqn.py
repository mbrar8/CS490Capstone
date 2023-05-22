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
                    route_file='lankershim_intersect2.rou.xml',
                    out_csv_name='dqn_results/result_lr0.0001vmax1000atom1',
                    use_gui=True,
                    num_seconds=2800,
                    single_agent=True)


register_env("sumo", create_sumo)

config = (
        DQNConfig()
        .environment("sumo")
        .rollouts(num_rollout_workers=1)
        .framework("torch")
        .training(model={"fcnet_hiddens":[256,128,128]}, lr=0.0001, noisy=True,v_max=1000, num_atoms=1)
        .evaluation(evaluation_num_workers=1)
        )

algo = config.build(use_copy=False)

for i in range(100):
    print(pretty_print(algo.train()))

    if i % 5 == 0:
        checkpoint_dir = algo.save()
        print(f"Checkpoint saved in directory {checkpoint_dir}")

algo.evaluate()
