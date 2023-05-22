import gymnasium as gym
import sumo_rl
import os
import sys
from ray.rllib.algorithms.sac.sac import SACConfig
from ray.tune.registry import register_env
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
                    out_csv_name='sac_results/result_lr0.00001batch2800clipFalse',
                    use_gui=True,
                    num_seconds=2800,
                    single_agent=True)


register_env("sumo", create_sumo)

config = (
        SACConfig()
        .environment("sumo")
        .rollouts(num_rollout_workers=1)
        .framework("torch")
        .training(model={"fcnet_hiddens": [256,128,128]}, clip_actions=False, 
            optimization_config={"actor_learning_rate": 0.00001, "critic_learning_rate": 0.00001, "entropy_learning_rate":0.00001}, train_batch_size=2800)
        .evaluation(evaluation_num_workers=1)
        )

algo = config.build(use_copy=False)

for i in range(100):
    print(pretty_print(algo.train()))

    if i % 5 == 0:
        checkpoint_dir = algo.save()
        print(f"Checkpoint saved in directory {checkpoint_dir}")

algo.evaluate()
