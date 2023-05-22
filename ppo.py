import gymnasium as gym
import sumo_rl
import os
import sys
from ray.rllib.algorithms.ppo import PPOConfig
from ray import air, tune
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
                    out_csv_name='ppo_results/result_lr0.00001,batch256,clip10000',
                    use_gui=True,
                    num_seconds=2800,
                    single_agent=True)


register_env("sumo", create_sumo)

config = (
        PPOConfig()
        .environment("sumo")
        .rollouts(num_rollout_workers=1)
        .framework("torch")
        .training(model={"fcnet_hiddens": [256,128,128]}, lr=0.00001, train_batch_size=256, vf_clip_param=10000)
        .evaluation(evaluation_num_workers=1)
        )

algo = config.build(use_copy=False)

#param_space = {"lr": tune.grid_search([0.0001,0.00001,0.000001]),
#            "train_batch_size": tune.grid_search([256, 4000, 10000]),
#            "vf_clip_param": tune.grid_search([10,100,10000])
#            }

#print('Starting training')

#tuner = tune.Tuner(algo, param_space=param_space)
#tuner.fit()


for i in range(100):
    print(pretty_print(algo.train()))

    if i % 5 == 0:
        checkpoint_dir = algo.save()
        print(f"Checkpoint saved in directory {checkpoint_dir}")

algo.evaluate()
