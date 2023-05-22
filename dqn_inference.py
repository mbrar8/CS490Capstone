import gymnasium as gym
import sumo_rl
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.utils.framework import try_import_tf
from ray.tune.registry import register_env

def create_sumo(env_config={}):
    return sumo_rl.environment.env.SumoEnvironment(net_file='lankershim_intersect2.net.xml',
                    route_file='lankershim_intersect2.rou.xml',
                    out_csv_name='dqn_inter2_inference',
                    use_gui=True,
                    num_seconds=2800,
                    single_agent=True)


register_env("sumo", create_sumo)


algo = Algorithm.from_checkpoint('C:\\Users\mahee\\ray_results\\DQN_sumo_2023-04-02_16-46-27bpfhc8vc\\checkpoint_000006')


env = create_sumo()

pol = algo.get_policy()

obs, info = env.reset()
done = False
while not done:
    next_obs, reward, terminated, truncated, info = env.step(pol.compute_single_action(obs))
    obs = next_obs
    done = terminated or truncated
