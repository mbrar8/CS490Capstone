# CS490Capstone
RL based traffic signal control for CS 490 Capstone


Implemented DQN, PPO, SAC on Lankershim boulevard intersection 2 using SUMO traffic tool, sumo-rl library for implementing RL in SUMO, and rllib for RL algorithms.

Each model is now implemented with three-layer neural network. DQN has best performance, followed closely by PPO. Generalization to other conditions is not yet finished, expect SAC to perform well there.


Images: Saved images of summed cumulative vehicle delays across episodes for different models

baseline_results: Results of random sampling actions

baseline_results_constant: Results of fixed time signal

dqn_generalization: DQN results on new intersection/traffic conditions

dqn_queue_results: DQN results with queue reward

dqn_results: DQN results

** Same for PPO and SAC **

sumo_inter2: SUMO config (not necessary for training)

baseline.py: Train baseline model

cologne1: Alternate intersection

dqn.py/ppo.py/sac.py: Train DQN/PPO/SAC models

inference.py/dqn_inference.py/ppo_inference.py/sac_inference.py: Run trained policy for visualization

inter2.sumocfg - SUMO simulation file to view the intersection (not necessary for training)

lankershim_intersect2.net.xml - Intersection network for SUMO

lankershim_intersect2.rou.xml - Traffic simulation for SUMO (specifies routes and how many vehicles per route)

lankershim_intesect2_mod.rou.xml - Modified traffic conditions to test generalization

osm.view.xml - SUMO file for viewing simulation

