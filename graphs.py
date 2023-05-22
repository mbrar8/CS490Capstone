import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re

folder = "ppo_results/batch10000"


vals = [0] * (len(os.listdir(folder)) + 1)

min_val = 99999999999
min_speed = 9999999
min_stop = 999999
min_ep = 0

for filename in os.listdir(folder):
    f = os.path.join(folder, filename)
    if os.path.isfile(f):
        csv = pd.read_csv(f)
        total_wait = np.mean(csv['T1_accumulated_waiting_time'])
        total_stop = np.mean(csv['T1_stopped'])
        total_speed = np.mean(csv['T1_average_speed'])
        episode_num = int(re.findall(r'\d+', f.split('_')[-1])[0])
        if episode_num == 1 or episode_num == 2:
            continue
        if total_wait < min_val:
            min_val = total_wait
            min_speed = total_speed
            min_stop = total_stop
            min_ep = episode_num
        vals[episode_num] = total_wait


x = [i for i in range(0, len(os.listdir(folder)) + 1)]

plt.plot(x[3:], vals[3:])
plt.xlabel('Episodes')
plt.ylabel('Mean cumulative waiting time')
plt.title('Mean Wait Time vs Episodes for PPO')
plt.savefig('ppo_10000.jpg')
plt.show()

print("Minimum total wait: " + str(min_val))

print("Minimum total stop: " + str(min_stop))

print("Minimum total speed: " + str(min_speed))

print("Minimum episode: " + str(min_ep))

