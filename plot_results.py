import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

results_txt = open("./dqn_Field-v0_log.json",'r').read()
results_json = json.loads(results_txt)
results = pd.DataFrame(results_json)
rewards = results["episode_reward"].values

upper_rate = np.repeat(93.0341992879573, rewards.size)
lower_rate = np.repeat(71.85755595980157, rewards.size)

x_vals = np.arange(start = 0, stop = rewards.size * 200, step = 200)

#test_results = np.array((129.375, 138.350, 137.328, 125.914, 124.160))
#test_eps = np.repeat(300000, 5)

plt.plot(x_vals, rewards)
plt.plot(x_vals, upper_rate)
plt.plot(x_vals, lower_rate)
#plt.plot(test_eps, test_results, 'bo')
plt.ylabel("Episode Reward")
plt.xlabel("Episode")
plt.title("Reward for UCB1 and RL Agent")
plt.show()
