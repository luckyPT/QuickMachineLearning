import gym # loading the Gym library
import numpy as np
import time
env = gym.make("FrozenLake-v1", render_mode='human')
env.reset()
env.render()

print("Action space: ", env.action_space)
print("Observation space: ", env.observation_space)

# 值迭代逻辑：https://github.com/apachecn/apachecn-dl-zh/blob/master/docs/rl-tf/03.md
"""
U = np.zeros([env.observation_space.n])

#since terminal states have utility values equal to their reward
U[15] = 1 #goal state
U[[5,7,11,12]] = -1 #hole states
termS = [5,7,11,12,15] #terminal states
#set hyperparameters
y = 0.8 #discount factor lambda

eps = 1e-3 #threshold if the learning difference i.e. prev_u - U goes below this value break the learning

i=0
while(True):
    i+=1
    prev_u = np.copy(U)
    for s in range(env.observation_space.n):
        q_sa = [sum([p*(r + y*prev_u[s_]) for p, s_, r, _ in env.env.P[s][a]]) for a in range(env.action_space.n)]
        if s not in termS:
            U[s] = max(q_sa)
    if (np.sum(np.fabs(prev_u - U)) <= eps):
        print ('Value-iteration converged at iteration# %d.' %(i+1))
        break

print("After learning completion printing the utilities for each states below from state ids 0-15")
print()
print(U[:4])
print(U[4:8])
print(U[8:12])
print(U[12:16])
"""

MAX_ITERATIONS = 100
for i in range(MAX_ITERATIONS):
    random_action = env.action_space.sample()
    new_state, reward, truncated, done, info = env.step(random_action)
    env.render()
    if done:
        print("done")
        break
    if truncated:
        print("truncated")
        break
time.sleep(3)