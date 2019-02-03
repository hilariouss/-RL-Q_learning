# This source code is about the 'Q-learning', which utilizes the Q table
# The grid world is 4 x 4 (state), and the dimension of action is 4(up, down, right, left)

import tensorflow as tf
import numpy as np
import gym
import random
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')

Q_table = np.zeros([env.observation_space.n, env.action_space.n])
lr = 0.85
y = 0.99
num_episodes = 2000
rList = []
for i in range(num_episodes):
    s = env.reset()
    rAll = 0
    d = False
    j = 0
    while j < 99:
        j += 1
        # Q테이블을 참고하여 그리디하게 액션 선택(max)
        a = np.argmax(Q_table[s, :] + np.random.randn(1, env.action_space.n)*(1./(i+1)))
        s1, r, d, _ = env.step(a)
        # 환경으로부터 새로운 환경과 보상을 얻는다.
        Q_table[s, a] = Q_table[s, a] + lr*(r + y*np.max(Q_table[s1, :]) - Q_table[s,a])
        rAll += r
        s = s1
        if d == True:
            break

rList.append(rAll)
print("Score over time: " + str(sum(rList)/num_episodes))
print("Final Q-table values")
print(Q_table)