import env
import numpy as np
import csv

p1 = env.Agent()
p2 = env.Agent()

num_episodes = 2
for episode in range(num_episodes):
    g = env.ConnectFive(p1, p2)
    current_state = g.state
    while True:
        action = p1.take_action(current_state)
        new_state, reward, done = g.step(action)
        p1.update_q_table(current_state, action, new_state, reward)
        if done:
            break
        current_state = new_state
        action = p2.take_action(current_state)
        new_state, reward, done = g.step(action)
        p2.update_q_table(current_state, action, new_state, reward)
        if done:
            break
        current_state = new_state
    if (episode+1) % 100 == 0:
        print(episode+1)
    if episode == 10000:
        p1.epsilon = 0.2
        p2.epsilon = 0.2
    if episode == 100000:
        p1.epsilon = 0.1
        p2.epsilon = 0.1
    if episode == 1000000:
        p1.epsilon = 0.1
        p2.epsilon = 0.1

w = csv.writer(open("c_q_table_p1.csv", "w"))
for key, val in p1.q_table.items():
    w.writerow([key, val])
w = csv.writer(open("c_q_table_p2.csv", "w"))
for key, val in p2.q_table.items():
    w.writerow([key, val])
    
p2 = env.Human()
g = env.ConnectFive(p1, p2)
current_state = g.state
while True:
    action = p1.take_action(current_state)
    current_state, _, done = g.step(action)
    print(g.board)
    print(g.p1.action_space)
    if done:
        break
    action = p2.take_action(current_state)
    current_state, _, done = g.step(action)
    print(g.board)
    if done:
        break