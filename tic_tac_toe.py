import env
import numpy as np
import csv

p1 = env.Agent()
p2 = env.Agent()

num_episodes = 10000
p1_scores = []
p2_scores = []
best_p1_score = -500
best_p2_score = -500

for episode in range(num_episodes):
    g = env.TicTacToe(p1, p2)
    current_state = g.state
    p1_score = 0
    p2_score = 0
    while True:
        action = p1.take_action(current_state)
        new_state, reward, done = g.step(action)
        p1_score += reward
        p1.update_q_table(current_state, action, new_state, reward)
        if done:
            break
        current_state = new_state
        action = p2.take_action(current_state)
        new_state, reward, done = g.step(action)
        p2_score += reward
        p2.update_q_table(current_state, action, new_state, reward)
        if done:
            break
        current_state = new_state
    p1_scores.append(p1_score)
    p2_scores.append(p2_score)
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

w = csv.writer(open("t_q_table_p1.csv", "w"))
for key, val in p1.q_table.items():
    w.writerow([key, val])
w = csv.writer(open("t_q_table_p2.csv", "w"))
for key, val in p2.q_table.items():
    w.writerow([key, val])
    
p2 = env.Human()
g = env.TicTacToe(p1, p2)
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