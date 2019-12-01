import numpy as np

class Player:
    # Player, Base Clasee with name, action_space, actions (history), and states (history)
    def __init__(self, name):
        self.name = name
        self.action_space = []
        self.actions = []
        self.states = []

class Agent(Player):
    # Agent, Derived Class of Player Class, with q learning hyper-parameters(alpha, gamma, epsilon) and q table[(state, action)]
    def __init__(self, name=None, alpha=0.2, gamma=0.9, epsilon=0.3):
        Player.__init__(self, name)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}
    
    def take_action(self, state):
        # explore or choose an action with maximum reward
        if np.random.uniform(0,1) < self.epsilon:
            # explore random action
            action = np.random.choice(len(self.action_space))
            action = self.action_space[action]
        else:
            # choose logical action
            future_reward = -500
            for a in self.action_space:
                if (state, a) in self.q_table and self.q_table[(state, a)] > future_reward:
                    future_reward = self.q_table[(state, a)]
                    action = a
            # if has not visited the state yet, choose random action
            if future_reward == -500:
                action = np.random.choice(len(self.action_space))
                action = self.action_space[action]
        
        # log into actions and states (history)
        self.actions.append(action)
        self.states.append(state)

        return action

    def update_q_table(self, state, action, new_state, reward):
        # update q value at q_table[(state, action)]
        # find maximum reward from new_state that results from taking an action from (current) state
        future_reward = -500
        for a in self.action_space:
            # if not visited yet, initialize to 0
            if (new_state, a) not in self.q_table:
                self.q_table[(new_state, a)] = 0
            if self.q_table[(new_state, a)] > future_reward:
                future_reward = self.q_table[(new_state, a)]
        
        # if not visited yet, initialize to 0
        if (state, action) not in self.q_table:
            self.q_table[(state, action)] = 0
        # update rule: q-value(t+1) = (1-learning_rate) * q-value(t) + learning_rate * (reward + decay_factor * future_reward)
        self.q_table[(state, action)] = (1-self.alpha)*self.q_table[(state, action)] + self.alpha * (reward + self.gamma * future_reward)

class Human(Player):
    def __init__(self, name=None):
        Player.__init__(self, name)
    
    def take_action(self, state):
        for i in range(len(self.action_space)):
            print(f'Press {i+1} to choose {self.action_space[i]}')
        action = int(input("What's your choice: "))
        action = self.action_space[action-1]
        self.actions.append(action)
        self.states.append(state)
        return action

class Game:
    def __init__(self, p1, p2=None):
        self.p1 = p1
        self.p1.action_space = []
        self.done = False
        if p2 is not None:
            self.p2 = p2
            self.p2.action_space = []

class TicTacToe(Game):
    def __init__(self, p1, p2):
        Game.__init__(self, p1, p2)
        self.board = np.zeros((3,3))
        self.state = self.hash_function(self.board)
        self.playing_now = 1
        self.winner = 0
        for i in range(3):
            for j in range(3):
                self.p1.action_space.append((i, j))
                self.p2.action_space.append((i, j))
        
    def hash_function(self, board):
        return str(board.reshape(9))

    def is_done(self):
        # win
        if np.max(self.board.sum(1)) == 3 or np.min(self.board.sum(1)) == -3 or np.max(self.board.sum(0)) == 3 or np.min(self.board.sum(0)) == -3 or np.sum(self.board.diagonal()) == 3*self.playing_now or np.sum(np.flip(self.board, 1).diagonal()) == 3*self.playing_now:
            self.done = True
            self.winner = self.playing_now
            return True
        # tie
        if len(self.p1.action_space) == 0 and len(self.p2.action_space) == 0:
            self.done = True
            self.winner = 0
            return True
        
        return False
    
    def step(self, action):
        self.board[action[0], action[1]] = self.playing_now
        self.p1.action_space.remove(action)
        self.p2.action_space.remove(action)
        self.state = self.hash_function(self.board)
        
        self.is_done()

        # winner
        if self.done is True and self.winner is not 0:
            reward = 1
            return self.state, reward, self.done
        # tie
        if self.done is True and self.winner is 0:
            reward = 0
            return self.state, reward, self.done

        # continue playing if not done
        self.playing_now *= -1
        reward = -1
        return self.state, reward, self.done

class ConnectFive(Game):
    def __init__(self, p1, p2):
        Game.__init__(self, p1, p2)
        self.board = np.zeros((20,20))
        self.state = self.hash_function(self.board)
        self.playing_now = 1
        self.winner = 0
        for i in range(20):
            for j in range(20):
                self.p1.action_space.append((i,j))
                self.p2.action_space.append((i,j))

    def hash_function(self, board):
        return str(board.reshape(400))

    def is_done(self):
        # winner
        for i in range(20-5+1):
            for j in range(20-5+1):
                block = self.board[i:i+5,j:j+5]
                if np.max(block.sum(1)) == 5 or np.min(block.sum(1)) == -5 or np.max(block.sum(0)) == 5 or np.min(block.sum(0)) == -5 or np.sum(block.diagonal()) == 5*self.playing_now or np.sum(np.flip(block, 1).diagonal()) == 5*self.playing_now:
                    self.done = True
                    self.winner = self.playing_now
                    return True
        # tie
        if len(self.p1.action_space) == 0 and len(self.p2.action_space) == 0:
            self.done = True
            self.winner = 0
            return True
        
        return False

    def step(self, action):
        # update the board according to the action
        self.board[action[0], action[1]] = self.playing_now
        # update action space
        self.p1.action_space.remove(action)
        self.p2.action_space.remove(action)
        # update state
        self.state = self.hash_function(self.board)
        # chekc if done
        self.is_done()

        # winner
        if self.done is True and self.winner is not 0:
            reward = 1
            return self.state, reward, self.done
        # tie
        if self.done is True and self.winner is 0:
            reward = 0
            return self.state, reward, self.done

        # continue playing if not done
        self.playing_now *= -1
        reward = -1
        return self.state, reward, self.done
    