import copy
import random

from game import Game, states

HIT = 0
STAND = 1
DISCOUNT = 0.95 #This is the gamma value for all value calculations

class Agent:
    def __init__(self):

        # For MC values
        self.MC_values = {} # Dictionary: Store the MC value of each state
        self.S_MC = {}      # Dictionary: Store the sum of returns in each state
        self.N_MC = {}      # Dictionary: Store the number of samples of each state
        # MC_values should be equal to S_MC divided by N_MC on each state (important for passing tests)

        # For TD values
        self.TD_values = {}  # Dictionary: Store the TD value of each state
        self.N_TD = {}       # Dictionary: Store the number of samples of each state

        # For Q-learning values
        self.Q_values = {}   # Dictionary: Store the Q-Learning value of each state and action
        self.N_Q = {}        # Dictionary: Store the number of samples of each state for each action

        # Initialization of the values
        for s in states:
            self.MC_values[s] = 0
            self.S_MC[s] = 0
            self.N_MC[s] = 0
            self.TD_values[s] = 0
            self.N_TD[s] = 0
            self.Q_values[s] = [0,0] # First element is the Q value of "Hit", second element is the Q value of "Stand"
            self.N_Q[s] = [0,0] # First element is the number of visits of "Hit" at state s, second element is the Q value of "Stand" at s

        # Game simulator
        # NOTE: see the comment of `init_cards()` method in `game.py` for description of the initial game states       
        self.simulator = Game()

    # NOTE: do not modify this function
    # This is the fixed policy given to you, for which you need to perform MC and TD policy evaluation. 
    @staticmethod
    def default_policy(state):
        user_sum = state[0]
        user_A_active = state[1]
        actual_user_sum = user_sum + user_A_active * 10
        if actual_user_sum < 14:
            return 0
        else:
            return 1

    # NOTE: do not modify this function
    # This is the fixed learning rate for TD and Q learning. 
    @staticmethod
    def alpha(n):
        return 10.0/(9 + n)
   
    def make_one_transition(self, action):
        if self.simulator.game_over():
            return None, 0

        if action == HIT:
            self.simulator.act_hit()
        else:
            self.simulator.act_stand()

        next_state = self.simulator.state
        reward = self.simulator.check_reward()

        return next_state, reward

    def MC_run(self, num_simulation, tester=False):

        # Perform num_simulation rounds of simulations in each cycle of the overall game loop
        for simulation in range(num_simulation):

            # Do not modify the following three lines
            if tester:
                self.tester_print(simulation, num_simulation, "MC")
            self.simulator.reset()

            trajectory = []
            while not self.simulator.game_over():
                state = self.simulator.state
                action = self.default_policy(state)
                next_state, reward = self.make_one_transition(action)
                trajectory.append((state, reward))

            G = 0
            for state, reward in reversed(trajectory):
                G = reward + DISCOUNT * G
                self.S_MC[state] += G
                self.N_MC[state] += 1
                self.MC_values[state] = self.S_MC[state] / self.N_MC[state]

            if self.simulator.game_over():
                final_state = self.simulator.state
                final_reward = self.simulator.check_reward()
                self.S_MC[final_state] += final_reward
                self.N_MC[final_state] += 1
                self.MC_values[final_state] = self.S_MC[final_state] / self.N_MC[final_state]
    
    def TD_run(self, num_simulation, tester=False):

        # Perform num_simulation rounds of simulations in each cycle of the overall game loop
        for simulation in range(num_simulation):

            # Do not modify the following three lines
            if tester:
                self.tester_print(simulation, num_simulation, "TD")
            self.simulator.reset()

            while not self.simulator.game_over():
                state = self.simulator.state
                action = self.default_policy(state)
                next_state, reward = self.make_one_transition(action)

                self.N_TD[state] += 1
                alpha = self.alpha(self.N_TD[state])

                next_value = 0 if self.simulator.game_over() else self.TD_values[next_state]

                td_error = reward + DISCOUNT * next_value - self.TD_values[state]
                self.TD_values[state] += alpha * td_error

            if self.simulator.game_over():
                final_state = self.simulator.state
                final_reward = self.simulator.check_reward()
                self.N_TD[final_state] += 1
                alpha = self.alpha(self.N_TD[final_state])
                td_error = final_reward - self.TD_values[final_state]
                self.TD_values[final_state] += alpha * td_error
                
    def Q_run(self, num_simulation, tester=False, epsilon=0.4):

        # Perform num_simulation rounds of simulations in each cycle of the overall game loop
        for simulation in range(num_simulation):

            # Do not modify the following three lines
            if tester:
                self.tester_print(simulation, num_simulation, "Q")
            self.simulator.reset()

            while not self.simulator.game_over():
                state = self.simulator.state
                action = self.pick_action(state, epsilon)
                next_state, reward = self.make_one_transition(action)

                self.N_Q[state][action] += 1
                alpha = self.alpha(self.N_Q[state][action])

                if self.simulator.game_over():
                    max_next_q = 0
                else:
                    max_next_q = max(self.Q_values[next_state][HIT], self.Q_values[next_state][STAND])

                q_error = reward + DISCOUNT * max_next_q - self.Q_values[state][action]
                self.Q_values[state][action] += alpha * q_error

    def pick_action(self, s, epsilon):
        if random.random() < epsilon:
            return random.randint(0, 1)
        else:
            if self.Q_values[s][HIT] > self.Q_values[s][STAND]:
                return HIT
            elif self.Q_values[s][STAND] > self.Q_values[s][HIT]:
                return STAND
            else:
                return random.randint(0, 1)

    ####Do not modify anything below this line####

    #Note: do not modify
    def autoplay_decision(self, state):
        hitQ, standQ = self.Q_values[state][HIT], self.Q_values[state][STAND]
        if hitQ > standQ:
            return HIT
        if standQ > hitQ:
            return STAND
        return HIT #Before Q-learning takes effect, just always HIT

    # NOTE: do not modify
    def save(self, filename):
        with open(filename, "w") as file:
            for table in [self.MC_values, self.TD_values, self.Q_values, self.S_MC, self.N_MC, self.N_TD, self.N_Q]:
                for key in table:
                    key_str = str(key).replace(" ", "")
                    entry_str = str(table[key]).replace(" ", "")
                    file.write(f"{key_str} {entry_str}\n")
                file.write("\n")

    # NOTE: do not modify
    def load(self, filename):
        with open(filename) as file:
            text = file.read()
            MC_values_text, TD_values_text, Q_values_text, S_MC_text, N_MC_text, NTD_text, NQ_text, _  = text.split("\n\n")
            
            def extract_key(key_str):
                return tuple([int(x) for x in key_str[1:-1].split(",")])
            
            for table, text in zip(
                [self.MC_values, self.TD_values, self.Q_values, self.S_MC, self.N_MC, self.N_TD, self.N_Q], 
                [MC_values_text, TD_values_text, Q_values_text, S_MC_text, N_MC_text, NTD_text, NQ_text]
            ):
                for line in text.split("\n"):
                    key_str, entry_str = line.split(" ")
                    key = extract_key(key_str)
                    table[key] = eval(entry_str)

    # NOTE: do not modify
    @staticmethod
    def tester_print(i, n, name):
        print(f"\r  {name} {i + 1}/{n}", end="")
        if i == n - 1:
            print()
