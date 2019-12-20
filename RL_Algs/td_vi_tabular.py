import numpy as np
import random
import pickle

class TD_VI: 
    #Implements an instance of tabular Temporal Difference Value Iteration (TDVI).

    def __init__(self,total_states,priori_vals,alpha=0.3,epsilon=0.9,episode=0):
        '''
        Attributes
        -----------
        total_states : int
            the total number of possible states in the environment
        priori_vals : list
            A list of states in which the value is know a priori, such as win condition states, or when loading a saved value table from a file.
            Elements of priori_vals should have the format [state_index (int), value (float)].
        alpha : float
            The learning rate of the TD learning. Controls how much to change our previous value estimate in light of new data (default 0.3).
        epsilon : float
            The probability that a random action is chosen instead of a greedy action. This value decays over the course of training (default 0.9).
        '''

        self.P2_IDX = 8954 #index at which states correspond to player 2's turn.
        self.total_states = total_states
        self.vals = np.zeros(self.total_states)
        self.alpha = alpha
        self.episode = episode
        self.epsilon = epsilon*pow(0.999,episode) #if episode > 0, account for epsilon decay

        #initializes the value table from the a priori values passed in.
        for val in priori_vals:
            self.vals[val[0]] = val[1]

    def choose_act(self,state,transitions,greedy,lock):
        '''a transition is a list of the form [[player_action (int), new_state (int)], [new_new_state_1 (int)],...,new_new_state_N (int)]]
        transitions is the list of all valid transitions from the given state, and new new states from all valid opponent actions from the new state
        if new_state is a terminal state, the opponent list should be an empty list'''

        rand = random.uniform(0,1)
        if rand < self.epsilon and greedy == False:
            #randomly sample from legal actions
            chosen_transition = random.sample(transitions,1)[0]
            chosen_act = chosen_transition[0][0]
            new_state = chosen_transition[0][1]
        else:
            #decide action based on the maximum Q-value of each legal action
            if state < self.P2_IDX:
                temp_vals = self.vals #value table is stored such that +1 is a player 1 win and -1 is a player 2 win. 
            else:
                temp_vals = [-1*v for v in self.vals] #need to invert values to select actions for player 2.

            Q_vals = [] #Q_val corresponding to each action in transitons
            for transition in transitions:
                new_state = transition[0][1]
                if transition[1] == []: #this means new_state is a terminal state, and the value is known a priori
                    Q_vals.append(temp_vals[new_state])
                else: #we need to get the value based on the opponents worst-case move selection from the new state
                    min_Q = float('inf') 
                    for new_new_state in transition[1]: #this for loop finds the min Q-value of our opponents state transitions (worst-case opponent move)
                        Q_val = temp_vals[new_new_state]
                        if(Q_val < min_Q):
                            min_Q = Q_val
                            min_state = new_new_state
                    Q_vals.append(min_Q)
                    if(greedy == False):
                        new_new_state = min_state
                        self.update_value(new_state,new_new_state, lock) #update the value of new_state based on the value of new_new_state

            max_Q = max(Q_vals) #select the max of the Q_vals calculated in the previous for loop, which will tell us which action to select.
            best_transitions = [i for i,j in enumerate(Q_vals) if j == max_Q] #if there is a tie between multiple actions, randomly choose one of them
            chosen_t = random.sample(best_transitions,1)[0] 
            chosen_act = transitions[chosen_t][0][0] 
            new_state = transitions[chosen_t][0][1]
            if(greedy == False):
                self.update_value(state,new_state, lock) #update the value of state based on the value of new_state
        return chosen_act

    def update_value(self,state,new_state,lock):
        #updates the value function of the state and transition state according to the temporal difference update rule
        lock.acquire()
        self.vals[state] += self.alpha*(self.vals[new_state] - self.vals[state])
        lock.release()

    def increment_episode(self):
        self.episode += 1
        self.epsilon *= 0.999 #decay the probability of selecting a random action when calling choose_act

    def save_val_table(self):
        #saves the value table for a particular player_action
        pickle_file = './saved_models/saved_vals.pkl'
        pickle.dump([self.episode,self.vals.tolist()],open(pickle_file,"wb"))

    def get_state_val(self,state_idx):
        return self.vals[state_idx]