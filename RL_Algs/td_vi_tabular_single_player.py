import numpy as np
import random
import pickle
import multiprocessing as mp

class TD_VI(object): 
    #Implements an instance of tabular Temporal Difference Value Iteration (TDVI).

    def __init__(self,alpha=0.3,epsilon=0.9,gamma=0.9):
        '''
        Attributes
        -----------
        alpha : float
            The learning rate of the TD learning. Controls how much to change our previous value estimate in light of new data (default 0.3).
        epsilon : float
            The probability that a random action is chosen instead of a greedy action. This value decays over the course of training (default 0.9).
        '''
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

    def choose_act(self,vals,state,transitions,greedy,episode,lock):
        '''a transition is a list of tuples of the form [player_action (int), new_state (int)]'''
        ep = self.epsilon*pow(0.999,episode)
        rand = random.uniform(0,1)
        if rand < ep and greedy == False:
            #randomly sample from legal actions
            chosen_transition = random.sample(transitions,1)
            chosen_act = chosen_transition[0]
            new_state = chosen_transition[1]
        else:
            temp_vals = vals
            Q_vals = [] #Q_val corresponding to each action in transitons
            for transition in transitions:
                new_state = transition[1]
                Q_vals.append(temp_vals[new_state])
            max_Q = max(Q_vals) #select the max of the Q_vals calculated in the previous for loop, which will tell us which action to select.
            best_transitions = [i for i,j in enumerate(Q_vals) if j == max_Q] #if there is a tie between multiple actions, randomly choose one of them
            chosen_t = random.sample(best_transitions,1)
            chosen_act = transitions[chosen_t][0] 
            new_state = transitions[chosen_t][1]
            if(greedy == False):
                self.update_value(vals,state,new_state,lock) #update the value of state based on the value of new_state
        return chosen_act

    def update_value(self,vals,state,new_state,lock):
        #updates the value function of the state and transition state according to the temporal difference update rule
        lock.acquire()
        vals[state] += self.alpha*(vals[new_state] - vals[state])
        lock.release()

    def save_val_table(self,vals,episode):
        #saves the value table for a particular player_action
        save_vals = [val for val in vals]
        pickle_file = './saved_models/saved_vals.pkl'
        pickle.dump([episode,save_vals],open(pickle_file,"wb"))

    def get_state_val(self,vals,state_idx):
        return vals[state_idx]