import gym
from gym import spaces, error
import xml.etree.ElementTree as ET
import os, sys

class TicTacToeEnv(gym.Env):
    '''
    The Tic-Tac-Toe environment

    The action variable is one of 9 integers in [0,8], each corresponding to a grid space to place the player's symbol.

    The state variable is a list of 10 elements, the first of which is an integer indicating which player's turn it is (1 or 2), 
    and the final 9 are either 0 (nothing in the grid space), 1 (player 1's symbol 'x' in the grid space), or 2 (player 2's symbol 'o' in the grid space)

    The grid spaces are is indexed as follows:
    |0|1|2|
    |3|4|5|
    |6|7|8|

    '''

    def __init__(self):
        super(TicTacToeEnv, self).__init__()

    def init(self, symbols):
        self.symbols = {
            symbols[0]: "x",
            symbols[1]: "o"
        }
        self.reward = 0 #We don't use this for TD_VI, but the ai-gym API requires the step function to return it. We'll need it for other RL algorithms.
        self.legal_states = [] #list of all legal states for the game to be in

        #There might be a better way to write these for loops but I couldn't think of one. Goal is to iterate through every possible tic-tac-toe state variable.
        for a in range(1,3):
            for b in range(0,3):
                for c in range (0,3):
                    for d in range(0,3):
                        for e in range(0,3):
                            for f in range(0,3):
                                for g in range(0,3):
                                    for h in range(0,3):
                                        for i in range (0,3):
                                            for j in range(0,3):                                             
                                                state = [a,b,c,d,e,f,g,h,i,j]
                                                grid = state[1:]
                                                #This if statement checks that the grid is a state that can be legally reached in the game
                                                if abs(grid.count(1) - grid.count(2)) <= 1:
                                                    winner = self.grid_winner(grid)
                                                    if winner != -1: #-1 means grid has multiple winners, which is impossible.
                                                        self.legal_states.append(state)
        self.total_states = len(self.legal_states)

    def reset(self):
        self.state_vector = [1,0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.action_mask = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        return self.state_vector, self.action_mask

    # ------------------------------------------ GAME STATE CHECK ----------------------------------------
    def grid_winner(self,grid):
        '''
        Checks to see if the current grid is a terminal state, and if so determines the outcome of the game.
        
        Attributes
        -----------
        grid : list
            A 9-element list corresponding to the current tic-tac-toe grid.
        Returns
        ---------
        -1 if the grid shows both a player 1 win and player 2 win (not possible).
        0 if the grid is in a non-terminal state.
        1 if the grid shows at least 1 player 1 win and no player 2 wins.
        2 if the grid shows at least 1 player 2 win and no player 1 wins.
        3 if the grid shows a draw.
        '''
        wins = []
        if (grid[0] == grid[1]) and (grid[0] == grid[2]) and (grid[0] != 0):
            wins.append(grid[0])
        elif (grid[3] == grid[4]) and (grid[3] == grid[5]) and (grid[3] != 0):
            wins.append(grid[3])
        elif (grid[6] == grid[7]) and (grid[6] == grid[8]) and (grid[6] != 0):
            wins.append(grid[6])
        elif (grid[0] == grid[3]) and (grid[0] == grid[6]) and (grid[0] != 0):
            wins.append(grid[0])
        elif (grid[1] == grid[4]) and (grid[1] == grid[7]) and (grid[1] != 0):
            wins.append(grid[1])
        elif (grid[2] == grid[5]) and (grid[2] == grid[8]) and (grid[2] != 0):
            wins.append(grid[2])
        elif (grid[0] == grid[4]) and (grid[0] == grid[8]) and (grid[0] != 0):
            wins.append(grid[0])
        elif (grid[2] == grid[4]) and (grid[2] == grid[6]) and (grid[2] != 0):
            wins.append(grid[2])
        else:
            draw = True
            for i in range(9):
                if grid[i] == 0:
                    draw = False
            if draw:
                return 3
            else:
                return 0    #return 0 for non-terminal state

        #this checks for the impossible case of a player 1 win and player 2 win on the grid
        if 1 in wins and 2 in wins:
            return -1
        else:
            return wins[0]

    # ------------------------------------------ ACTIONS ----------------------------------------
    def step(self, action):
        '''
        Executes an action in the environment. 

        Attributes
        -----------
        action : int
            An integer between 0 and 8 corresponding to a grid spot to place the player symbol.

        Returns
        ---------
        A list of the new state after the action is executed, a new mask of illegal actions for random sampling, the reward, and whether the new state is terminal (done).

        '''
        player = self.state_vector[0]
        is_position_already_used = False
        if self.state_vector[action+1] != 0:
            is_position_already_used = True

        if is_position_already_used:
            self.state_vector[action+1] = "Bad"
            reward_type = 'bad_position'
            done = True
        else:
            self.state_vector[action+1] = player
            grid = self.state_vector[1:]
            self.action_mask[action] = 1
            #Change the current player
            self.state_vector[0] = 2 if self.state_vector[0] == 1 else 1

            if self.grid_winner(grid) == player: #winner
                done = True
            elif self.grid_winner(grid) == 3: #draw
                done = True
            else: #non-terminal state
                done = False
        return self.state_vector, self.action_mask, self.reward, done

    def get_single_transition(self,state,action):
        '''
        Returns the state transition for a "hypothetical" action.

        Attributes
        -----------
        state : list
            A list of 10 integers, where the first is the player turn indicator, and the last 9 are the tic-tac-toe grid.
        action : int
            An integer between 0 and 8 corresponding to a grid spot to place the player symbol.

        Returns
        ---------
        What the new state would be if the hypothetical action were to be executed.

        '''
        new_state = state.copy()
        player = state[0]
        #check if the new state is legal, if not return 'Bad'. For debug only, should never actually happen.
        is_position_already_used = False
        if new_state[action+1] != 0:
            is_position_already_used = True
        if is_position_already_used:
            return 'Bad'
        else: #if it is legal update the new_state grid
            new_state[action+1] = player
        #change player turn of the new state
        if player == 1:
            new_state[0] = 2
        else:
            new_state[0] = 1 
        return new_state


    def get_possible_transitions(self, state):
        '''
        Returns all possible transitions (new_states) from legal actions in the current state and all possible subsequent transitions after the opponent's action (new_new_states).
        A transition is a list of the form [[player_action (int), new_state (int)], [new_new_state_1 (int)],...,new_new_state_N (int)]]
        transitions is the list of all valid transitions from the given state, and new new states from all valid opponent actions from the new state.
        If new_state is a terminal state, the opponent list should be an empty list.

        Attributes
        -----------
        state : list
            A list of 10 integers, where the first is the player turn indicator, and the last 9 are the tic-tac-toe grid.
        action : int
            An integer between 0 and 8 corresponding to a grid spot to place the player symbol.

        Returns
        ---------
        The list of transitions.

        '''
        transitions = []
        for player_act in range(9):
            new_state = self.get_single_transition(state,player_act)
            if(new_state != 'Bad'):
                new_state_idx = self.legal_states.index(new_state)
                if self.grid_winner(new_state[1:]) != 0: #new_state is a terminal state
                    transition = [[player_act,new_state_idx],[]]
                else: #get possible opponent transitions
                    opp_transitions = []
                    for opp_act in range(9):
                        new_new_state = self.get_single_transition(new_state,opp_act)
                        if(new_new_state != 'Bad'):
                            new_new_state_idx = self.legal_states.index(new_new_state)
                            opp_transitions.append(new_new_state_idx)
                    transition = [[player_act,new_state_idx],opp_transitions]
                transitions.append(transition)
        return transitions

    # ------------------------------------------ DISPLAY ----------------------------------------
    def get_state_vector_to_display(self):
        new_state_vector = []
        for value in self.state_vector[1:]:
            if value == 0:
                new_state_vector.append(value)
            else:
                new_state_vector.append(self.symbols[value])
        return new_state_vector

    def print_grid_line(self, grid, offset=0):
        print(" -------------")
        for i in range(3):
            if grid[i + offset] == 0:
                print(" | " + " ", end='')
            else:
                print(" | " + str(grid[i + offset]), end='')
        print(" |")

    def display_grid(self, grid):
        self.print_grid_line(grid)
        self.print_grid_line(grid, 3)
        self.print_grid_line(grid, 6)
        print(" -------------")

        print()

    def render(self, mode=None, close=False):
        self.display_grid(self.get_state_vector_to_display())

    def _close(self):
        return None

    def _seed(self, seed=None):
        return [seed]

'''
#This is just to be used for debugging.
if __name__ == "__main__":
    state = [2,2,2,2,2,1,1,1,1,1]
    print(state[1:])
    env = TicTacToeEnv()
    #print(env.get_state_index(state))
    print(env.multiple_wins(state[1:]))
'''