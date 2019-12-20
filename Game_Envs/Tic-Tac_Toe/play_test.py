import gym
import gym_tictactoe
import random
import os,sys
import numpy as np
sys.path.append('../../RL_Algs')
import td_vi_tabular as td_vi
import pickle
import multiprocessing as mp
from functools import partial
from stable_baselines.common.vec_env import SubprocVecEnv

def masked_random_sample(mask):
    #Sample a random action, given a mask that indicates illegal actions.
    valid_acts = []
    for i in range(len(mask)):
        if mask[i] == 0:
            valid_acts.append(i)
    action = random.sample(valid_acts,1)[0]
    return action

def td_vi_init(envs,load):
    #initializes the TD_VI RL model
    priori = []
    if load:
        episode, priori_vals = pickle.load(open('saved_models/saved_vals.pkl',"rb"))
        print('Successfully loaded Model (Pre-trained for {} episodes)'.format(episode))
        for i in range(len(priori_vals)):
            priori.append([i,priori_vals[i]])
        return td_vi.TD_VI(envs.get_attr("total_states", indices = 0), priori, episode=episode)
    else:
        #envs.env_method("init", symbols = [1,2])
        #envs.get_attr("legal_states")
        for i in range(len(envs.get_attr("legal_states", indices = 0))):
            state = envs.get_attr("legal_states", indices = 0)[i]
            winner = envs.env_method("grid_winner", state[1:])#envs.grid_winner(state[1:])
            if winner == 1: #value table is stored such that +1 is a player 1 win, -1 a player 2 win.
                priori.append([i,1])
            elif winner == 2:
                priori.append([i,-1])
        return td_vi.TD_VI(envs.get_attr("total_states", indices = 0), priori)#envs.get_attr("total_states"),priori)

def td_vi_init_na(envs,load):
    #initializes the TD_VI RL model
    env = envs[0]
    priori = []
    if load:
        episode, priori_vals = pickle.load(open('saved_models/saved_vals.pkl',"rb"))
        print('Successfully loaded Model (Pre-trained for {} episodes)'.format(episode))
        for i in range(len(priori_vals)):
            priori.append([i,priori_vals[i]])
        return td_vi.TD_VI(env.total_states,priori,episode=episode)
    else:
        for i in range(len(env.legal_states)):
            state = env.legal_states[i]
            winner = env.grid_winner(state[1:])
            if winner == 1: #value table is stored such that +1 is a player 1 win, -1 a player 2 win.
                priori.append([i,1])
            elif winner == 2:
                priori.append([i,-1])
        return td_vi.TD_VI(env.total_states,priori)

def run_episode_na(p1_control, p2_control, rl_model, envs, done, train, greedy, cpu, episode):
    lock.acquire()
    index = episode % cpu
    env = envs[index]
    state, mask = env.reset()
    while not done:
        player = state[0]
        if(train == False):
            env.render(mode=None)
            state_val = rl_model.get_state_val(env.legal_states.index(state))
            print('Player 1 State Value: {}     Player 2 State Value: {}'.format(state_val,-state_val))
        if player == 1:
        #P1's action selection goes in env.step()
            if(p1_control == 'random'):
                state, mask, reward, done = env.step(masked_random_sample(mask))
            elif(p1_control == 'td_vi'):
                transitions = env.get_possible_transitions(state)
                state_idx = env.legal_states.index(state)
                state, mask, reward, done = env.step(rl_model.choose_act(state_idx,transitions,greedy))
            else:
                print('Invalid Control for P1')

        elif player == 2:
        #P2's action selection goes in env.step()
            if(p2_control == 'random'):
                state, mask, reward, done = env.step(masked_random_sample(mask))
            elif(p2_control == 'td_vi'):
                transitions = env.get_possible_transitions(state)
                state_idx = env.legal_states.index(state)
                state, mask, reward, done = env.step(rl_model.choose_act(state_idx,transitions,greedy))
            else:
                print('Invalid Control for P2')            

        else:
            print('Invalid Turn Counter Value')

        if done:
            winner = env.grid_winner(state[1:])
            if(train == False):
                env.render(mode=None)
            if winner == 1:
                print("Episode {}: Player 1 Win!".format(episode))
            elif winner == 3:
                print("Episode {}: Draw!".format(episode))
            elif winner == 2:
                print("Episode {}: Player 2 Win!".format(episode))
            else:
                print('Invalid End Condition')

            if p1_control == 'td_vi' or p2_control == 'td_vi':
                rl_model.increment_episode()
                if train:
                    rl_model.save_val_table()
    lock.release()

def run_episode(p1_control, p2_control, rl_model, envs, done, train, greedy, cpu, lock, episode): #
    lock.acquire()
    index = episode % cpu
    state, mask = envs.env_method("reset", indices = index)[0]
    while not done:
        player = state[0]
        if(train == False):
            envs.env_method("render", mode = None, indices = index)#render(mode=None)
            state_val = rl_model.get_state_val(envs.get_attr("legal_states", indices = index)[0].index(state))#rl_model.get_state_val(envs.legal_states.index(state))
            print('Player 1 State Value: {}     Player 2 State Value: {}'.format(state_val,-state_val))
        if player == 1:
            #P1's action selection goes in env.step()
            if(p1_control == 'random'):
                state, mask, reward, done = envs.env_method("step", masked_random_sample(mask), indices = index)[0]#envs.step(masked_random_sample(mask))
            elif(p1_control == 'td_vi'):
                transitions = envs.env_method("get_possible_transitions", state, indices = index)[0]#envs.get_possible_transitions(state)
                state_idx = envs.get_attr("legal_states", indices = index)[0].index(state)
                state, mask, reward, done = envs.env_method("step", rl_model.choose_act(state_idx, transitions, greedy), indices = index)[0]#envs.step(rl_model.choose_act(state_idx,transitions,greedy))
            else:
                print('Invalid Control for P1')

        elif player == 2:
            #P2's action selection goes in env.step()
            if(p2_control == 'random'):
                 state, mask, reward, done = envs.env_method("step", masked_random_sample(mask), indices = index)[0]#envs.step(masked_random_sample(mask))
            elif(p2_control == 'td_vi'):
                transitions = envs.env_method("get_possible_transitions", state, indices = index)[0]#envs.get_possible_transitions(state)
                state_idx = envs.get_attr("legal_states", indices = index)[0].index(state)
                state, mask, reward, done = envs.env_method("step", rl_model.choose_act(state_idx, transitions, greedy), indices = index)[0]#envs.step(rl_model.choose_act(state_idx,transitions,greedy))
            else:
                print('Invalid Control for P2')            

        else:
            print('Invalid Turn Counter Value')

        if done:
            winner = envs.env_method("grid_winner", state[1:], indices = index)[0]#envs.grid_winner(state[1:])
            if(train == False):
                envs.env_method("render", mode = None, indicies = index)[0]#env.render(mode=None)
            if winner == 1:
                print("Episode {}: Player 1 Win!".format(episode))
            elif winner == 3:
                print("Episode {}: Draw!".format(episode))
            elif winner == 2:
                print("Episode {}: Player 2 Win!".format(episode))
            else:
                print('Invalid End Condition')

            if p1_control == 'td_vi' or p2_control == 'td_vi':
                rl_model.increment_episode()
                if train:
                    rl_model.save_val_table()
    lock.release()

def init(l):
    global lock
    lock = l

def play(p1_control='td_vi',p2_control='td_vi',episodes=5000,train=True,load=False):
    '''
    Runs the simulated games of Tic-Tac-Toe.

    Parameters
    -----------
    p1_control : string
        The control policy for p1. Currently implemented options are 'random' and 'td_vi' (default = 'td_vi').
    p2_control : string
        The control policy for p2. Currently implemented options are 'random' and 'td_vi' (default = 'td_vi').
    episodes : int
        The number of episodes (games) to run (default = 5000).
    train : bool
        if True (default): Updates the RL model after every action as well as sometimes chooses actions randomly.
        if False: Treat the current RL mdoel as fixed and always use it to choose actions.
    load : bool
        If True: load a saved RL model from a file.
        If False (default): initialize the value table from scratch.
    '''
    cpu = mp.cpu_count() - 1
    envs = [gym.make('TicTacToe-v1') for _ in range(cpu)]#envs = [lambda: gym.make('TicTacToe-v1') for _ in range(cpu)]
    #envs = SubprocVecEnv(envs)
    #envs.env_method("init", symbols = [1,2])
    for env in envs: 
        env.init(symbols = [1,2])

    greedy = not train

    if p1_control == 'td_vi' or p2_control == 'td_vi':
        rl_model = td_vi_init_na(envs, load)#rl_model = td_vi_init(envs, load)
    
    #thread job: episode function, uses an environment instance that isn't been used.
    #lock = mp.Lock()
    #pool = mp.Pool(cpu)
    l = mp.Lock()
    pool = mp.Pool(cpu, initializer=init, initargs = (l,))
    done = False
    #func = partial(run_episode_na, p1_control = p1_control, p2_control = p2_control, rl_model = rl_model, 
    #    envs = envs, done = done, train = train, greedy = greedy, lock = lock ,cpu = cpu)
    func = partial(run_episode_na, p1_control, p2_control, rl_model, envs, done, train, greedy, cpu)
    pool.map(func, range(episodes))
    pool.close
    pool.join
    #for episode in range(episodes):
    #    run_episode(p1_control, p2_control, rl_model, envs, done, train, greedy, cpu, lock, episode)
    



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--p1_control', type=str, default='td_vi',  help='Control Player 1 with RL (td_vi) or Randomly (random).')
    parser.add_argument('--p2_control', type=str, default='td_vi',  help='Control Player 2 with RL (td_vi) or Randomly (random).')
    parser.add_argument('--episodes', type=int, default=7500, help='Number of episodes (games) to simulate.')
    parser.add_argument('--train', type=int, default= 1, help='Whether to train the RL model. 1 is true, 0 is false.')
    parser.add_argument('--load', type=int, default = 0,  help='Whether to load a saved RL model from a file. 1 is true, 0 is false.')

    args = parser.parse_args()
    print(args)

    #run the command "python play.py" with default values to train for 10000 episodes.
    #after training, run the command "python play.py --p2_control=random --episodes=3 --train=0 --load=1" to see the trained RL model play 3 games against a random agent.
    if args.train == 1:
        train = True
    else:
        train = False

    if args.load == 1:
        load = True
    else:
        load = False

    play(p1_control=args.p1_control,p2_control=args.p2_control,episodes=args.episodes,train=train,load=load)