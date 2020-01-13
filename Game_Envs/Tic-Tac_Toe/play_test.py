import warnings
warnings.filterwarnings('ignore')
import gym
import gym_tictactoe
import random
import os,sys
import numpy as np
sys.path.append('../../RL_Algs')
import td_vi_tabular as td_vi
import pickle
import multiprocessing as mp
from multiprocessing.managers import BaseManager
from functools import partial
from stable_baselines.common.vec_env import SubprocVecEnv
from pytictoc import TicToc


def masked_random_sample(mask):
    #Sample a random action, given a mask that indicates illegal actions.
    valid_acts = []
    for i in range(len(mask)):
        if mask[i] == 0:
            valid_acts.append(i)
    action = random.sample(valid_acts,1)[0]
    return action

def get_priori_vals(envs,load):
    env = envs[0]
    priori = []
    if load:
        episode, priori_vals = pickle.load(open('saved_models/saved_vals.pkl',"rb"))
        print('Successfully loaded Model (Pre-trained for {} episodes)'.format(episode))
        for i in range(len(priori_vals)):
            priori.append([i,priori_vals[i]])
    else:
        episode = 0
        for i in range(len(env.legal_states)):
            state = env.legal_states[i]
            winner = env.grid_winner(state[1:])
            if winner == 1: #value table is stored such that +1 is a player 1 win, -1 a player 2 win.
                priori.append([i,1])
            elif winner == 2:
                priori.append([i,-1])
    return priori,episode

def run_episode(p1_control, p2_control,rl_model, envs, done, train, greedy,iterable):
    #get the first available environment from the envs list
    occ_lock.acquire()
    for i in range(len(occupied)):
        if occupied[i] == 0:
            occupied[i] = 1
            env_index = i
            break
    occ_lock.release()

    env = envs[env_index]
    state, mask = env.reset()

    #get the ordered episode value and increment it for the next thread
    ep_lock.acquire()
    ep = episode.value
    episode.value += 1
    ep_lock.release()

    while not done:
        player = state[0]
        if(train == False):
            env.render(mode=None)
            state_val = rl_model.get_state_val(vals,env.legal_states.index(state))
            print('Player 1 State Value: {}     Player 2 State Value: {}'.format(state_val,-state_val))
        if player == 1:
        #P1's action selection goes in env.step()
            if(p1_control == 'random'):
                state, mask, reward, done = env.step(masked_random_sample(mask))
            elif(p1_control == 'td_vi'):
                transitions = env.get_possible_transitions(state)
                state_idx = env.legal_states.index(state)
                state, mask, reward, done = env.step(rl_model.choose_act(vals,state_idx,transitions,greedy,ep,val_lock))
            else:
                print('Invalid Control for P1')

        elif player == 2:
        #P2's action selection goes in env.step()
            if(p2_control == 'random'):
                state, mask, reward, done = env.step(masked_random_sample(mask))
            elif(p2_control == 'td_vi'):
                transitions = env.get_possible_transitions(state)
                state_idx = env.legal_states.index(state)
                state, mask, reward, done = env.step(rl_model.choose_act(vals,state_idx,transitions,greedy,ep,val_lock))
            else:
                print('Invalid Control for P2')            

        else:
            print('Invalid Turn Counter Value')

        if done:
            winner = env.grid_winner(state[1:])
            if(train == False):
                env.render(mode=None)
            if winner == 1:
                print("Episode {}: Player 1 Win!".format(ep))
            elif winner == 3:
                print("Episode {}: Draw!".format(ep))
            elif winner == 2:
                print("Episode {}: Player 2 Win!".format(ep))
            else:
                print('Invalid End Condition')

            if p1_control == 'td_vi' or p2_control == 'td_vi':
                #rl_model.increment_episode()
                if train:
                    rl_model.save_val_table(vals,ep)

            #free this environment to be used by another thread
            occ_lock.acquire()
            occupied[env_index] = 0
            occ_lock.release()

def init(occ,ol,vals,vl,episode,el):
    global occ_lock
    occ_lock = ol
    global ep_lock
    ep_lock = el
    global val_lock
    val_lock = vl

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
    greedy = not train
    if train:
        cpu = 16 #optimal value varies per machine. Test multiple values < mp.cpu_count()
    else:
        cpu = 1

    envs = [gym.make('TicTacToe-v1') for _ in range(cpu)]
    for env in envs: 
        env.init(symbols = [1,2])

    global vals
    vals = mp.Array('d',envs[0].total_states,lock=False)
    priori_vals, start_episode = get_priori_vals(envs,load)

    if train:
        iterable = range(start_episode,episodes)
        total_eps = episodes-start_episode
        cpu = min(episodes-start_episode,cpu)
    else:
        iterable = range(episodes)
        total_eps = episodes

    for val in priori_vals:
        vals[val[0]] = val[1]
    if p1_control == 'td_vi' or p2_control == 'td_vi':
        rl_model = td_vi.TD_VI()#rl_model = td_vi_init(envs, load)

    global occupied
    occupied = mp.Array('i',len(envs),lock=False)
    ol = mp.Lock()
    global episode
    episode = mp.Value('i',start_episode)
    el = mp.Lock()
    vl = mp.Lock()
    t = TicToc()
    t.tic()
    pool = mp.Pool(cpu, initializer=init, initargs =(occupied,ol,vals,vl,episode,el,))
    done = False
    func = partial(run_episode, p1_control, p2_control, rl_model, envs, done, train, greedy)
    pool.map(func, iterable)
    pool.close()
    pool.join()
    t.toc('Completed {} Episodes in '.format(total_eps))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--p1_control', type=str, default='td_vi',  help='Control Player 1 with RL (td_vi) or Randomly (random).')
    parser.add_argument('--p2_control', type=str, default='td_vi',  help='Control Player 2 with RL (td_vi) or Randomly (random).')
    parser.add_argument('--episodes', type=int, default=25000, help='Number of episodes (games) to simulate.')
    parser.add_argument('--train', type=int, default= 1, help='Whether to train the RL model. 1 is true, 0 is false.')
    parser.add_argument('--load', type=int, default = 0,  help='Whether to load a saved RL model from a file. 1 is true, 0 is false.')

    args = parser.parse_args()
    print(args)

    #run the command "python play_test.py" with default values to train for 10000 episodes.
    #after training, run the command "python play_test.py --episodes=3 --train=0 --load=1" to see the trained RL model play 3 games against itself
    if args.train == 1:
        train = True
    else:
        train = False

    if args.load == 1:
        load = True
    else:
        load = False

    play(p1_control=args.p1_control,p2_control=args.p2_control,episodes=args.episodes,train=train,load=load)