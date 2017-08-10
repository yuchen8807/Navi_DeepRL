#!/usr/bin/env python
"""Run turtlebot to navigation with deepRL."""

import argparse
import os
import tensorflow as tf
from algorithm.DQNAgent_yuchen import DQNAgent
from agent.agent_mobile import AgentMobile
import gym  # for testing

# <<<<<<<<<<<<<<<<<<<<<<<<<
def main():
    parser = argparse.ArgumentParser(description='Run DQN on Atari Breakout')
    parser.add_argument('-o', '--output', default='./log/', help='Directory to save data to')
    parser.add_argument('--env', default='Seaquest-v0', help='Atari env name')
    #learning parameters.
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--gamma', default=0.99, type=float, help='Discount factor')
    parser.add_argument('--batch_size', default=32, type=int, help='Minibatch size')
    parser.add_argument('--learning_rate', default=0.0001, type=float, help='Learning rate')
    parser.add_argument('--initial_epsilon', default=1.0, type=float, help='Initial exploration probability in epsilon-greedy')
    parser.add_argument('--final_epsilon', default=0.05, type=float, help='Final exploration probability in epsilon-greedy')
    parser.add_argument('--exploration_steps', default=1000000, type=int, help='Number of steps over which the initial value of epsilon is linearly annealed to its final value')
    parser.add_argument('--num_samples', default=12000000, type=int, help='Number of training samples from the environment in training')
    #image:
    parser.add_argument('--num_frames', default=4, type=int, help='Number of frames to feed to Q-Network')
    parser.add_argument('--frame_width', default=84, type=int, help='Resized frame width')
    parser.add_argument('--frame_height', default=84, type=int, help='Resized frame height')
    #update or save frequency.
    parser.add_argument('--replay_memory_size', default=400000, type=int, help='Number of replay memory the agent uses for training')
    parser.add_argument('--target_update_freq', default=10000, type=int, help='The frequency with which the target network is updated')
    parser.add_argument('--train_freq', default=4, type=int, help='The frequency of actions wrt Q-network update')
    parser.add_argument('--save_freq', default=50000, type=int, help='The frequency with which the network is saved')
    parser.add_argument('--eval_freq', default=100000, type=int, help='The frequency with which the policy is evlauted')
    parser.add_argument('--num_burn_in', default=20000, type=int, help='Number of steps to populate the replay memory before training starts')
    parser.add_argument('--max_episode_length', default = 1000, type=int, help = 'max length of each episode')
    parser.add_argument('--num_episodes_at_test', default = 20, type=int, help='Number of episodes the agent plays at test')
    #mode.
    parser.add_argument('--load_network', default=False, action='store_true', help='Load trained mode')
    parser.add_argument('--load_network_path', default='', help='the path to the trained mode file')
    parser.add_argument('--train', default=True, dest='train', action='store_true', help='Train mode')
    parser.add_argument('--test', dest='train', action='store_false', help='Test mode')
    parser.add_argument('--ddqn', default=False, dest='ddqn', action='store_true', help='enable ddqn')
    parser.add_argument('--no_monitor', default=False, action='store_true', help='do not record video')

    args = parser.parse_args()
    print("==== Output saved to: ", args.output)
    print("==== Args used:")
    print(args)

    # define mobile environment.
    env =  AgentMobile()
    #env = gym.make(args.env) # atari environment 

    # here is where you should start up a session,
    # create your DQN agent, create your model, etc.
    # then you can run your fit method.
    '''
    num_linearActions = 3
    num_angleActions = 3
    print(" #actions dimension: ", num_linearActions, num_angleActions)
    '''
    num_actions = 4 # 0-go , 1-turn left , 2-turn right , 3-slow
    #num_actions = env.action_space.n
    print(" #actions dimension: ", num_actions)

    dqn = DQNAgent(args, num_actions)
    if args.train:
        print(">> Training mode.")
        dqn.fit(env, args.num_samples, args.max_episode_length)
    else:
        print(">> Evaluation mode.")
        dqn.evaluate(env, args.num_episodes_at_test, 0, args.max_episode_length, not args.no_monitor)

if __name__ == '__main__':
    main()
