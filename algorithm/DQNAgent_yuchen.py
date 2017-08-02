#!/usr/bin/env python
# coding=utf-8

''' Tensorflow implemente DQN.

yuchen
30.07.2017, Hamburg
'''
from algorithm.policy import *
from algorithm.objectives import *
from algorithm.preprocessors import *
from algorithm.utils import *
from algorithm.core import *
from helper import *
from QNetwork_yuchen import Qnetwork, save_scalar

import numpy as np
import sys
from gym import wrappers
import tensorflow as tf
print(tf.__version__)
import time

"""Main DQN agent."""
class DQNAgent:
    """Class implementing DQN.
    Parameters
    ----------
    q_network: [keras.models.Model] Your Q-network model.
    preprocessor: [deeprl_hw2.core.Preprocessor]The preprocessor class. See the associated classes for more
      details.
    memory: [deeprl_hw2.core.Memory] Your replay memory.
    gamma: [float] Discount factor.
    target_update_freq: [float] Frequency to update the target network. You can either provide a
      number representing a soft target update (see utils.py) or a
      hard target update (see utils.py and Atari paper.)
    num_burn_in: [int]  Before you begin updating the Q-network, your replay memory has
      to be filled up with some number of samples. This number says
      how many.
    train_freq: [int]  How often you actually update your Q-Network. Sometimes
      stability is improved if you collect a couple samples for your
      replay memory, for every Q-network update that you run.
    batch_size: [int] How many samples in each minibatch.
    """
    def __init__(self, args, num_actions):
        self.num_actions = num_actions
        input_shape = (args.frame_height, args.frame_width, args.num_frames)
        self.history_processor = HistoryPreprocessor(args.num_frames - 1)
        self.atari_processor = AtariPreprocessor()
        self.memory = ReplayMemory(args)
        self.policy = LinearDecayGreedyEpsilonPolicy(args.initial_epsilon, args.final_epsilon, args.exploration_steps)
        self.gamma = args.gamma
        self.target_update_freq = args.target_update_freq
        self.num_burn_in = args.num_burn_in
        self.train_freq = args.train_freq
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.frame_width = args.frame_width
        self.frame_height = args.frame_height
        self.num_frames = args.num_frames
        self.output_path = args.output
        self.output_path_videos = args.output + '/videos/'
        self.output_path_images = args.output + '/images/'
        self.save_freq = args.save_freq
        self.load_network = args.load_network
        self.load_network_path = args.load_network_path
        self.enable_ddqn = args.ddqn
        self.args = args

        self.h_size = 512
        self.tau = 0.001

        tf.reset_default_graph()
        # We define the cells for the primary and target q-networks
        cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.h_size, state_is_tuple=True)
        cellT = tf.contrib.rnn.BasicLSTMCell(num_units=self.h_size, state_is_tuple=True)

        self.q_network = Qnetwork(args, h_size=self.h_size, num_frames=self.num_frames, num_actions=self.num_actions, rnn_cell_1=cell, myScope="QNet")
        self.target_network = Qnetwork(args, h_size=self.h_size, num_frames=self.num_frames, num_actions=self.num_actions, rnn_cell_1=cellT, myScope="TargetNet")

        print(">>>> Net mode: Using double dqn: %s" % ( self.enable_ddqn))
        self.eval_freq = args.eval_freq
        #self.no_experience = args.no_experience
        #self.no_target = args.no_target
        #print(">>>> Target fixing: %s, Experience replay: %s" % (not self.no_target, not self.no_experience))

        # initialize target network
        init = tf.global_variables_initializer()
        self.saver = tf.train.Saver(max_to_keep=2) # the maximum number of recent checkpoint files to keep. As new files are created, older files are deleted
        trainables = tf.trainable_variables()
        print(trainables, len(trainables))
        self.targetOps = updateTargetGraph(trainables, self.tau) #update operation

        config = tf.ConfigProto() # set config for Session
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True # allow TF to find device
        self.sess = tf.Session(config=config)
        self.sess.run(init)
        updateTarget(self.targetOps, self.sess)
        self.writer = tf.summary.FileWriter(self.output_path) # save to tensorboard

    def calc_q_values(self, state):
        """Given a state (or batch of states) calculate the Q-values.
        Basically run your network on these states.
        Return
        ------
        Q-values for the state(s)
        """
        state = state[None, :, :, :]
        # return self.q_network.predict_on_batch(state)
        # print state.shape
        # Qout = self.sess.run(self.q_network.rnn_outputs,\
        #             feed_dict={self.q_network.imageIn: state, self.q_network.batch_size:1})
        # print Qout.shape
        Qout = self.sess.run(self.q_network.Qout,\
                    feed_dict={self.q_network.imageIn: state, self.q_network.batch_size:1})
        # print Qout.shape
        return Qout

    def select_action(self, state, is_training = True, **kwargs):
        """Select the action based on the current state.

        You will probably want to vary your behavior here based on
        which stage of training your in. For example, if you're still
        collecting random samples you might want to use a
        UniformRandomPolicy.

        If you're testing, you might want to use a GreedyEpsilonPolicy
        with a low epsilon.

        If you're training, you might want to use the
        LinearDecayGreedyEpsilonPolicy.

        This would also be a good place to call
        process_state_for_network in your preprocessor.

        Returns
        --------
        selected action
        """
        q_values = self.calc_q_values(state)
        if is_training:
            if kwargs['policy_type'] == 'UniformRandomPolicy':
                return UniformRandomPolicy(self.num_actions).select_action()
            else:
                # linear decay greedy epsilon policy
                return self.policy.select_action(q_values, is_training)
        else:
            # return GreedyEpsilonPolicy(0.05).select_action(q_values)
            return GreedyPolicy().select_action(q_values)

    def update_policy(self, current_sample = None):
        """Update your policy.
        Behavior may differ based on what stage of training your
        in. If you're in training mode then you should check if you
        should update your network parameters based on the current
        step and the value you set for train_freq.

        Inside, you'll want to sample a minibatch, calculate the
        target values, update your network, and then update your
        target values.

        You might want to return the loss and other metrics as an
        output. They can help you monitor how training is going.
        """
        batch_size = self.batch_size
        samples = self.memory.sample(batch_size)
        samples = self.atari_processor.process_batch(samples) #uint8 convert to float32.

        # samples = <state, action, next_state, reward>
        states = np.stack([x.state for x in samples])
        actions = np.asarray([x.action for x in samples])
        # action_mask = np.zeros((batch_size, self.num_actions))
        # action_mask[range(batch_size), actions] = 1.0
        next_states = np.stack([x.next_state for x in samples])
        mask = np.asarray([1 - int(x.is_terminal) for x in samples])
        rewards = np.asarray([x.reward for x in samples])

        # compute prediction of target network condition on next_states
        # next_qa_value = self.target_network.predict_on_batch(next_states)
        next_qa_value = self.sess.run(self.target_network.Qout,\
                feed_dict={self.target_network.imageIn: next_states, self.target_network.batch_size:batch_size})

        if self.enable_ddqn: # enable double DQN.
            qa_value = self.sess.run(self.q_network.Qout,\
                    feed_dict={self.q_network.imageIn: next_states, self.q_network.batch_size:batch_size})
            max_actions = np.argmax(qa_value, axis = 1)
            next_qa_value = next_qa_value[range(batch_size), max_actions]
        else:
            next_qa_value = np.max(next_qa_value, axis = 1)
        # print rewards.shape, mask.shape, next_qa_value.shape, batch_size
        # DQN: y = r + gama * max(Q)
        # double DQN: y = r + gama * Q(s, argmaxQ)
        target = rewards + self.gamma * mask * next_qa_value

        loss, _, rnn = self.sess.run([self.q_network.loss, self.q_network.updateModel, self.q_network.rnn], \
                    feed_dict={self.q_network.imageIn: states, self.q_network.batch_size:batch_size, \
                    self.q_network.actions: actions, self.q_network.targetQ: target})
        # the mean of target on a batch.
        return loss, np.mean(target)

    def fit(self, env, num_iterations, max_episode_length=None):
        """Fit your model to the provided environment (training process.).

        Its a good idea to print out things like loss, average reward,
        Q-values, etc to see if your agent is actually improving.

        You should probably also periodically save your network
        weights and any other useful info.

        This is where you should sample actions from your network,
        collect experience samples and add them to your replay memory,
        and update your network parameters.

        Parameters
        ----------
        env: gym.Env
          This is your Atari environment. You should wrap the
          environment using the wrap_atari_env function in the
          utils.py
        num_iterations: int
          How many samples/updates to perform.
        max_episode_length: int
          How long a single episode should last before the agent
          resets. Can help exploration.
        """
        is_training = True
        print("Training starts.")
        self.save_model(0)
        eval_count = 0

        state = env.reset()
        burn_in = True
        idx_episode = 1
        episode_loss = .0
        episode_frames = 0
        episode_reward = .0
        episode_raw_reward = .0
        episode_target_value = .0
        #for t in range(self.num_burn_in + num_iterations):
        for t in range(5): #for testing
            action_state = self.history_processor.process_state_for_network(
                self.atari_processor.process_state_for_network(state))
            policy_type = "UniformRandomPolicy" if burn_in else "LinearDecayGreedyEpsilonPolicy"
            action = self.select_action(action_state, is_training, policy_type = policy_type)
            processed_state = self.atari_processor.process_state_for_memory(state) #image convert to uint8.

            state, reward, done, info = env.step(action)
            time.sleep(1)
            state = np.copy(np.asarray(state))
            processed_next_state = self.atari_processor.process_state_for_network(state)
            action_next_state = np.dstack((action_state, processed_next_state))
            action_next_state = action_next_state[:, :, 1:] # delect the first state, then from 1 start.
            print('Raw:', 'step =', t, 'action=', action, 'next_state=', tf.shape(processed_next_state))

            processed_reward = self.atari_processor.process_reward(reward)

            self.memory.append(processed_state, action, processed_reward, done)
            #current_sample = Sample(action_state, action, processed_reward, action_next_state, done)

            if not burn_in: #True, here don't run at first.
                episode_frames += 1
                episode_reward += processed_reward
                episode_raw_reward += reward
                if episode_frames > max_episode_length: #max_episode_length=1000
                    done = True

            if done: # run when reaching terminal.
                # adding last frame of this episode only to save last state. and action, reward, done doesn't matter here
                last_frame = self.atari_processor.process_state_for_memory(state)
                self.memory.append(last_frame, action, 0, done)

                if not burn_in: #True, here don't run at first.
                    avg_target_value = episode_target_value / episode_frames
                    print(">>> Training (episode): time_step %d, episode %d, length_frames %d, reward %.0f, raw_reward %.0f, loss %.4f, target value %.4f, policy step %d, memory cap %d" %
                        (t, idx_episode, episode_frames, episode_reward, episode_raw_reward, episode_loss,
                        avg_target_value, self.policy.step, self.memory.current))
                    sys.stdout.flush()
                    save_scalar(idx_episode, 'train/episode_frames', episode_frames, self.writer)
                    save_scalar(idx_episode, 'train/episode_reward', episode_reward, self.writer)
                    save_scalar(idx_episode, 'train/episode_raw_reward', episode_raw_reward, self.writer)
                    save_scalar(idx_episode, 'train/episode_loss', episode_loss, self.writer)
                    save_scalar(idx_episode, 'train_avg/avg_reward', episode_reward / episode_frames, self.writer)
                    save_scalar(idx_episode, 'train_avg/avg_target_value', avg_target_value, self.writer)
                    save_scalar(idx_episode, 'train_avg/avg_loss', episode_loss / episode_frames, self.writer)
                    episode_frames = 0
                    episode_reward = .0
                    episode_raw_reward = .0
                    episode_loss = .0
                    episode_target_value = .0
                    idx_episode += 1
                burn_in = (t < self.num_burn_in) # decide if memory is enough or not.
                state = env.reset()
                self.atari_processor.reset()
                self.history_processor.reset()

            if not burn_in: #True, here don't run at first.
                if t % self.train_freq == 0: # update Q-network , for every 4 episode.
                    #loss, target_value = self.update_policy(current_sample) #when no_experience replay
                    loss, target_value = self.update_policy()
                    episode_loss += loss
                    episode_target_value += target_value

                if t % (self.train_freq * self.target_update_freq) == 0:# update target Q-network, for every 4*100 episode.
                    updateTarget(self.targetOps, self.sess)
                    print("----- Synced.")
                if t % self.save_freq == 0:# save Q-network, for every 100 episode.
                    self.save_model(idx_episode)
                '''
                if t % (self.eval_freq * self.train_freq) == 0: # test Q-network, for every 4*100 episode
                    episode_reward_mean, episode_reward_std, eval_count = self.evaluate(env, 20, eval_count, max_episode_length, True)
                    save_scalar(t, 'eval/eval_episode_reward_mean', episode_reward_mean, self.writer)
                    save_scalar(t, 'eval/eval_episode_reward_std', episode_reward_std, self.writer)
                '''
        self.save_model(idx_episode)
        '''
        file_memory = open('./results/memory_RL.csv', 'w')
        memoryResults = dict(
                        actions = self.memory.actions.tolist(),
                        states = self.memory.screens.tolist(),
                        rewards = self.memory.rewards.tolist(),
                        terminals = self.memory.terminals.tolist() )
        file_memory.write(str(memoryResults))
        '''
        #time.sleep(0.5)
        #file_memory.close()
    #end of fit method.

    def save_model(self, idx_episode):
        safe_path = self.output_path + "/qnet" + str(idx_episode) + ".cptk"
        self.saver.save(self.sess, safe_path)
        # self.q_network.save_weights(safe_path)
        print("+++++++++ Network at", idx_episode, "saved to:", safe_path)

    def restore_model(self, restore_path):
        self.saver.restore(self.sess, restore_path)
        print("+++++++++ Network restored from: %s", restore_path)

    def evaluate(self, env, num_episodes, eval_count, max_episode_length=None, monitor=True):
        """Test your agent with a provided environment.

        You shouldn't update your network parameters here. Also if you
        have any layers that vary in behavior between train/test time
        (such as dropout or batch norm), you should set them to test.

        Basically run your policy on the environment and collect stats
        like cumulative reward, average episode length, etc.

        You can also call the render function here if you want to
        visually inspect your policy.
        """
        print("Evaluation starts.")
        plt.figure(1, figsize=(45, 20))

        is_training = False
        if self.load_network:
            # self.q_network.load_weights(self.load_network_path)
            # print("Load network from:", self.load_network_path)
            self.restore_model(self.load_network_path)
        if monitor:
            env = wrappers.Monitor(env, self.output_path_videos, video_callable=lambda x:True, resume=True)
        state = env.reset()

        idx_episode = 1
        episode_frames = 0
        episode_reward = np.zeros(num_episodes)
        t = 0

        while idx_episode <= num_episodes:
            t += 1
            action_state = self.history_processor.process_state_for_network(
                self.atari_processor.process_state_for_network(state))
            action = self.select_action(action_state, is_training, policy_type = 'GreedyEpsilonPolicy')

            action_state_ori = self.history_processor.process_state_for_network_ori(
                self.atari_processor.process_state_for_network_ori(state))

            dice = np.random.random()

            state, reward, done, info = env.step(action)

            if dice < 0.1:
                attention_a = self.sess.run(self.q_network.attention_a,\
                            feed_dict={self.q_network.imageIn: action_state[None, :, :, :], self.q_network.batch_size:1})
                # print attention_a.shape #(1, 10, 1)
                attention_a = np.reshape(attention_a, (-1))
                for alpha_idx in range(action_state_ori.shape[3]):
                    plt.subplot(2, action_state_ori.shape[3]//2+1, alpha_idx+1)
                    img = action_state_ori[:, :, :, alpha_idx] #(210, 160, 3)
                    plt.imshow(img)
                    # plt.text(0, 1, 'Weight: %.4f'%(att ention_a[alpha_idx]) , color='black', weight='bold', backgroundcolor='white', fontsize=30)
                plt.subplot(2, action_state_ori.shape[3]//2+1, action_state_ori.shape[3]+2)
                plt.imshow(state)
                # plt.text(0, 1, 'Next state after taking the action %s'%(action), color='black', weight='bold', backgroundcolor='white', fontsize=20)
                plt.axis('off')
                plt.savefig('%sattention_ep%d-frame%d.png'%(self.output_path_images, eval_count, episode_frames))
                print('---- Image saved at: %sattention_ep%d-frame%d.png'%(self.output_path_images, eval_count, episode_frames))

            episode_frames += 1
            episode_reward[idx_episode-1] += reward
            if episode_frames > max_episode_length:
                done = True
            if done:
                print("Eval: time %d, episode %d, length %d, reward %.0f. @eval_count %s" %
                    (t, idx_episode, episode_frames, episode_reward[idx_episode-1], eval_count))
                eval_count += 1
                save_scalar(eval_count, 'eval/eval_episode_raw_reward', episode_reward[idx_episode-1], self.writer)
                save_scalar(eval_count, 'eval/eval_episode_raw_length', episode_frames, self.writer)
                sys.stdout.flush()
                state = env.reset()
                episode_frames = 0
                idx_episode += 1
                self.atari_processor.reset()
                self.history_processor.reset()

        reward_mean = np.mean(episode_reward)
        reward_std = np.std(episode_reward)
        print("Evaluation summury: num_episodes [%d], reward_mean [%.3f], reward_std [%.3f]" %
            (num_episodes, reward_mean, reward_std))
        sys.stdout.flush()

        return reward_mean, reward_std, eval_count
