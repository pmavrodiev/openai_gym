#!/usr/bin/env python

""" References
    [1] - Google DeepMind, Human-level control through deep reenforcement learning, 2015
"""
import numpy as np
import _pickle as pickle

import gym
import logging
import logging.handlers
import gzip
import sys
import random
import lasagne
import theano
import theano.tensor as T

from helpers import *



def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    return I.astype(np.uint8).flatten()


""" SET-UP """

# logging
logging_level = logging.DEBUG
flogging_level = logging.INFO

formatter = logging.Formatter(fmt=("%(asctime)s - %(levelname)s - %(module)s - %(message)s"))
shandler = logging.StreamHandler()
shandler.setFormatter(formatter)
shandler.setLevel(logging_level)

fhandler = logging.handlers.RotatingFileHandler('pong.log',encoding='utf-8',maxBytes=1e+8,backupCount=100)
fhandler.setFormatter(formatter)
fhandler.setLevel(flogging_level)

logger = logging.getLogger('PONG')
logger.setLevel(logging_level)
logger.addHandler(shandler)
logger.addHandler(fhandler)

# parse config
params = parse_config('config.properties')

for k in params.keys():
    logger.debug(k + " --> " + str(params[k]))


""" BUILD THE CONV NET https://github.com/Lasagne/Lasagne"""

def build_convnet(params, input_var=None, ):
    # example input: shape=(None, 1, 80, 80)
    convnet = lasagne.layers.InputLayer(shape=(params["batchsize"],params["channels"],params["rows"],params["cols"]),
                                     input_var=input_var)
    # 1st convolution layer.
    # Example params for input size 80x80:  32 filters, (4x4) filter size, stride = 4, no padding
    # will result in 20x20 output - (80 - 4) / 4 + 1 = 20 - for each of the 32 filters
    convnet = lasagne.layers.Conv2DLayer(convnet, num_filters = params["filters_cnn_h1"],
                                         filter_size = params["filtersize_cnn_h1"],
                                         stride = params["stride_cnn_h1"],
                                         nonlinearity=lasagne.nonlinearities.rectify,
                                         W=lasagne.init.GlorotUniform(gain="relu"),
                                         b=lasagne.init.Constant(0.))
    # 2nd convolution layer
    # Example params for input size 20x20x32: 64 filters, (2x2) filter size, stride = 2, no padding
    # will result in 10x10 - (20 - 2) / 2 + 1 - for each of the 64 filters
    # hence final output is 10x10x64
    convnet = lasagne.layers.Conv2DLayer(convnet, num_filters = params["filters_cnn_h2"],
                                         filter_size = params["filtersize_cnn_h2"],
                                         stride = params["stride_cnn_h2"],
                                         nonlinearity=lasagne.nonlinearities.rectify,
                                         W=lasagne.init.GlorotUniform(gain="relu"),
                                         b=lasagne.init.Constant(0.))
    # The final output layer
    # Example params: 3 output units for each of the 3 possible actions - UP, DOWN, NOOP
    # The convention is the same as action_space, i.e.:
    # : output[0] = NOOP, output[1] = UP, output[2] = DOWN
    convnet = lasagne.layers.DenseLayer(convnet, num_units = params["num_units_cnn_out"],
                                        nonlinearity = lasagne.nonlinearities.linear,
                                        W=lasagne.init.GlorotUniform(gain="relu"),
                                        b=lasagne.init.Constant(0.))

    return convnet


def select_action(q_vals_actions, T=1):
    """
        Select an action according to the Boltzmann probability distribution
    """
    total_mass = np.sum(np.exp(q_vals_actions / T))
    probs = np.exp(q_vals_actions / T ) / total_mass
    return np.random.choice(range(len(q_vals_actions)), p = probs)


prev_x = None # used in computing the difference frame

running_reward = 0
reward_sum = 0
n_episode= 0
# total number of frames to train
n_total_frames = 10**6
# number of total frames elapsed since the beginning of the game
n_frames = 0
# number of frames in the current episode
n_frames_episode = 0
#
input_var = T.tensor4("inputs")
target_var = T.tensor3("target_var")
targets = np.zeros(shape=(params["minibatchsize"],params["num_units_cnn_out"]))
prediction = T.tensor3("target_var")
input_shape = (-1,1,params["rows"],params["cols"])
Q = build_convnet(params, input_var)
#
action_space = {0:(0,'NOOP'), 1:(2,'UP'), 2:(3,'DOWN')}
history_counter = 0
actions_history = np.zeros(shape=params["memory_size"])
rewards_history = np.zeros(shape=params["memory_size"])
# state_history[m,0] and state_history[m,1] - x_j and x_{j+1} recorded at history position m
state_history = np.zeros(shape=(params["memory_size"], 2, params["rows"]*params["cols"]))
q_history = np.zeros(shape=(params["memory_size"], len(action_space)))

prediction =  lasagne.layers.get_output(Q)
loss = lasagne.objectives.squared_error(prediction, target_var)
loss = lasagne.objectives.aggregate(loss, mode="sum")
#loss = loss.mean() # take the mean over the minibatch
parameters = lasagne.layers.get_all_params(Q, trainable=True)
updates = lasagne.updates.rmsprop(loss, parameters, learning_rate=params['learning_rate'],rho=params["decay_rate"])
#
train_fn = theano.function([input_var, target_var], loss, updates=updates)
train_error = 0
#
env = gym.make("Pong-v0")
observation = env.reset()

""" MAIN LOOP """
running_reward = 0
while n_frames < n_total_frames:
    # preprocess (crop) the observation, set input to network to be difference image
    # reshape convention (examples, channels, rows, columns)
    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(params["rows"]*params["cols"])
    prev_x = cur_x

    # forward pass
    # get the Q values of all actions given current state x
    Q_x_a = theano.function([input_var],lasagne.layers.get_output(Q))(x.reshape(input_shape))

    action = select_action(Q_x_a[0], T = 1)
    # map the action, which is simply the index of the selected Q output,
    # to the action space of the simulator
    sim_action = action_space[action][0]

    # step the environment and get new measurements
    observation, reward, done, info = env.step(sim_action)

    # update the q values, state and actions history
    q_history[history_counter,:] = Q_x_a[0]
    actions_history[history_counter] = action

    state_history[history_counter,0] = x.reshape(params["rows"]*params["cols"])
    # store the next state in the history
    state_history[history_counter,1] = prepro(observation) - cur_x # next - current = next difference frame

    running_reward += reward

    rewards_history[history_counter] = reward

    n_frames += 1
    n_frames_episode += 1

    logger.info("[n_frame %d, episode %d, n_frames_episode %d]: action %s, reward %d",
                n_frames, n_episode, n_frames_episode, action_space[action][1], reward)


    logger.debug("history_counter %d, memory_size %d", history_counter, params["memory_size"])

    # an episode finished, i.e. score up to 21 for either player
    if done:
        logger.info("\nepisode %d finished, cumulative reward %d", n_episode, running_reward)
        logger.debug("[n_frame %d, n_frames_episode %d, history_counter %d]",
                    n_frames, n_frames_episode, history_counter)
        n_episode = n_episode + 1
        """
        the last action occured at index history_counter, update its Q-value to either -1 or 1
        the Q values of all other actions taken in previous steps will be similarly fixed to
        the discounted rewards in discounted_epr.
        The Q values of non-taken actions in all previous steps remain as they are

        e.g. last_action = NOOP =>  q_history = [NOOP, UP, DOWN] = [FIX, Qhat, Qhat]
        q_historyp[[row_indeces],[column_indeces]]

        """
        running_reward = 0
        n_frames_episode = 0
        observation = env.reset() # reset env
        prev_x = None


    # UPDATE
    if  n_frames>32 and n_episode % params["update_frequency"] == 0 :

        # perform param update on a random minibatch sampled from the history
        # Algorithm 1 of [1]
        if n_frames > params["memory_size"]:
            # history is full - sample from all of it
            minibatch_indeces = random.sample(range(params["memory_size"]),params["minibatchsize"])
        else:
            # sample up until the recorded history
            minibatch_indeces = random.sample(range(n_frames),params["minibatchsize"])

        # update q_history and generate targets
        for j in range(len(minibatch_indeces)):
            val = minibatch_indeces[j]
            # inner loop of Algorithm 1 of [1]
            targets[j,:] = q_history[val,:]
            if rewards_history[val] == -1 or rewards_history[val] == 1:
                # pong specific rewards at the end of an episode
                targets[j, actions_history[action]] = rewards_history[val]
            else:
                x_history_j = state_history[val,1].reshape(input_shape)
                Q_xj_a = theano.function([input_var],lasagne.layers.get_output(Q))(x_history_j)
                targets[j,:] = targets[j,:] + params["discount_rewards"]*Q_xj_a[0]


        # sample inputs and targets
        inputs = state_history[minibatch_indeces,0].reshape(input_shape)
        # make targets a 3D tensor
        targets = targets.reshape(-1,1,params["num_units_cnn_out"])

        train_error += train_fn(inputs, targets)
        logger.info("Training error %f", train_error)
    # end update
    history_counter = (history_counter + 1) % params["memory_size"]
