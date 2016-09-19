#!/usr/bin/env python

import numpy as np
import _pickle as pickle
import configparser
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
    return I.astype(np.uint8)


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

### hyperparameters
config = configparser.ConfigParser()
config.read('config.properties')
decay_rate = float(config['NEURAL_NETWORK']['decay_rate']) # decay factor for RMSProp leaky sum of grad^2
resume = config['NEURAL_NETWORK']['resume'].lower() == 'true' # resume from previous checkpoint?
params = {}
params["batchsize"] = None if config['NEURAL_NETWORK']['batchsize'].lower() == "none" else int(config['NEURAL_NETWORK']['batchsize'])
params["channels"] = int(config['NEURAL_NETWORK']['channels'])
params["rows"] = int(config['NEURAL_NETWORK']['rows'])
params["cols"] = int(config['NEURAL_NETWORK']['cols'])
params["discount_rewards"] = float(config['NEURAL_NETWORK']['discount_rewards'])
params["filters_cnn_h1"] = int(config['NEURAL_NETWORK']['filters_cnn_h1'])
params["filtersize_cnn_h1"] = int(config['NEURAL_NETWORK']['filtersize_cnn_h1'])
params["stride_cnn_h1"] = int(config['NEURAL_NETWORK']['stride_cnn_h1'])
params["filters_cnn_h2"] = int(config['NEURAL_NETWORK']['filters_cnn_h2'])
params["filtersize_cnn_h2"] = int(config['NEURAL_NETWORK']['filtersize_cnn_h2'])
params["stride_cnn_h2"] = int(config['NEURAL_NETWORK']['stride_cnn_h2'])
params["num_units_cnn_out"] = int(config['NEURAL_NETWORK']['num_units_cnn_out'])
params["filters_cnn_h2"] = int(config['NEURAL_NETWORK']['filters_cnn_h2'])
params["minibatchsize"] = int(config['NEURAL_NETWORK']['minibatchsize'])
params["memory_size"] = int(config['NEURAL_NETWORK']['memory_size'])
params["update_frequency"] = int(config['NEURAL_NETWORK']['update_frequency'])
params["discount_factor"] = float(config['NEURAL_NETWORK']['discount_factor'])
learning_rate = float(config['NEURAL_NETWORK']['learning_rate'])
replay_start_size = float(config['NEURAL_NETWORK']['replay_start_size'])
epsilon_start = float(config['NEURAL_NETWORK']['epsilon_start'])
epsilon_end = float(config['NEURAL_NETWORK']['epsilon_end'])
epsilon_frame_exploration = float(config['NEURAL_NETWORK']['epsilon_frame_exploration'])


for k in params.keys():
    logger.debug(k + " --> " + str(params[k]))

# update buffers that add up gradients over a batch
# grad_buffer = { k : np.zeros_like(v) for k,v in model.items() }
# rmsprop memory
# rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() }

prev_x = None # used in computing the difference frame
xs,hs,dlogps,drhos,drs = [],[],[],[],[]
running_reward = 0
reward_sum = 0
n_episode= 0
n_frames = 0 # number of frames in an episode
n_frames_episode = 0 # number of frames in the current episode

#
env = gym.make("Pong-v0")
observation = env.reset()
action_space = {0:(0,'NOOP'), 1:(2,'UP'), 2:(3,'DOWN')}

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
    convnet = lasagne.layers.DenseLayer(convnet, num_units = params["num_units_cnn_out"],
                                        nonlinearity = lasagne.nonlinearities.rectify,
                                        W=lasagne.init.GlorotUniform(gain="relu"),
                                        b=lasagne.init.Constant(0.))

    return convnet



""" MAIN LOOP """

input_var = T.tensor4("inputs")
input_shape = (-1,1,params["rows"],params["cols"])
keep_going = True
Q = build_convnet(params, input_var)
Qhat = Q
actions_history = np.zeros(shape=params["memory_size"])
rewards_history = np.zeros(shape=params["memory_size"])
state_history = np.zeros(shape=(params["memory_size"], params["rows"]*params["cols"]))
history_counter = 0


while keep_going:
    n_frames = n_frames+1
    n_frames_episode = n_frames_episode + 1
    # preprocess (crop) the observation, set input to network to be difference image
    # reshape convention (examples, channels, rows, columns)
    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(params["rows"]*params["cols"])
    prev_x = cur_x

    # forward pass
    # the Q values of all actions given current state s
    Q_s_a = theano.function([input_var],lasagne.layers.get_output(Q))(x.reshape(input_shape))
    print(Q_s_a)
    # print(Q_s_a[0].argmax())
    Q_s_a_max_idx = np.argwhere(Q_s_a[0] == max(Q_s_a[0])).flatten()
    # print(Q_s_a_max_idx)
    random_action_idx = random.sample(list(Q_s_a_max_idx),1)[0]
    action = action_space[Q_s_a_max_idx[0]] if len(Q_s_a_max_idx) == 1 else action_space[random_action_idx]
    print(action[1])

    # store the current state and action
    state_history[history_counter,:] = x.reshape(params["rows"]*params["cols"])
    actions_history[history_counter] = action[0]

    # step the environment and get new measurements
    observation, reward, done, info = env.step(action[0])

    running_reward = running_reward + reward
    print("Running reward " + str(running_reward))

    rewards_history[history_counter] = reward
    history_counter = (history_counter + 1) % params["memory_size"]

    # an episode finished, i.e. score up to 21 for either player
    if done:
        n_episode = n_episode + 1
        print(history_counter)
        print(n_frames_episode)
        # discount the rewards for the last episode only
        discounted_epr = discount_rewards(rewards_history[(history_counter-n_frames_episode):history_counter],
                                          gamma=params["discount_rewards"])
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        print(discounted_epr)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        # update the history
        rewards_history[(history_counter-n_frames_episode):history_counter] = discounted_epr
        n_frames_episode = 0
        keep_going=False
        observation = env.reset() # reset env

    # perform param update on a random minibatch sampled from the history
    if n_frames > params["memory_size"]:
        # history is full - sample from all of it
        minibatch_indeces = random.sample(range(params["memory_size"]),params["minibatchsize"])
    else:
        # sample up until the recorded history
        minibatch_indeces = random.sample(range(n_frames),params["minibatchsize"])

    # do the parameter updates


    """
     grad that encourages the action that was taken to be taken
     (see http://cs231n.github.io/neural-networks-2/#losses if confused)
    """
    # dlogps.append(y - aprob)

    # step the environment and get new measurements
    # observation, reward, done, info = env.step(action)
    # reward_sum += reward

    # drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)
    # n_frames += 1

    """

        logger.info('Finished episode %d with %d frames.', episode_number, n_frames)
        episode_number += 1
        n_frames = 0
        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epdrho = np.vstack(drhos)
        epr = np.vstack(drs)
        xs,hs,dlogps,drhos,drs = [],[],[],[],[] # reset array memory

        # compute the discounted reward backwards through time
        discounted_epr = discount_rewards(epr)
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
        grad = policy_backward(eph, epdlogp, epdrho)

        grad_buffer['W1'] += grad['W1'] # accumulate grad over batch
        grad_buffer['W2'] += grad['W2'] # accumulate grad over batch

        # perform rmsprop parameter update every batch_size episodes
        if episode_number % batch_size == 0:
            logger.info('Performing param update')
            print(grad_buffer['W1'])
            for k in ['W1','W2']:
                g = grad_buffer[k] # gradient
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
                model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(model[k]) # reset batch gradient buffer

        # boring book-keeping
        if episode_number % 1000 == 0:
            logger.info("Writing model to file, episode %d",episode_number)
            pickle.dump(model, open('save_'+str(episode_number)+'_memory2.p', 'wb'))

        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        logger.info('Resetting env. episode total/mean reward, up frequency and sum(W1): %f \t %f \t %f \t %f',
                    reward_sum, running_reward,model['frequency_up'],np.sum(model['W1']))
        reward_sum = 0
        observation = env.reset() # reset env
        x_t1 = None

        if reward == 1 or reward == -1:
            logger.info('ep %d finished, reward: %f', episode_number, reward)
        else:
            logger.error('ep %d finished, but received invalid reward: %f !!!', episode_number, reward)
    """