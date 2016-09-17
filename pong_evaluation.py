#!/usr/bin/env python

import numpy as np
import _pickle as pickle
import configparser
import gym
import gzip
import sys


""" AUXILLIARY FUNCTIONS """
# sigmoid "squashing" function to interval [0,1]
def sigmoid(x):
    with np.errstate(all='raise'):
        try:
            rv = 1.0 / (1.0 + np.exp(-x))
        except FloatingPointError:
            pickle.dump(model, open('save_err.p', 'wb'))
            logger.error('sigmoid(x): Floating point error because of %f',x)
            sys.exit(1)
    return rv

def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()

""" POLICY GRADIENTS """

def policy_forward(x):
    h = np.dot(model['W1'], x)
    h[h <= 0] = 0 # ReLU nonlinearity no inhibitory neurons
    rho = np.dot(model['W2'], h)
    p = sigmoid(rho)
    return rho, p, h # return probability of taking action 2, and hidden state

""" SET-UP """

### hyperparameters
config = configparser.ConfigParser()
config.read('config.properties')
H = int(config['NEURAL_NETWORK']['number_neurons']) # number of hidden layer neurons
# number of pixels
D = int(config['NEURAL_NETWORK']['x']) # input dimensionality: 80x80 grid


#
model = pickle.load(open('save_47000.p','rb'))
prev_x = None # used in computing the difference frame
#
neuron = 0
log_file = "neuron-" + str(neuron) + ".log"
f = open(log_file,'a')

env = gym.make("Pong-v0")
env.monitor.start('./eval/pong-L2-5')
done=False
action_space = {'UP': 2, 'DOWN': 3}
next=False

for i_episode in range(100):
    observation = env.reset()
    while not next:
        env.render()
        cur_x = prepro(observation)
        x = cur_x - prev_x if prev_x is not None else np.zeros(D)
        prev_x = cur_x
        #
        # Agent plays
        rho, aprob, h = policy_forward(x)
        f.write(str(h[neuron])+"\n")
        # UP = 2, DOWN = 3
        action = action_space['UP'] if np.random.uniform() < aprob else action_space['DOWN'] # roll the dice!
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished")
            next=True

env.monitor.close()
f.close()
