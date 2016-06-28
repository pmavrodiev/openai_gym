import numpy as np
import _pickle as pickle
import configparser
import gym
import logging
import logging.handlers
import gzip


""" AUXILLIARY FUNCTIONS """
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

def dsigmoid(x):
    """ this is the derivative of sigmoid """
    ex = np.exp(x)
    return ex / ((ex + 1.0)**2.0)

def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    return I

""" POLICY GRADIENTS """

def policy_forward(x):
    h = np.dot(model['W1'], x)
    h[h <= 0] = 0 # ReLU nonlinearity no inhibitory neurons
    rho = np.dot(model['W2'], h)
    p = sigmoid(rho)
    return rho, p, h # return probability of taking action 2, and hidden state

def policy_backward(eph, epdlogp, epdrho):
    """ backward pass. (eph is array of intermediate hidden states) """
    dW2 = np.dot(eph.T, epdlogp*epdrho).ravel()
    dh = np.outer(epdlogp*epdrho, model['W2'])
    dh[eph <= 0] = 0 # backpro prelu. if the neuron was inactive, do not change its weight(?)
    dW1 = np.dot(dh.T, epx) - alpha*np.sum(model['W1'])
    return {'W1':dW1, 'W2':dW2}

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0:
            running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

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
logger.setLevel(flogging_level)
logger.addHandler(shandler)
logger.addHandler(fhandler)

### hyperparameters
config = configparser.ConfigParser()
config.read('config.properties')
H = int(config['NEURAL_NETWORK']['number_neurons']) # number of hidden layer neurons
batch_size = int(config['NEURAL_NETWORK']['batch_size']) # every how many episodes (games) to do a param update?
learning_rate = float(config['NEURAL_NETWORK']['learning_rate'])
gamma = float(config['NEURAL_NETWORK']['gamma']) # discount factor for reward
alpha = float(config['NEURAL_NETWORK']['alpha']) # L2 regularization
decay_rate = float(config['NEURAL_NETWORK']['decay_rate']) # decay factor for RMSProp leaky sum of grad^2
resume = config['NEURAL_NETWORK']['resume'].lower() == 'true' # resume from previous checkpoint?
render = config['NEURAL_NETWORK']['render'].lower() == 'true'
# number of pixels
D = int(config['NEURAL_NETWORK']['x']) # input dimensionality: 80x80 grid

# log the configs
logger.info('H:%d,D:%d,batch_size:%d,learning_rate:%f,gamma:%f,decay_rate:%f,resume:%s,render:%s',
           H,D,batch_size,learning_rate,gamma,decay_rate,resume,render)

if resume:
    logger.info("Resuming from a checkpoint")
    model = pickle.load(open('save.p', 'rb'))
else:
    model = {}
    model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
    model['W2'] = np.random.randn(H) / np.sqrt(H)
    model['frequency_up'] = 0

# update buffers that add up gradients over a batch
grad_buffer = { k : np.zeros_like(v) for k,v in model.items() }
# rmsprop memory
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() }

prev_x = None # used in computing the difference frame
xs,hs,dlogps,drhos,drs = [],[],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 0
n_frames = 0 # number of frames in an episode
n_actions = 0 # total number of actions taken for the entire learning process

#
env = gym.make("Pong-v0")
observation = env.reset()
action_space = {'UP': 2, 'DOWN': 3}

""" MAIN LOOP """

keep_going = True
while keep_going:
    if render:
        env.render()

    # preprocess (crop) the observation, set input to network to be difference image
    cur_x = prepro(observation).astype(np.float).ravel()
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x

    # forward the policy network and sample an action from the returned probability
    # aprob is the probability of going UP
    rho, aprob, h = policy_forward(x)
    # UP = 2, DOWN = 3
    action = action_space['UP'] if np.random.uniform() < aprob else action_space['DOWN'] # roll the dice!

    # record various intermediates (needed later for backprop)
    xs.append(x) # observation
    hs.append(h) # hidden state
    drhos.append(dsigmoid(rho)) # the first derivative of the sigmoid activation function at rho
    y = 1 if action == action_space['UP'] else 0 # take an action
    n_actions += 1
    model['frequency_up'] = (y + (n_actions-1)*model['frequency_up']) / n_actions

    """
     grad that encourages the action that was taken to be taken
     (see http://cs231n.github.io/neural-networks-2/#losses if confused)
    """
    dlogps.append(y - aprob)

    # step the environment and get new measurements
    observation, reward, done, info = env.step(action)
    reward_sum += reward

    drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)
    n_frames += 1

    if done: # an episode finished, i.e. score up to 21 for either player
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
            for k in ['W1','W2']:
                g = grad_buffer[k] # gradient
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
                model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(model[k]) # reset batch gradient buffer

        # boring book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        logger.info('Resetting env. episode total/mean reward and up frequency were: %f \t %f \t %f',
                    reward_sum, running_reward,model['frequency_up'])
        if episode_number % 1000 == 0:
            pickle.dump(model, open('save_'+str(episode_number)+'.p', 'wb'))
        reward_sum = 0
        observation = env.reset() # reset env
        prev_x = None

        if reward == 1 or reward == -1:
            logger.info('ep %d finished, reward: %f', episode_number, reward)
        else:
            logger.error('ep %d finished, but received invalid reward: %f !!!', episode_number, reward)
