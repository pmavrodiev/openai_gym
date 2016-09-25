import numpy as np
import configparser

""" AUXILLIARY FUNCTIONS """

def discount_rewards(r, gamma):
    """ take 1D float array of rewards and compute discounted reward """
    # sanity checks
    if r[0] != 0.0 or (r[r.size-1] != 1.0 and r[r.size-1] != -1.0):
        raise ValueError("Bad indexing for discounting rewards - wrong array boundaries, %f and %f",
                         r[0],r[r.size-1])

    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0:
            running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def parse_config(file):
    config = configparser.ConfigParser()
    config.read(file)
    params = {}
    # decay factor for RMSProp
    params["decay_rate"] = float(config['NEURAL_NETWORK']['decay_rate'])
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
    # the number of frames stored as memory size.
    # alternatively could implement memory as all frames for a given number of completed episodes
    params["memory_size"] = int(config['NEURAL_NETWORK']['memory_size'])
    # update_frequency - every update_frequency episodes update the Q network
    params["update_frequency"] = int(config['NEURAL_NETWORK']['update_frequency'])
    params["learning_rate"] = float(config['NEURAL_NETWORK']['learning_rate'])
    params["replay_start_size"] = float(config['NEURAL_NETWORK']['replay_start_size'])
    params["epsilon_start"] = float(config['NEURAL_NETWORK']['epsilon_start'])
    params["epsilon_end"] = float(config['NEURAL_NETWORK']['epsilon_end'])
    params["epsilon_frame_exploration"] = float(config['NEURAL_NETWORK']['epsilon_frame_exploration'])

    return params