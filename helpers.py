import numpy as np

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

