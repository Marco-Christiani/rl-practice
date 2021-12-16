import math
import random

import numpy as np
import torch


def min_max_scale(arr, min_vals, max_vals):
    return torch.tensor((arr-min_vals)/(max_vals-min_vals), dtype=torch.float32)


def pre_proc(obs):
    min_vals = np.array([-4.8, -1000, -0.418, -1000])
    max_vals = -1*min_vals
    with torch.no_grad():
        x = min_max_scale(obs, min_vals, max_vals)
        return x


def e_greedy(steps, eps_start=0.9, eps_end=0.05, eps_decay=200):
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * \
        math.exp(-1. * steps / eps_decay)
    if sample <= eps_threshold:
        return random.randrange(2)
    return None
