import math
import random
from typing import List
from typing import Optional

import numpy as np
import torch


def min_max_scale(arr, min_vals, max_vals):
    return torch.tensor((arr-min_vals)/(max_vals-min_vals), dtype=torch.float32)


def pre_proc(obs: np.ndarray) -> torch.Tensor:
    min_vals = np.array([-4.8, -1000, -0.418, -1000])
    max_vals = -1*min_vals
    with torch.no_grad():
        x = min_max_scale(obs, min_vals, max_vals)
        return x


def e_greedy(steps: int, eps_start: float = 0.9, eps_end: float = 0.05,
             eps_decay: float = 100) -> Optional[List[int]]:
    sample = random.random()
    # eps_threshold = eps_end + (eps_start - eps_end) * \
    #     math.exp(-1. * steps / eps_decay)
    # if sample <= eps_threshold:
    if sample <= 0.2:
        return random.randint(0, 1)
    return None
