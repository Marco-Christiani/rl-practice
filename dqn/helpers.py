import math
import random
from typing import List
from typing import Optional

import numpy as np
import torch

min_vals = np.array([-4.8, -1000, -0.418, -1000])
max_vals = -1*min_vals


def pre_proc(obs: np.ndarray) -> torch.Tensor:
    # with torch.no_grad():
    # return torch.Tensor((obs-min_vals)/(max_vals-min_vals))
    return torch.Tensor(obs)


def e_greedy(episode: int = 1, epsilon: int = 0.2) -> Optional[List[int]]:
    sample = random.random()
    # if sample <= epsilon/episode:
    #     return random.randint(0, 1)
    if episode > 5e4:
        epsilon /= 2
    if episode > 4e5:
        epsilon = 0
    if sample <= epsilon:
        return random.randint(0, 1)
    return None
