"""Small utilities used across train / eval entry points."""
import random

import numpy as np
import torch


def set_seed(seed):
    """Seed python / numpy / torch (CPU + CUDA) and force deterministic cuDNN."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cycle(it):
    """Endlessly re-iterate over `it` (`itertools.cycle` materializes; this does not)."""
    while True:
        for i in it:
            yield i
