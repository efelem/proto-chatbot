import random
from typing import Union

import numpy as np
import torch


def clean_text(inp):
    if inp is None:
        return None
    inp = inp.replace("<0x0A>", "\n")
    inp = inp.replace("</s>", "")
    inp = inp.replace("</s-", "")
    return inp


def clean(history):
    return [[clean_text(q), clean_text(a)] for q, a in history]

def seed_everything(seed: Union[int, None] = None):
    """Set the random seed for reproducible results.

    Set seed for multiple libraries including 'random', 'numpy',
    and 'torch' to ensure. If no seed is provided, a random integer
    between 0 and 1,000,000,000 will be used.

    Args:
        seed (Union[int, None], optional): The seed value to set. Defaults to None.

    Example:
        seed_everything(42)  # Set the seed to 42 for reproducibility
    """

    if seed is None:
        seed = random.randint(0, 1000000000)
    print(f"Setting seed to {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
