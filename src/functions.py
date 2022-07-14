import os
import numpy as np
import tensorflow as tf

def set_seed(seed):
    """
    Sets a global random seed of your choice
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)