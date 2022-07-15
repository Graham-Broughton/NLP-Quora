import os
import numpy as np
import tensorflow as tf
import pickle

def set_seed(seed):
    """
    Sets a global random seed of your choice
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def get_pickle(feature):
    return pickle.load(open(f'drive/MyDrive/Quora_Duplicate_Questions/src/{feature}.pkl', 'rb'))