import sys
sys.path.append('..')
from common import *
from vizdoom import *
import cv2
import numpy as np
np.random.seed(TEST_RANDOM_SEED)
import keras
import random
random.seed(TEST_RANDOM_SEED)

def test_setup(wad):
  game = doom_navigation_setup(TEST_RANDOM_SEED, wad)
  wait_idle(game, WAIT_BEFORE_START_TICS)
  return game

# limit memory usage
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = TEST_MEMORY_FRACTION
set_session(tf.Session(config=config))
