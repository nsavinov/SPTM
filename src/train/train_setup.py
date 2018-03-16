import sys
sys.path.append('..')
from common import *

# limit memory usage
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = TRAIN_MEMORY_FRACTION
set_session(tf.Session(config=config))
