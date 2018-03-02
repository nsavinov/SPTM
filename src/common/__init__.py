from constants import *
from resnet import *
from util import *

# limit memory usage
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
#CHANGED Alexey changed to 0.15 to fit 5 runs per gpu
#CHANGED 0.2 for training
config.gpu_options.per_process_gpu_memory_fraction = 0.15  #0.15 #0.2 #0.3 #0.8 #0.4 #0.3 #0.2
set_session(tf.Session(config=config))
