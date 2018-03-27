import sys
sys.path.append('..')
from common import *

# limit memory usage
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = TRAIN_MEMORY_FRACTION
set_session(tf.Session(config=config))

def setup_training_paths(experiment_id):
  experiment_path = EXPERIMENTS_PATH_TEMPLATE % experiment_id
  logs_path = LOGS_PATH_TEMPLATE % experiment_id
  models_path = MODELS_PATH_TEMPLATE % experiment_id
  current_model_path = CURRENT_MODEL_PATH_TEMPLATE % experiment_id
  assert (not os.path.exists(experiment_path)), 'Experiment folder %s already exists' % experiment_path
  os.makedirs(experiment_path)
  os.makedirs(logs_path)
  os.makedirs(models_path)
  return logs_path, current_model_path
