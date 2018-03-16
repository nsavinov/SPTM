from train_setup import *

if __name__ == '__main__':
  if sys.argv[1] == 'action':
    model = keras.models.load_model(ACTION_MODEL_PATH)
    model.save_weights(ACTION_MODEL_WEIGHTS_PATH)
  elif sys.argv[1] == 'edge':
    model = keras.models.load_model(EDGE_MODEL_PATH)
    model.save_weights(EDGE_MODEL_WEIGHTS_PATH)
  else:
    raise Exception('Unknown resave mode!')
