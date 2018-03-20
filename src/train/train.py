from train_setup import *

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

def action_future_selector(current, actions):
  future = current + random.randint(1, MAX_ACTION_DISTANCE)
  if future >= MAX_CONTINUOUS_PLAY:
    return None, None, None
  return future, actions[current], future + 1

def edge_future_selector(current, actions):
  ahead = current + random.randint(1, MAX_ACTION_DISTANCE)
  if ahead >= MAX_CONTINUOUS_PLAY:
    return None, None, None
  if random.random() < 0.5:
    y = 1
    future = ahead
  else:
    y = 0
    before = None
    after = None
    index_before_max = current - NEGATIVE_SAMPLE_MULTIPLIER * MAX_ACTION_DISTANCE
    index_after_min = current + NEGATIVE_SAMPLE_MULTIPLIER * MAX_ACTION_DISTANCE
    if index_before_max >= 0:
      before = random.randint(0, index_before_max)
    if index_after_min < MAX_CONTINUOUS_PLAY:
      after = random.randint(index_after_min, MAX_CONTINUOUS_PLAY - 1)
    if before is None:
      future = after
    elif after is None:
      future = before
    else:
      if random.random() < 0.5:
        future = before
      else:
        future = after
  return future, y, ahead + 1

def get_state_encoding(frames, current, future, state_encoding_frames):
  future_frame = frames[future]
  current_frame = frames[current]
  if state_encoding_frames == 2:
    if current > 0:
      previous_frame = frames[current - 1]
    else:
      previous_frame = current_frame
    return np.concatenate((previous_frame, current_frame, future_frame), axis=2)
  elif state_encoding_frames == 1:
    return np.concatenate((current_frame, future_frame), axis=2)
  else:
    raise Exception('Not implemented!')

def data_generator(future_selector, state_encoding_frames, num_classes):
  game = doom_navigation_setup(DEFAULT_RANDOM_SEED, TRAIN_WAD)
  while True:
    x_result = []
    y_result = []
    for episode in xrange(NUMBER_OF_EPISODES):
      game.set_doom_map(MAP_NAME_TEMPLATE % random.randint(MIN_RANDOM_TEXTURE_MAP_INDEX,
                                                           MAX_RANDOM_TEXTURE_MAP_INDEX))
      game.new_episode()
      frames = []
      actions = []
      for _ in xrange(MAX_CONTINUOUS_PLAY):
        frame = game.get_state().screen_buffer.transpose(VIZDOOM_TO_TF)
        action_index = random.randint(0, ACTION_CLASSES - 1)
        game_make_action_wrapper(game, ACTIONS_LIST[action_index], TRAIN_REPEAT)
        frames.append(frame)
        actions.append(action_index)
      current = 0
      while True:
        future, y, new_current = future_selector(current, actions)
        if future is None:
          break
        x_result.append(get_state_encoding(frames, current, future, state_encoding_frames))
        y_result.append(y)
        current = new_current
    x_result = np.array(x_result)
    y_result = np.array(y_result)
    perm = np.random.permutation(x_result.shape[0])
    x_result = x_result[perm, ...]
    y_result = keras.utils.to_categorical(y_result[perm, ...], num_classes=num_classes)
    number_of_batches = x_result.shape[0] / BATCH_SIZE
    for batch_index in xrange(number_of_batches):
      from_index = batch_index * BATCH_SIZE
      to_index = (batch_index + 1) * BATCH_SIZE
      yield (x_result[from_index:to_index, ...],
             y_result[from_index:to_index, ...])

def train_main(mode):
  logs_path, current_model_path = setup_training_paths(EXPERIMENT_OUTPUT_FOLDER)
  if mode == 'action':
    num_classes = ACTION_CLASSES
    state_encoding_frames = ACTION_STATE_ENCODING_FRAMES
    network = ACTION_NETWORK
    future_selector = action_future_selector
  elif mode == 'edge':
    num_classes = EDGE_CLASSES
    state_encoding_frames = EDGE_STATE_ENCODING_FRAMES
    network = EDGE_NETWORK
    future_selector = edge_future_selector
  else:
    raise Exception('Unknown mode!')
  model = network(((1 + state_encoding_frames) * NET_CHANNELS, NET_HEIGHT, NET_WIDTH), num_classes)
  adam = keras.optimizers.Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
  model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
  callbacks_list = [keras.callbacks.TensorBoard(log_dir=logs_path, write_graph=False),
                    keras.callbacks.ModelCheckpoint(current_model_path,
                                                    period=MODEL_CHECKPOINT_PERIOD)]
  model.fit_generator(data_generator(future_selector, state_encoding_frames, num_classes),
                      steps_per_epoch=DUMP_AFTER_BATCHES,
                      epochs=MAX_EPOCHS,
                      callbacks=callbacks_list)

if __name__ == '__main__':
  train_main(sys.argv[1])
