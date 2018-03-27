from train_setup import *

def data_generator():
  game = doom_navigation_setup(DEFAULT_RANDOM_SEED, TRAIN_WAD)
  game.set_doom_map(MAP_NAME_TEMPLATE % random.randint(MIN_RANDOM_TEXTURE_MAP_INDEX,
                                                       MAX_RANDOM_TEXTURE_MAP_INDEX))
  game.new_episode()
  yield_count = 0
  while True:
    if yield_count >= ACTION_MAX_YIELD_COUNT_BEFORE_RESTART:
      game.set_doom_map(MAP_NAME_TEMPLATE % random.randint(MIN_RANDOM_TEXTURE_MAP_INDEX,
                                                           MAX_RANDOM_TEXTURE_MAP_INDEX))
      game.new_episode()
      yield_count = 0
    x = []
    y = []
    for _ in xrange(MAX_CONTINUOUS_PLAY):
      current_x = game.get_state().screen_buffer.transpose(VIZDOOM_TO_TF)
      action_index = random.randint(0, ACTION_CLASSES - 1)
      game_make_action_wrapper(game, ACTIONS_LIST[action_index], TRAIN_REPEAT)
      current_y = action_index
      x.append(current_x)
      y.append(current_y)
    first_second_pairs = []
    current_first = 0
    while True:
      distance = random.randint(1, MAX_ACTION_DISTANCE)
      second = current_first + distance
      if second >= MAX_CONTINUOUS_PLAY:
        break
      first_second_pairs.append((current_first, second))
      current_first = second + 1
    random.shuffle(first_second_pairs)
    x_result = []
    y_result = []
    for first, second in first_second_pairs:
      future_x = x[second]
      current_x = x[first]
      previous_x = current_x
      if first > 0:
        previous_x = x[first - 1]
      current_y = y[first]
      x_result.append(np.concatenate((previous_x, current_x, future_x), axis=2))
      y_result.append(current_y)
      if len(x_result) == BATCH_SIZE:
        yield_count += 1
        yield (np.array(x_result),
               keras.utils.to_categorical(np.array(y_result),
                                          num_classes=ACTION_CLASSES))
        x_result = []
        y_result = []

if __name__ == '__main__':
  logs_path, current_model_path = setup_training_paths(EXPERIMENT_OUTPUT_FOLDER)
  model = ACTION_NETWORK(((1 + ACTION_STATE_ENCODING_FRAMES) * NET_CHANNELS, NET_HEIGHT, NET_WIDTH), ACTION_CLASSES)
  adam = keras.optimizers.Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
  model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
  callbacks_list = [keras.callbacks.TensorBoard(log_dir=logs_path, write_graph=False),
                    keras.callbacks.ModelCheckpoint(current_model_path,
                                                    period=MODEL_CHECKPOINT_PERIOD)]
  model.fit_generator(data_generator(),
                      steps_per_epoch=DUMP_AFTER_BATCHES,
                      epochs=ACTION_MAX_EPOCHS,
                      callbacks=callbacks_list)
