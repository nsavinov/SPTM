from train_setup import *

def data_generator():
  game = doom_navigation_setup(DEFAULT_RANDOM_SEED, TRAIN_WAD)
  while True:
    x_result = []
    y_result = []
    for episode in xrange(EDGE_EPISODES):
      game.set_doom_map(MAP_NAME_TEMPLATE % random.randint(MIN_RANDOM_TEXTURE_MAP_INDEX,
                                                           MAX_RANDOM_TEXTURE_MAP_INDEX))
      game.new_episode()
      x = []
      for _ in xrange(MAX_CONTINUOUS_PLAY):
        current_x = game.get_state().screen_buffer.transpose(VIZDOOM_TO_TF)
        action_index = random.randint(0, ACTION_CLASSES - 1)
        game_make_action_wrapper(game, ACTIONS_LIST[action_index], TRAIN_REPEAT)
        x.append(current_x)
      first_second_label = []
      current_first = 0
      while True:
        y = None
        current_second = None
        if random.random() < 0.5:
          y = 1
          second = current_first + random.randint(1, MAX_ACTION_DISTANCE)
          if second >= MAX_CONTINUOUS_PLAY:
            break
          current_second = second
        else:
          y = 0
          second = current_first + random.randint(1, MAX_ACTION_DISTANCE)
          if second >= MAX_CONTINUOUS_PLAY:
            break
          current_second_before = None
          current_second_after = None
          index_before_max = current_first - NEGATIVE_SAMPLE_MULTIPLIER * MAX_ACTION_DISTANCE
          index_after_min = current_first + NEGATIVE_SAMPLE_MULTIPLIER * MAX_ACTION_DISTANCE
          if index_before_max >= 0:
            current_second_before = random.randint(0, index_before_max)
          if index_after_min < MAX_CONTINUOUS_PLAY:
            current_second_after = random.randint(index_after_min, MAX_CONTINUOUS_PLAY - 1)
          if current_second_before is None:
            current_second = current_second_after
          elif current_second_after is None:
            current_second = current_second_before
          else:
            if random.random() < 0.5:
              current_second = current_second_before
            else:
              current_second = current_second_after
        first_second_label.append((current_first, current_second, y))
        current_first = second + 1
      random.shuffle(first_second_label)
      for first, second, y in first_second_label:
        future_x = x[second]
        current_x = x[first]
        current_y = y
        x_result.append(np.concatenate((current_x, future_x), axis=2))
        y_result.append(current_y)
    number_of_batches = len(x_result) / BATCH_SIZE
    for batch_index in xrange(number_of_batches):
      from_index = batch_index * BATCH_SIZE
      to_index = (batch_index + 1) * BATCH_SIZE
      yield (np.array(x_result[from_index:to_index]),
             keras.utils.to_categorical(np.array(y_result[from_index:to_index]),
                                        num_classes=EDGE_CLASSES))

if __name__ == '__main__':
  logs_path, current_model_path = setup_training_paths(EXPERIMENT_OUTPUT_FOLDER)
  model = EDGE_NETWORK(((1 + EDGE_STATE_ENCODING_FRAMES) * NET_CHANNELS, NET_HEIGHT, NET_WIDTH), EDGE_CLASSES)
  adam = keras.optimizers.Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
  model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
  callbacks_list = [keras.callbacks.TensorBoard(log_dir=logs_path, write_graph=False),
                    keras.callbacks.ModelCheckpoint(current_model_path,
                                                    period=MODEL_CHECKPOINT_PERIOD)]
  model.fit_generator(data_generator(),
                      steps_per_epoch=DUMP_AFTER_BATCHES,
                      epochs=EDGE_MAX_EPOCHS,
                      callbacks=callbacks_list)
