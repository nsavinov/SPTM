import sys

from navigator import *

def load_memory_buffer_from_lmp(wad, lmp, skip_factor, max_frames):
  game = test_setup(wad)
  game.replay_episode(lmp)
  keyframes = []
  keyframe_coordinates = []
  keyframe_actions = []
  counter = 0
  if WRITE_EXPLORATION_VIDEO:
    exploration_video_writer = VideoWriter('exploration_video.mov',
                                           (SHOW_WIDTH, SHOW_HEIGHT),
                                           mode='replace',
                                           framerate=FPS)
  while not game.is_episode_finished():
    state = game.get_state()
    current_frame = state.screen_buffer.transpose(VIZDOOM_TO_TF)
    if WRITE_EXPLORATION_VIDEO:
      exploration_video_writer.add_frame(current_frame)
    if counter % skip_factor == 0:
      keyframes.append(current_frame)
      keyframe_coordinates.append(state.game_variables)
    game.advance_action()
    if counter % skip_factor == 0:
      keyframe_actions.append(game.get_last_action())
      # print keyframe_actions[-1]
    counter += 1
    if max_frames is not None:
      if counter >= max_frames:
        break
  if WRITE_EXPLORATION_VIDEO:
    exploration_video_writer.close()
  return keyframes, keyframe_coordinates, keyframe_actions

def load_goal_frame_from_lmp(wad, lmp):
  keyframes, keyframe_coordinates, _ = load_memory_buffer_from_lmp(wad, lmp, 1, None)
  return keyframes[-1], keyframe_coordinates[-1]

def main_exploration(navigator, environment):
  test_wad = TEST_SETUPS[environment].wad
  game = test_setup(test_wad)
  memory_buffer_wad = test_wad
  memory_buffer_lmp = TEST_SETUPS[environment].memory_buffer_lmp
  if not os.path.isfile(memory_buffer_lmp):
    game.set_doom_map(TEST_SETUPS[environment].exploration_map)
    game.new_episode(memory_buffer_lmp)
    max_number_of_steps = MAX_NUMBER_OF_STEPS_EXPLORATION
    navigator.setup_exploration(max_number_of_steps, game, environment, TEST_SETUPS[environment].box)
    while not navigator.check_termination():
      print 'completed:', 100 * float(navigator.get_steps()) / float(max_number_of_steps), '%'
      navigator.policy_explore_step(walkthrough=True)
    navigator.save_recordings()
    game.new_episode() # to make it record the data right here
  keyframes, keyframe_coordinates, keyframe_actions = load_memory_buffer_from_lmp(memory_buffer_wad, memory_buffer_lmp, MEMORY_SUBSAMPLING, MEMORY_MAX_FRAMES)
  return keyframes, keyframe_coordinates, keyframe_actions

def main_navigation(navigator, environment, mode, keyframes, keyframe_coordinates, keyframe_actions):
  print CURRENT_NAVIGATION_ENVIRONMENT, environment
  print CURRENT_NAVIGATION_MODE, mode
  test_wad = TEST_SETUPS[environment].wad
  game = test_setup(test_wad)
  results = []
  maps = TEST_SETUPS[environment].maps
  goal_locations = TEST_SETUPS[environment].goal_locations
  goal_lmps = TEST_SETUPS[environment].goal_lmps
  number_of_maps = len(maps)
  number_of_goals = len(goal_lmps)
  goal_frames = []
  for goal_index, goal_lmp in enumerate(goal_lmps):
    goal_frame, goal_frame_coordinates = load_goal_frame_from_lmp(test_wad, goal_lmp)
    if not check_if_close(goal_locations[goal_index],
                          goal_frame_coordinates):
      raise Exception('Goal incorrectly specified in lmp!')
    goal_frames.append(goal_frame)
  for trial_index in xrange(NUMBER_OF_TRIALS):
    print 'Trial index:', trial_index
    for map_index, map_name in enumerate(maps):
      print 'Map name:', map_name
      for goal_index, goal_frame in enumerate(goal_frames):
        goal_name = TEST_SETUPS[environment].goal_names[goal_index]
        print 'Goal name:', goal_name
        goal_location = goal_locations[goal_index]
        game.set_doom_map(map_name)
        movie_filename = '%s_%s_%s_%d_%d_%s.mov' % (environment, mode, navigator.exploration_model_directory, trial_index, map_index, goal_name)
        lmp_save_path = os.path.join(EVALUATION_PATH, movie_filename + '.lmp') 
        # print lmp_save_path
        game.new_episode() #lmp_save_path
        max_number_of_steps = MAX_NUMBER_OF_STEPS_NAVIGATION
        goal_localization_keyframe_index = navigator.setup_navigation_test(max_number_of_steps, game, goal_location, keyframes, keyframe_coordinates, keyframe_actions, goal_frame, movie_filename, TEST_SETUPS[environment].box, environment)
        goal_localization_distance = get_distance(goal_location,
                                                  keyframe_coordinates[goal_localization_keyframe_index])
        print 'Localization distance:', goal_localization_distance
        if mode == 'explore':
          navigator.show_memory_to_exploration_policy(keyframes)
        while not navigator.check_termination():
          print 'completed:', 100 * float(navigator.get_steps()) / float(max_number_of_steps), '%'
          if mode == 'policy':
            navigator.policy_navigation_step()
          elif mode == 'random':
            navigator.random_explore_step()
          elif mode == 'explore':
            navigator.policy_explore_step()
          elif mode == 'teach_and_repeat':
            navigator.policy_navigation_step(teach_and_repeat=True)
          else:
            raise Exception('Please provide the mode: policy or random or explore!')
        results.append((navigator.check_goal_reached(),
                        navigator.get_steps(),
                        goal_localization_distance))
        print results[-1]
        navigator.save_recordings()
  game.new_episode()
  print FINAL_RESULTS, results
  number_of_successes = sum([first for first, _, _ in results])
  print 'Average success:', float(number_of_successes) / float(len(results))
  print 'Average success path length:', float(sum([first * second for first, second, _ in results])) / float(max(1, number_of_successes))
  print 'Average goal localization distance:', float(sum(third for _, _, third in results)) / float(len(results))

if __name__ == '__main__':
  environment, mode = sys.argv[1], sys.argv[2]
  if len(sys.argv) == 4:
    exploration_model_directory = sys.argv[3]
  else:
    exploration_model_directory = 'none'
  print EXPLORATION_MODEL_DIRECTORY, exploration_model_directory
  navigator = Navigator(exploration_model_directory)
  print 'Starting exploration!'
  keyframes, keyframe_coordinates, keyframe_actions = main_exploration(navigator, environment)
  print 'Memory size:', len(keyframes)
  print 'Exploration finished!'
  print 'Starting navigation!'
  main_navigation(navigator, environment, mode, keyframes, keyframe_coordinates, keyframe_actions)
  print 'Navigation finished!'
