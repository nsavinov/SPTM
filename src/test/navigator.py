from sptm import *

def check_if_close(first_point, second_point):
  if ((first_point[0] - second_point[0]) ** 2 +
      (first_point[1] - second_point[1]) ** 2 <= GOAL_DISTANCE_ALLOWANCE ** 2):
    return True
  else:
    return False

class Navigator:
  def __init__(self, exploration_model_directory):
    self.exploration_model_directory = exploration_model_directory
    self.action_model = load_keras_model(3, len(ACTIONS_LIST), ACTION_MODEL_WEIGHTS_PATH)
    self.memory = SPTM()
    self.trial_index = -1
    print 'Navigator ready!'

  def get_screens_and_coordinates(self):
    return self.screens, self.coordinates

  def setup_video(self, movie_path):
    self.record_video = True
    self.navigation_video_writer = NavigationVideoWriter(os.path.join(EVALUATION_PATH, movie_path),
                                                         nonstop=True)

  def setup_trajectories(self, trajectories_name, box):
    self.record_trajectories = True
    self.trajectory_plotter = TrajectoryPlotter(os.path.join(EVALUATION_PATH, trajectories_name), *box)

  def common_setup(self, game):
    self.game = game
    self.screens = []
    self.coordinates = []
    self.steps = 0
    self.just_started = True
    self.termination = False
    self.record_video = False
    self.record_trajectories = False

  def setup_exploration(self, step_budget, game, environment, box):
    self.looking_for_goal = False
    self.step_budget = step_budget
    self.common_setup(game)
    # self.setup_video('movie.mov')
    self.setup_trajectories(environment + '_exploration.pdf', box)
    self.goal_frame = None

  def setup_navigation_test(self, step_budget, game, goal_location, keyframes, keyframe_coordinates, keyframe_actions, goal_frame, movie_path, box, environment):
    self.trial_index += 1
    self.looking_for_goal = True
    self.step_budget = step_budget
    self.common_setup(game)
    self.goal_location = goal_location
    self.setup_video(movie_path)
    self.setup_trajectories(movie_path + '.pdf', box)
    self.keyframes = keyframes[:]
    self.keyframe_coordinates = keyframe_coordinates[:]
    self.keyframe_actions = keyframe_actions[:]
    self.goal_frame = goal_frame
    goal_localization_keyframe_index = self.process_memory(environment)
    self.not_localized_count = 0
    return goal_localization_keyframe_index

  def record_all(self, first_frame, second_frame, x, y):
    if self.record_video:
      self.navigation_video_writer.write(first_frame, second_frame, 1, 1)
    if self.record_trajectories:
      self.trajectory_plotter.add_point([x, y])

  def process_memory(self, environment):
    self.memory.plot_shortest_path = False #make True for showing the shortest path from the starting position
    self.memory.environment = environment
    self.memory.trial_index = self.trial_index
    self.memory.set_shortcuts_cache_file(environment)
    self.memory.build_graph(self.keyframes, self.keyframe_coordinates)
    best_index, best_probability = self.memory.set_goal(self.goal_frame, self.goal_location, self.keyframe_coordinates)
    print 'Goal localization confidence:', best_probability
    self.keyframes.append(self.goal_frame)
    self.keyframe_coordinates.append(self.goal_location) #NOTE: these are not the exact goal frame coordinates, but close
    self.memory.compute_shortest_paths(len(self.keyframes) - 1)
    return best_index

  def save_recordings(self):
    if self.record_video:
      self.navigation_video_writer.close()
    if self.record_trajectories:
      self.trajectory_plotter.save()

  def get_steps(self):
    return self.steps

  def set_intermediate_reachable_goal(self):
    current_screen = self.game.get_state().screen_buffer.transpose(VIZDOOM_TO_TF)
    self.target_index, self.nn = self.memory.find_intermediate_reachable_goal(current_screen, self.game.get_state().game_variables, self.keyframe_coordinates)
    if self.target_index is None:
      self.target_index = len(self.keyframes) - 1
      self.not_localized_count = 1
    else:
      self.not_localized_count = 0

  def record_all_during_repeat(self, right_image):
    for index in xrange(-TEST_REPEAT, 0):
      left_image = self.screens[index]
      if right_image is None:
        right_image = left_image
      self.record_all(left_image,
                      right_image,
                      self.coordinates[index][0],
                      self.coordinates[index][1])

  def random_explore_step(self):
    if self.check_frozen_with_repeat():
      return
    self._random_explore_step_with_repeat()
    self.record_all_during_repeat(self.keyframes[-1])

  def policy_explore_step(self, walkthrough=False):
    if self.check_frozen_with_repeat():
      return
    self._policy_explore_step_with_repeat()
    target_frame = None
    if not walkthrough:
      target_frame = self.keyframes[-1]
    self.record_all_during_repeat(target_frame)

  def policy_navigation_step(self, teach_and_repeat=False):
    self.set_intermediate_reachable_goal()
    if self.not_localized_count == 0:
      action_function = self._align_step_with_repeat
      action_function_arguments = (teach_and_repeat,)
      number_of_actions = DEEP_NET_ACTIONS
    else:
      action_function = self._policy_explore_step_with_repeat
      action_function_arguments = ()
      number_of_actions = 5
    for counter in xrange(number_of_actions):
      if self.check_frozen_with_repeat():
        break
      action_function(*action_function_arguments)
      self.record_all_during_repeat(self.keyframes[self.target_index])

  def log_navigation_state(self):
    self.screens.append(self.game.get_state().screen_buffer.transpose(VIZDOOM_TO_TF))
    self.coordinates.append(self.game.get_state().game_variables)

  def _align_step_with_repeat(self, teach_and_repeat):
    self.log_navigation_state()
    if self.just_started:
      for _ in xrange(TEST_REPEAT):
        self.screens.append(self.screens[-1])
        self.coordinates.append(self.coordinates[-1])
      self.just_started = False
    if not teach_and_repeat:
      first_arg = self.screens[-1 - TEST_REPEAT]
      second_arg = self.screens[-1]
      third_arg = self.keyframes[self.target_index]
      if HIGH_RESOLUTION_VIDEO:
        first_arg = double_downsampling(first_arg)
        second_arg = double_downsampling(second_arg)
        third_arg = double_downsampling(third_arg)
      x = np.expand_dims(np.concatenate((first_arg,
                                         second_arg,
                                         third_arg), axis=2), axis=0)
      action_probabilities = np.squeeze(self.action_model.predict(x,
                                                                  batch_size=1))
      action_index = np.random.choice(len(ACTIONS_LIST), p=action_probabilities)
      action = ACTIONS_LIST[action_index]
    else:
      if np.random.rand() < (1.0 - TEACH_AND_REPEAT_RANDOMIZATION):
        if self.target_index > self.nn and self.target_index < len(self.keyframe_actions):
          action = self.keyframe_actions[self.nn]
        elif self.target_index < self.nn:
          action = inverse_action(self.keyframe_actions[self.nn - 1])
        else:
          action_index = np.random.choice(len(ACTIONS_LIST))
          action = ACTIONS_LIST[action_index]
      else:
        action_index = np.random.choice(len(ACTIONS_LIST))
        action = ACTIONS_LIST[action_index]
    for repeat_index in xrange(TEST_REPEAT):
      if repeat_index > 0:
        self.log_navigation_state()
      game_make_action_wrapper(self.game, action, 1)
      self.steps += 1

  def _random_explore_step_with_repeat(self):
    action_index = random.randint(0, len(ACTIONS_LIST) - 1)
    self.game.set_action(ACTIONS_LIST[action_index])
    for repeat_index in xrange(TEST_REPEAT):
      self.log_navigation_state()
      self.game.advance_action(1, True)
      self.steps += 1

  def _policy_explore_step_with_repeat(self):
    self._random_explore_step_with_repeat()

  def check_frozen_with_repeat(self):
    if (self.check_goal_reached() or
        self.steps + TEST_REPEAT > self.step_budget):
      self.termination = True
      return True
    else:
      return False

  def check_termination(self):
    return self.termination

  def check_goal_reached(self):
    if self.looking_for_goal:
      current_coordinates = self.game.get_state().game_variables
      return check_if_close(current_coordinates, self.goal_location)
    else:
      return False
