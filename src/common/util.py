#!/usr/bin/env python
import cPickle
import cv2
import numpy as np
import h5py
from vizdoom import *
import math
import os
import os.path
import sys
import random
import scipy.misc

from constants import *
from video_writer import *

import cv2
import os
import cPickle
import numpy as np
np.random.seed(DEFAULT_RANDOM_SEED)
import keras
import random
random.seed(DEFAULT_RANDOM_SEED)

def mean(numbers):
  return float(sum(numbers)) / max(len(numbers), 1)

def wait_idle(game, wait_idle_tics=WAIT_IDLE_TICS):
  if wait_idle_tics > 0:
    game.make_action(STAY_IDLE, wait_idle_tics)

def game_make_action_wrapper(game, action, repeat):
  game.make_action(action, repeat)
  wait_idle(game)
  return None

def save_list_of_arrays_to_hdf5(input, prefix):
  stacked = np.array(input)
  h5f = h5py.File(prefix + HDF5_NAME, 'w')
  h5f.create_dataset('dataset', data=stacked)
  h5f.close()

def load_array_from_hdf5(prefix):
  h5f = h5py.File(prefix + HDF5_NAME,'r')
  data = h5f['dataset'][:]
  h5f.close()
  return data

class StateRecorder():
  def __init__(self, game):
    self.game = game
    self.game_variables = []
    self.actions = []
    self.rewards = []
    self.screen_buffers = []

  def record_buffers(self, state):
    self.screen_buffers.append(state.screen_buffer.transpose(VIZDOOM_TO_TF))

  '''records current state, then makes the provided action'''
  def record(self, action_index, repeat):
    state = self.game.get_state()
    self.record_buffers(state)
    self.game_variables.append(state.game_variables)
    r = game_make_action_wrapper(self.game, ACTIONS_LIST[action_index], repeat)
    self.actions.append(action_index)
    self.rewards.append(r)

  def save_recorded_buffers(self):
    save_list_of_arrays_to_hdf5(self.screen_buffers, SCREEN_BUFFERS_PATH)

  def save_recorded(self):
    self.save_recorded_buffers()
    data = (self.game_variables,
            self.actions,
            self.rewards)
    with open(NAVIGATION_RECORDING_PATH, 'wb') as output_file:  
      cPickle.dump(data, output_file)

def downsample(input, factor):
  for _ in xrange(factor):
    input = cv2.pyrDown(input)
  return input

def double_downsampling(input):
  return cv2.pyrDown(cv2.pyrDown(input))

def double_upsampling(input):
  return cv2.pyrUp(cv2.pyrUp(input))

def color2gray(input):
  return cv2.cvtColor(input, cv2.COLOR_RGB2GRAY)

def doom_navigation_setup(seed, wad):
  game = DoomGame()
  game.load_config(DEFAULT_CONFIG)
  game.set_doom_scenario_path(wad)
  game.set_seed(seed)
  game.init()
  return game

def calculate_distance_angle(start_coordinates, current_coordinates):
  distance = math.sqrt((start_coordinates[0] - current_coordinates[0]) ** 2 +
                       (start_coordinates[1] - current_coordinates[1]) ** 2 + 
                       (start_coordinates[2] - current_coordinates[2]) ** 2)
  abs_angle_difference = math.fabs(start_coordinates[3] - current_coordinates[3])
  angle = min(abs_angle_difference, 360.0 - abs_angle_difference)
  return distance, angle

def generator(x, y, batch_size, max_action_distance):
  while True:
    number_of_samples = x.shape[0]
    x_list = []
    y_list = []
    for index in xrange(batch_size):
      choice = random.randint(0, number_of_samples - max_action_distance - 1)
      distance = random.randint(1, max_action_distance)
      current_x = x[choice]
      current_y = y[choice]
      future_x = x[choice + distance]
      x_list.append(np.concatenate((current_x, future_x), axis=2))
      y_list.append(current_y)
    yield np.array(x_list), np.array(y_list)

def vertically_stack_image_list(input_image_list):
  image_list = []
  for image in input_image_list:
    image_list.append(image)
    image_list.append(np.zeros([SHOW_BORDER, image.shape[1], SHOW_CHANNELS], dtype=np.uint8))
  return np.concatenate(image_list, axis=0)

def save_np_array_as_png(input, path):
  scipy.misc.toimage(input, cmin=0.0, cmax=255.0).save(path)

class NavigationVideoWriter():
  def __init__(self, save_path, nonstop=False):
    self.nonstop = nonstop
    self.video_writer = VideoWriter(save_path,
                                    (2 * SHOW_WIDTH + SHOW_BORDER, SHOW_HEIGHT),
                                    mode='replace',
                                    framerate=FPS)

  def side_by_side(self, first, second):
    if not HIGH_RESOLUTION_VIDEO:
      first = double_upsampling(first)
      second = double_upsampling(second)
    return np.concatenate((first,
                           np.zeros([SHOW_HEIGHT, SHOW_BORDER, SHOW_CHANNELS], dtype=np.uint8),
                           second), axis=1)

  def write(self, left, right, counter, deep_net_actions):
    side_by_side_screen = self.side_by_side(left, right)
    if not self.nonstop:
      if counter == 0:
        for _ in xrange(START_PAUSE_FRAMES):
          self.video_writer.add_frame(side_by_side_screen)
      elif counter + 1 < deep_net_actions:
        self.video_writer.add_frame(side_by_side_screen)
      else:
        for _ in xrange(END_PAUSE_FRAMES):
          self.video_writer.add_frame(side_by_side_screen)
        for _ in xrange(DELIMITER_FRAMES):
          self.video_writer.add_frame(np.zeros_like(side_by_side_screen))
    else:
      self.video_writer.add_frame(side_by_side_screen)

  def close(self):
    self.video_writer.close()

def make_deep_action(current_screen, goal_screen, model, game, repeat, randomized):
  x = np.expand_dims(np.concatenate((current_screen,
                                     goal_screen), axis=2), axis=0)
  action_probabilities = np.squeeze(model.predict(x,
                                                  batch_size=1))
  action_index = None
  if randomized:
    action_index = np.random.choice(len(ACTIONS_LIST), p=action_probabilities)
  else:
    action_index = np.argmax(action_probabilities)
  game_make_action_wrapper(game, ACTIONS_LIST[action_index], repeat)
  return action_index, action_probabilities, current_screen

def current_make_deep_action(goal_screen, model, game, repeat, randomized):
  state = game.get_state()
  current_screen = state.screen_buffer.transpose(VIZDOOM_TO_TF)
  return make_deep_action(current_screen, goal_screen, model, game, repeat, randomized)

def get_deep_prediction(current_screen, goal_screen, model):
  x = np.expand_dims(np.concatenate((current_screen,
                                     goal_screen), axis=2), axis=0)
  return np.squeeze(model.predict(x, batch_size=1))

def current_get_deep_prediction(goal_screen, model, game):
  state = game.get_state()
  current_screen = state.screen_buffer.transpose(VIZDOOM_TO_TF)
  return get_deep_prediction(current_screen, goal_screen, model)

def explore(game, number_of_actions):
  is_left = random.random() > 0.5
  start_moving_straight = random.randint(0, number_of_actions)
  for counter in xrange(number_of_actions):
    if counter >= start_moving_straight:
      action_index = INVERSE_ACTION_NAMES_INDEX['MOVE_FORWARD']
    else:
      if is_left:
        action_index = INVERSE_ACTION_NAMES_INDEX['TURN_LEFT']
      else:
        action_index = INVERSE_ACTION_NAMES_INDEX['TURN_RIGHT']
    game_make_action_wrapper(game, ACTIONS_LIST[action_index], TEST_REPEAT)

def get_distance(first_point, second_point):
  return math.sqrt((first_point[0] - second_point[0]) ** 2 +
                   (first_point[1] - second_point[1]) ** 2)

