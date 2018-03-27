import omg
import sys
import random
import os
import argparse

FLOOR_TEXTURE = 'ZZWOLF1'
CEILING_TEXTURE = 'ZIMMER8'
DM_UNIFORM_TEXTURE = 'WOOD10'
TRAIN_TEXTURES_PATH = '../../data/textures/train_textures.txt'
TEST_TEXTURES_PATH = '../../data/textures/test_textures.txt'
RANDOM_SEED = random.randint(100, 10000)
GOAL_TEXTURES = ['ZZWOLF11', 'ZZWOLF5', 'ZZWOLF9', 'TEKWALL5']
LANDMARK_TEXTURES = ['REDWALL1']
NUMBER_OF_IDENTICAL_TEST_MAPS = 5
NUMBER_OF_TRAIN_MAPS = 400
TEXTURES_PATH = TEST_TEXTURES_PATH
SUFFIX = '_manymaps_test.wad'
DM_TRAIN_LANDMARK_SPARSITY = 0.15

class DmTextureGenerator:
  def __init__(self, textures):
    self.textures = textures
    self.index = 0

  def next(self):
    if self.index == 0:
      random.shuffle(self.textures)
    returned_value = self.textures[self.index]
    self.index = (self.index + 1) % len(self.textures)
    return returned_value

# note: landmark does not influence this generator
class RandomTextureGenerator:
  def __init__(self, textures, unchanged, landmark):
    self.textures = textures
    self.unchanged = unchanged

  def next(self, current_value):
    if current_value not in self.unchanged:
      return random.choice(self.textures)
    else:
      return current_value

class DmTextureGeneratorWrapper:
  def __init__(self, textures, unchanged, landmark):
    self.dm_texture_generator = DmTextureGenerator(textures)
    self.unchanged = unchanged
    self.landmark = landmark

  def next(self, current_value):
    if current_value not in self.unchanged:
      if not self.landmark:
        if random.random() < DM_TRAIN_LANDMARK_SPARSITY:
          return self.dm_texture_generator.next()
        else:
          return DM_UNIFORM_TEXTURE
      else:
        if current_value in self.landmark:
          return self.dm_texture_generator.next()
        else:
          return DM_UNIFORM_TEXTURE
    else:
      return current_value

def get_textures(textures_file, additional_textures_list):
  with open(textures_file) as f:
    textures = f.read().split()
    textures.extend(additional_textures_list)
  return textures

def copy_attributes(in_map, out_map):
  to_copy = ['BEHAVIOR']
  for t in to_copy:
    if t in in_map:
      out_map[t] = in_map[t]

def set_floor_ceiling(map_editor):
  for s in map_editor.sectors:
    s.tx_floor = FLOOR_TEXTURE
    s.tx_ceil = CEILING_TEXTURE

def set_walls(map_editor, texture_generator):
  for s in map_editor.sidedefs:
    s.tx_mid = texture_generator.next(s.tx_mid)

def inner_change_textures(map_editor, texture_generator):
  set_walls(map_editor, texture_generator)
  set_floor_ceiling(map_editor)

def change_textures(in_map, texture_generator):
  map_editor = omg.MapEditor(in_map)
  inner_change_textures(map_editor, texture_generator)
  out_map = map_editor.to_lumps()
  copy_attributes(in_map, out_map)
  return out_map

def set_seed(mode):
  if mode == 'test':
    random.seed(RANDOM_SEED)

def add_map_to_wad(wad, map, index):
  wad.maps['MAP%.2d' % (index + 2)] = map

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--input', help='input wad file to process')
  parser.add_argument('--texture-sparsity', choices=['sparse', 'dense'], help='texture randomization mode - sparse or dense')
  parser.add_argument('--mode', choices=['train', 'test'], help='what those maps are going to be used for: train or test?')
  args = parser.parse_args()
  in_file = args.input
  out_file = in_file + SUFFIX
  wad = omg.WAD(in_file)
  if args.texture_sparsity == 'sparse':
    generator_class = DmTextureGeneratorWrapper
  else:
    generator_class = RandomTextureGenerator
  if args.mode == 'test':
    textures = get_textures(TEST_TEXTURES_PATH, [])
    number_of_maps = NUMBER_OF_IDENTICAL_TEST_MAPS
    generator_arguments = (textures, GOAL_TEXTURES, LANDMARK_TEXTURES) 
  else:
    textures = get_textures(TEST_TEXTURES_PATH, GOAL_TEXTURES)
    number_of_maps = NUMBER_OF_TRAIN_MAPS
    generator_arguments = (textures, [], []) 
  texture_generator = generator_class(*generator_arguments)
  for index in xrange(number_of_maps):
    print index
    set_seed(args.mode)
    add_map_to_wad(wad, change_textures(wad.maps['MAP01'], texture_generator), index)
  wad.to_file(out_file)

if __name__ == '__main__':
  main()
  