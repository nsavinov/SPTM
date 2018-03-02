DATA_PATH = '../../data/'

class TestSetup:
  def __init__(self,
               dir,
               wad,
               memory_buffer_lmp,
               goal_lmps,
               maps,
               exploration_map,
               goal_locations,
               goal_names,
               box):
    self.wad = DATA_PATH + dir + wad
    self.memory_buffer_lmp = DATA_PATH + dir + memory_buffer_lmp
    self.goal_lmps = [DATA_PATH + dir + value for value in goal_lmps]
    self.maps = maps
    self.exploration_map = exploration_map
    self.goal_locations = goal_locations
    self.goal_names = goal_names
    self.box = box
TEST_SETUPS = {}
STANDARD_MAPS = ['map02', 'map03', 'map04', 'map05']
EXPLORATION_MAP = 'map06'
STANDARD_GOAL_NAMES = ['tall_red_pillar',
                       'candelabra',
                       'tall_blue_torch',
                       'short_green_pillar']
TEST_SETUPS['deepmind_small'] = \
    TestSetup(
        dir='Test/deepmind_small/',
        wad='deepmind_small.wad_manymaps_test.wad',
        memory_buffer_lmp='deepmind_small.lmp',
        goal_lmps=['deepmind_small_tall_red_pillar.lmp',
                   'deepmind_small_candelabra.lmp',
                   'deepmind_small_tall_blue_torch.lmp',
                   'deepmind_small_short_green_pillar.lmp'],
        maps=STANDARD_MAPS,
        exploration_map=EXPLORATION_MAP,
        goal_locations=[(-64.0, -192.0),
                        (64.0, 64.0),
                        (320.0, -64.0),
                        (192.0, 64.0)],
        goal_names=STANDARD_GOAL_NAMES,
        box=[-512.0, -384.0, 768.0, 256.0])
TEST_SETUPS['deepmind_small_dm'] = \
    TestSetup(
        dir='Test/deepmind_small_dm/',
        wad='deepmind_small.wad_manymaps_test.wad',
        memory_buffer_lmp='deepmind_small.lmp',
        goal_lmps=['deepmind_small_tall_red_pillar.lmp',
                   'deepmind_small_candelabra.lmp',
                   'deepmind_small_tall_blue_torch.lmp',
                   'deepmind_small_short_green_pillar.lmp'],
        maps=STANDARD_MAPS,
        exploration_map=EXPLORATION_MAP,
        goal_locations=[(-64.0, -192.0),
                        (64.0, 64.0),
                        (320.0, -64.0),
                        (192.0, 64.0)],
        goal_names=STANDARD_GOAL_NAMES,
        box=[-512.0, -384.0, 768.0, 256.0])
TEST_SETUPS['deepmind_small_autoexplore'] = \
    TestSetup(
        dir='Test/deepmind_small_autoexplore/',
        wad='deepmind_small.wad_manymaps_test.wad',
        memory_buffer_lmp='deepmind_small.lmp',
        goal_lmps=['deepmind_small_tall_red_pillar.lmp',
                   'deepmind_small_candelabra.lmp',
                   'deepmind_small_tall_blue_torch.lmp',
                   'deepmind_small_short_green_pillar.lmp'],
        maps=STANDARD_MAPS,
        exploration_map=EXPLORATION_MAP,
        goal_locations=[(-64.0, -192.0),
                        (64.0, 64.0),
                        (320.0, -64.0),
                        (192.0, 64.0)],
        goal_names=STANDARD_GOAL_NAMES,
        box=[-512.0, -384.0, 768.0, 256.0])
TEST_SETUPS['open_space_five'] = \
    TestSetup(
        dir='Test/open_space_five/',
        wad='open_space_five.wad_manymaps_test.wad',
        memory_buffer_lmp='open_space_five.lmp',
        goal_lmps=['open_space_five_tall_red_pillar.lmp',
                   'open_space_five_candelabra.lmp',
                   'open_space_five_tall_blue_torch.lmp',
                   'open_space_five_short_green_pillar.lmp'],
        maps=STANDARD_MAPS,
        exploration_map=EXPLORATION_MAP,
        goal_locations=[(1728.0, 896.0),
                        (832.0, 1728.0),
                        (832.0, 128.0),
                        (1728.0, 1152.0)],
        goal_names=STANDARD_GOAL_NAMES,
        box=[0.0, 0.0, 1856.0, 1856.0])
TEST_SETUPS['star_maze'] = \
    TestSetup(
        dir='Test/star_maze/',
        wad='star_maze.wad_manymaps_test.wad',
        memory_buffer_lmp='star_maze.lmp',
        goal_lmps=['star_maze_tall_red_pillar.lmp',
                   'star_maze_candelabra.lmp',
                   'star_maze_tall_blue_torch.lmp',
                   'star_maze_short_green_pillar.lmp'],
        maps=STANDARD_MAPS,
        exploration_map=EXPLORATION_MAP,
        goal_locations=[(-448.0, -992.0),
                        (-704.0, -320.0),
                        (736.0, -320.0),
                        (544.0, 768.0)],
        goal_names=STANDARD_GOAL_NAMES,
        box=[-928.0, -1088.0, 1472.0, 864.0])
TEST_SETUPS['office1'] = \
    TestSetup(
        dir='Test/office1/',
        wad='office1.wad_manymaps_test.wad',
        memory_buffer_lmp='office1.lmp',
        goal_lmps=['office1_tall_red_pillar.lmp',
                   'office1_candelabra.lmp',
                   'office1_tall_blue_torch.lmp',
                   'office1_short_green_pillar.lmp'],
        maps=STANDARD_MAPS,
        exploration_map=EXPLORATION_MAP,
        goal_locations=[(320.0, 192.0),
                        (192.0, 192.0),
                        (960.0, -64.0),
                        (832.0, -576.0)],
        goal_names=STANDARD_GOAL_NAMES,
        box=[-384.0, -640.0, 1280.0, 256.0])
TEST_SETUPS['office1_dm'] = \
    TestSetup(
        dir='Test/office1_dm/',
        wad='office1.wad_manymaps_test.wad',
        memory_buffer_lmp='office1.lmp',
        goal_lmps=['office1_tall_red_pillar.lmp',
                   'office1_candelabra.lmp',
                   'office1_tall_blue_torch.lmp',
                   'office1_short_green_pillar.lmp'],
        maps=STANDARD_MAPS,
        exploration_map=EXPLORATION_MAP,
        goal_locations=[(320.0, 192.0),
                        (192.0, 192.0),
                        (960.0, -64.0),
                        (832.0, -576.0)],
        goal_names=STANDARD_GOAL_NAMES,
        box=[-384.0, -640.0, 1280.0, 256.0])
TEST_SETUPS['office1_autoexplore'] = \
    TestSetup(
        dir='Test/office1_autoexplore/',
        wad='office1.wad_manymaps_test.wad',
        memory_buffer_lmp='office1.lmp',
        goal_lmps=['office1_tall_red_pillar.lmp',
                   'office1_candelabra.lmp',
                   'office1_tall_blue_torch.lmp',
                   'office1_short_green_pillar.lmp'],
        maps=STANDARD_MAPS,
        exploration_map=EXPLORATION_MAP,
        goal_locations=[(320.0, 192.0),
                        (192.0, 192.0),
                        (960.0, -64.0),
                        (832.0, -576.0)],
        goal_names=STANDARD_GOAL_NAMES,
        box=[-384.0, -640.0, 1280.0, 256.0])
TEST_SETUPS['columns'] = \
    TestSetup(
        dir='Test/columns/',
        wad='columns.wad_manymaps_test.wad',
        memory_buffer_lmp='columns.lmp',
        goal_lmps=['columns_tall_red_pillar.lmp',
                   'columns_candelabra.lmp',
                   'columns_tall_blue_torch.lmp',
                   'columns_short_green_pillar.lmp'],
        maps=STANDARD_MAPS,
        exploration_map=EXPLORATION_MAP,
        goal_locations=[(-672.0, -480.0),
                        (-224.0, 352.0),
                        (256.0, 320.0),
                        (768.0, -448.0)],
        goal_names=STANDARD_GOAL_NAMES,
        box=[-704.0, -512.0, 832.0, 384.0])
TEST_SETUPS['columns_dm'] = \
    TestSetup(
        dir='Test/columns_dm/',
        wad='columns.wad_manymaps_test.wad',
        memory_buffer_lmp='columns.lmp',
        goal_lmps=['columns_tall_red_pillar.lmp',
                   'columns_candelabra.lmp',
                   'columns_tall_blue_torch.lmp',
                   'columns_short_green_pillar.lmp'],
        maps=STANDARD_MAPS,
        exploration_map=EXPLORATION_MAP,
        goal_locations=[(-672.0, -480.0),
                        (-224.0, 352.0),
                        (256.0, 320.0),
                        (768.0, -448.0)],
        goal_names=STANDARD_GOAL_NAMES,
        box=[-704.0, -512.0, 832.0, 384.0])
TEST_SETUPS['columns_autoexplore'] = \
    TestSetup(
        dir='Test/columns_autoexplore/',
        wad='columns.wad_manymaps_test.wad',
        memory_buffer_lmp='columns.lmp',
        goal_lmps=['columns_tall_red_pillar.lmp',
                   'columns_candelabra.lmp',
                   'columns_tall_blue_torch.lmp',
                   'columns_short_green_pillar.lmp'],
        maps=STANDARD_MAPS,
        exploration_map=EXPLORATION_MAP,
        goal_locations=[(-672.0, -480.0),
                        (-224.0, 352.0),
                        (256.0, 320.0),
                        (768.0, -448.0)],
        goal_names=STANDARD_GOAL_NAMES,
        box=[-704.0, -512.0, 832.0, 384.0])
TEST_SETUPS['office2'] = \
    TestSetup(
        dir='Test/office2/',
        wad='office2.wad_manymaps_test.wad',
        memory_buffer_lmp='office2.lmp',
        goal_lmps=['office2_tall_red_pillar.lmp',
                   'office2_candelabra.lmp',
                   'office2_tall_blue_torch.lmp',
                   'office2_short_green_pillar.lmp'],
        maps=STANDARD_MAPS,
        exploration_map=EXPLORATION_MAP,
        goal_locations=[(-384.0, -256.0),
                        (0.0, 0.0),
                        (352.0, -480.0),
                        (768.0, 32.0)],
        goal_names=STANDARD_GOAL_NAMES,
        box=[-576.0, -640.0, 832.0, 320.0])
TEST_SETUPS['topological_star_easier'] = \
    TestSetup(
        dir='Test/topological_star_easier/',
        wad='topological_star_easier.wad_manymaps_test.wad',
        memory_buffer_lmp='topological_star_easier.lmp',
        goal_lmps=['topological_star_easier_tall_red_pillar.lmp',
                   'topological_star_easier_candelabra.lmp',
                   'topological_star_easier_tall_blue_torch.lmp',
                   'topological_star_easier_short_green_pillar.lmp'],
        maps=STANDARD_MAPS,
        exploration_map=EXPLORATION_MAP,
        goal_locations=[(-832.0, -384.0),
                        (-704.0, -128.0),
                        (960.0, -384.0),
                        (960.0, 128.0)],
        goal_names=STANDARD_GOAL_NAMES,
        box=[-896.0, -448.0, 1024.0, 576.0])
TEST_SETUPS['open_space_two'] = \
    TestSetup(
        dir='Val/open_space_two/',
        wad='open_space_two.wad_manymaps_test.wad',
        memory_buffer_lmp='open_space_two.lmp',
        goal_lmps=['open_space_two_tall_red_pillar.lmp',
                   'open_space_two_candelabra.lmp',
                   'open_space_two_tall_blue_torch.lmp',
                   'open_space_two_short_green_pillar.lmp'],
        maps=STANDARD_MAPS,
        exploration_map=EXPLORATION_MAP,
        goal_locations=[(1728.0, 1600.0),
                        (1728.0, 128.0),
                        (128.0, 1728.0),
                        (128.0, 128.0)],
        goal_names=STANDARD_GOAL_NAMES,
        box=[0.0, 0.0, 1856.0, 1856.0])
TEST_SETUPS['branching'] = \
    TestSetup(
        dir='Val/branching/',
        wad='branching.wad_manymaps_test.wad',
        memory_buffer_lmp='branching.lmp',
        goal_lmps=['branching_tall_red_pillar.lmp',
                   'branching_candelabra.lmp',
                   'branching_tall_blue_torch.lmp',
                   'branching_short_green_pillar.lmp'],
        maps=STANDARD_MAPS,
        exploration_map=EXPLORATION_MAP,
        goal_locations=[(192.0, -448.0),
                        (64.0, 320.0),
                        (320.0, -64.0),
                        (448.0, -320.0)],
        goal_names=STANDARD_GOAL_NAMES,
        box=[-256.0, -768.0, 1024.0, 512.0])
TEST_SETUPS['deepmind_large'] = \
    TestSetup(
        dir='Val/deepmind_large/',
        wad='deepmind_large.wad_manymaps_test.wad',
        memory_buffer_lmp='deepmind_large.lmp',
        goal_lmps=['deepmind_large_tall_red_pillar.lmp',
                   'deepmind_large_candelabra.lmp',
                   'deepmind_large_tall_blue_torch.lmp',
                   'deepmind_large_short_green_pillar.lmp'],
        maps=STANDARD_MAPS,
        exploration_map=EXPLORATION_MAP,
        goal_locations=[(576.0, -320.0),
                        (1088.0, -576.0),
                        (320.0, -192.0),
                        (704.0, -832.0)],
        goal_names=STANDARD_GOAL_NAMES,
        box=[-640.0, -1024.0, 1280.0, 128.0])
TEST_SETUPS['deepmind_large_dm'] = \
    TestSetup(
        dir='Val/deepmind_large_dm/',
        wad='deepmind_large.wad_manymaps_test.wad',
        memory_buffer_lmp='deepmind_large.lmp',
        goal_lmps=['deepmind_large_tall_red_pillar.lmp',
                   'deepmind_large_candelabra.lmp',
                   'deepmind_large_tall_blue_torch.lmp',
                   'deepmind_large_short_green_pillar.lmp'],
        maps=STANDARD_MAPS,
        exploration_map=EXPLORATION_MAP,
        goal_locations=[(576.0, -320.0),
                        (1088.0, -576.0),
                        (320.0, -192.0),
                        (704.0, -832.0)],
        goal_names=STANDARD_GOAL_NAMES,
        box=[-640.0, -1024.0, 1280.0, 128.0])

def register_test_setups():
  return TEST_SETUPS
