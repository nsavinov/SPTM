from test_navigation_quantitative import *

def main_visualize_shortcuts(environment):
  keyframes, keyframe_coordinates, _ = main_exploration(None, environment)
  memory = SPTM()
  memory.set_shortcuts_cache_file(environment)
  memory.compute_shortcuts(keyframes, keyframe_coordinates)
  trajectory_plotter = TrajectoryPlotter(os.path.join(EVALUATION_PATH, environment + '_graph.pdf'), *TEST_SETUPS[environment].box)
  for point in keyframe_coordinates:
    trajectory_plotter.add_point((point[0], point[1]))
  for index in xrange(memory.get_number_of_shortcuts()):
    first, second = memory.get_shortcut(index)
    assert abs(first - second) > MIN_SHORTCUT_DISTANCE
    trajectory_plotter.add_edge((keyframe_coordinates[first][0],
                                 keyframe_coordinates[second][0],
                                 keyframe_coordinates[first][1],
                                 keyframe_coordinates[second][1]))
  trajectory_plotter.save()

if __name__ == '__main__':
  main_visualize_shortcuts(sys.argv[1])
