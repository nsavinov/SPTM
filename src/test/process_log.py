import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.append('../common')
sys.path.append('../../common')
sys.path.append('../../src/common')
from constants import *

SUCCESS_SCALING = 100.0
EXPLORE_WHEN_LOST = 'none'
METHOD_NAMES = [
('policy_' + EXPLORE_WHEN_LOST),
('teach_and_repeat_' + EXPLORE_WHEN_LOST)
]
METHOD_TO_LEGEND = {
  METHOD_NAMES[0] : 'Ours',
  METHOD_NAMES[1] : 'Teach-repeat'
}
METHOD_TO_COLOR = {
  METHOD_NAMES[0] : 'r',
  METHOD_NAMES[1] : 'y'
}
ENVIRONMENT_TO_PAPER_TITLE = {
  'deepmind_small' : 'Test-1',
  'deepmind_small_dm' : 'Test-1-DM',
  'deepmind_small_autoexplore': 'Test-1-Autoexplore',
  'open_space_five' : 'Test-2',
  'star_maze' : 'Test-3',
  'office1' : 'Test-4',
  'office1_autoexplore' : 'Test-4-Autoexplore',
  'office1_dm' : 'Test-4-DM',
  'columns' : 'Test-5',
  'columns_autoexplore' : 'Test-5-Autoexplore',
  'columns_dm' : 'Test-5-DM',
  'office2' : 'Test-6',
  'topological_star_easier' : 'Test-7',
  'open_space_two' : 'Val-1',
  'branching' : 'Val-2',
  'deepmind_large' : 'Val-3',
  'deepmind_large_dm' : 'Val-3-DM'
}
FONT_SCALAR = 1.7
MEDIUM_FONT = 20 / FONT_SCALAR
LARGE_FONT = 28 / FONT_SCALAR
HUGE_FONT = 30 / FONT_SCALAR
TITLE_FONT = 2 * LARGE_FONT
LEGEND_FONT = int(1.5 * MEDIUM_FONT)
AXIS_LABEL_FONT = int(1.5 * LARGE_FONT)
LINEWIDTH = 2
LEGEND_LINE_WIDTH = LINEWIDTH
plt.rc('font', size=HUGE_FONT)
plt.rc('text', usetex=False)
plt.rc('xtick', labelsize=HUGE_FONT)
plt.rc('ytick', labelsize=HUGE_FONT)
plt.rc('axes', labelsize=HUGE_FONT)

def check_append(prefix, line, output_list):
  line = line.strip()
  if line.startswith(prefix):
    output_list.append(line[len(prefix) + 1:])

def add_to_plots(plots, input):
  FAIL_STEPS = MAX_NUMBER_OF_STEPS_NAVIGATION + 1
  environment, mode, result = input
  steps = []
  success_rate = float(sum([value for value, _, _ in result])) / float(len(result))
  print environment, mode, success_rate
  for success, length, _ in result:
    if success:
      steps.append(length)
    else:
      steps.append(FAIL_STEPS)
  steps.sort()
  cumulative = {}
  for index, step in enumerate(steps):
    if step < FAIL_STEPS:
      cumulative[step] = float(index + 1) / float(len(steps))
    else:
      cumulative[step] = success_rate
  if environment in plots:
    figure, axes = plots[environment]
    plt.sca(axes)
  else:
    figure, axes = plt.subplots()
    plots[environment] = figure, axes
  sorted_cumulative = sorted(cumulative.items())
  # print sorted_cumulative
  x = [0] + [value for value, _ in sorted_cumulative] + [FAIL_STEPS]
  y = [0] + [value for _, value in sorted_cumulative] + [success_rate]
  y = [SUCCESS_SCALING * value for value in y]
  plt.plot(x, y, METHOD_TO_COLOR[mode], linewidth=LINEWIDTH, label=METHOD_TO_LEGEND[mode])
  plt.title(ENVIRONMENT_TO_PAPER_TITLE[environment], fontsize=TITLE_FONT)
  plt.xlabel('Steps', fontsize=AXIS_LABEL_FONT)
  if ENVIRONMENT_TO_PAPER_TITLE[environment] in ['Test-1', 'Test-5', 'Val-1']:
    plt.ylabel('Success rate', fontsize=AXIS_LABEL_FONT)
  plt.axis([0, FAIL_STEPS, 0, 1.0 * SUCCESS_SCALING])
  plt.grid(linestyle='dotted')
  print ENVIRONMENT_TO_PAPER_TITLE[environment]
  if ENVIRONMENT_TO_PAPER_TITLE[environment] in ['Val-3']:
    leg = plt.legend(shadow=True, fontsize=LEGEND_FONT, loc='upper left', fancybox=True, framealpha=1.0)
    for legobj in leg.legendHandles:
      legobj.set_linewidth(LEGEND_LINE_WIDTH)

def main_process_log(log_file):
  environments = []
  modes = []
  results = []
  exploration_models = []
  with open(log_file) as input:
    for line in input:
      check_append(EXPLORATION_MODEL_DIRECTORY, line, exploration_models)
      check_append(CURRENT_NAVIGATION_ENVIRONMENT, line, environments)
      check_append(CURRENT_NAVIGATION_MODE, line, modes)
      check_append(FINAL_RESULTS, line, results)
  results = [eval(result) for result in results]
  for index, mode in enumerate(modes):
    modes[index] += '_' + exploration_models[index]
  print modes
  plots = {}
  print environments, modes
  for triplet in zip(environments, modes, results):
    add_to_plots(plots, triplet)
  for environment, (figure, axes) in plots.iteritems():
    figure.savefig(environment + '_plots.pdf', bbox_inches='tight')

if __name__ == '__main__':
  log_file = sys.argv[1]
  main_process_log(log_file)
