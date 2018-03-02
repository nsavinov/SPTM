import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

class TrajectoryPlotter:
  def __init__(self, save_path, min_x, min_y, max_x, max_y):
    self.size_x = max_x - min_x
    self.min_x = min_x
    self.size_y = max_y - min_y
    self.min_y = min_y
    self.square_size = max([self.size_x, self.size_y])
    self.save_path = save_path
    self.points = []
    self.edges = []

  def add_point(self, point):
    self.points.append(point)

  def add_edge(self, edge):
    self.edges.append(edge)

  def save(self):
    plt.plot([x - self.min_x for x, y in self.points], [y - self.min_y for x, y in self.points], 'b')
    for edge in self.edges:
      first_x, second_x, first_y, second_y = edge
      plt.plot([first_x - self.min_x, second_x - self.min_x], [first_y - self.min_y, second_y - self.min_y], 'r')
    plt.axis([0, self.size_x, 0, self.size_y])
    plt.xticks([])
    plt.yticks([])
    plt.savefig(self.save_path, bbox_inches='tight')
    plt.close()
