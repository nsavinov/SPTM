import numpy as np
import subprocess as sp
import os

class VideoWriter:

  def __init__(self, out_file, resolution, framerate=24, rgb=True, mode='check'):
    if rgb:
      pix_fmt = 'rgb24'
    else:
      pix_fmt = 'gray'

    command = [ 'ffmpeg',
          '-y', # (optional) overwrite output file if it exists
          '-f', 'rawvideo',
          '-loglevel', 'panic',
          '-vcodec','rawvideo',
          '-s', '%dx%d' % tuple(resolution), # size of one frame
          '-pix_fmt', pix_fmt,
          '-r', str(framerate), # frames per second
          '-i', '-', # The imput comes from a pipe
          '-an', # Tells FFMPEG not to expect any audio
          '-vcodec', 'h264',
          out_file]

    if mode == 'replace':
      if os.path.isfile(out_file):
        os.remove(out_file)
    elif mode == 'check':
      if os.path.isfile(out_file):
        raise Exception('File ' + out_file + ' already exists')
    elif mode == 'append':
      pass
    else:
      raise Exception('Unknown mode ' + mode)

    self.proc = sp.Popen(command, stdin=sp.PIPE)

  def add_frame(self, img):
    self.proc.stdin.write(img.astype(np.uint8).tostring())

  def close(self):
    self.proc.stdin.close()
    self.proc.wait()
    