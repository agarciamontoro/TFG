# coding: utf-8
import numpy as np
import h5py

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

from progress_lib import progress_bar_init

import os
import time
import re

import argparse

# ============================= PARSE ARGUMENTS ============================= #

parser = argparse.ArgumentParser(description='Creates a gif movie from the nbody simulation')

parser.add_argument('-i', '--input', dest='input', type=str,
                    default="Output/nbody.hdf5",
                    help='Path to the file where the data is stored.')

parser.add_argument('-o', '--output', dest='output', type=str,
                    default="out_default.gif",
                    help='Path to the file where the result will be stored.')

parser.add_argument('-f', '--frames', dest='numFrames', type=int,
                    default=200,
                    help='Number of steps simulated.')

parser.add_argument('-b', '--bodies', dest='numBodies', type=int,
                    default=8192,
                    help='Number of bodies used in the simulation.')

args = parser.parse_args()

num_frames = args.numFrames
out_path = args.output
num_bodies = args.numBodies

# Get data
hdf5_root = h5py.File(args.input, "r")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# ax.patch.set_visible(False)

axes = fig.gca()
axes.set_xlim([-15, 15])
axes.set_ylim([-15, 15])
axes.set_zlim([-15, 15])

axes.set_axis_off()
ax.set_axis_bgcolor((0.15, 0.15, 0.15))

axes.set_position([0, 0, 1, 1])

azimuth = 0
ax.view_init(elev=10., azim=azimuth)

scat1 = ax.scatter([], [], [], c="darkcyan", s=2, lw=0)
scat2 = ax.scatter([], [], [], c="darkolivegreen", s=2, lw=0)
scat3 = ax.scatter([], [], [], c="paleturquoise", s=2, lw=0)
scat4 = ax.scatter([], [], [], c="olive", s=2, lw=0)
scat5 = ax.scatter([], [], [], c="darkcyan", s=2, lw=0)
scat6 = ax.scatter([], [], [], c="darkolivegreen", s=2, lw=0)

progress_bar = progress_bar_init(num_frames-1)

old_end = time.time()

def animate(frame):
    global old_end
    start = old_end
    set_name = "out_%03d.csv" % frame

    data = hdf5_root[set_name]

    and_disk = data[:16384, :]
    mil_disk = data[16384:32768, :]
    and_bulg = data[32768:40960, :]
    mil_bulg = data[40960:49152, :]
    and_halo = data[49152:65536, :]
    mil_halo = data[65536:, :]

    scat1._offsets3d = (np.ma.ravel(and_disk[:, 0]),
                        np.ma.ravel(and_disk[:, 1]),
                        np.ma.ravel(and_disk[:, 2]))

    scat2._offsets3d = (np.ma.ravel(mil_disk[:, 0]),
                        np.ma.ravel(mil_disk[:, 1]),
                        np.ma.ravel(mil_disk[:, 2]))

    scat3._offsets3d = (np.ma.ravel(and_bulg[:, 0]),
                        np.ma.ravel(and_bulg[:, 1]),
                        np.ma.ravel(and_bulg[:, 2]))

    scat4._offsets3d = (np.ma.ravel(mil_bulg[:, 0]),
                        np.ma.ravel(mil_bulg[:, 1]),
                        np.ma.ravel(mil_bulg[:, 2]))

    scat5._offsets3d = (np.ma.ravel(and_halo[:, 0]),
                        np.ma.ravel(and_halo[:, 1]),
                        np.ma.ravel(and_halo[:, 2]))

    scat6._offsets3d = (np.ma.ravel(mil_halo[:, 0]),
                        np.ma.ravel(mil_halo[:, 1]),
                        np.ma.ravel(mil_halo[:, 2]))

    end = time.time()
    old_end = end
    progress_bar(end - start)
    return scat1, scat2,

anim = animation.FuncAnimation(fig, animate,
                               frames=num_frames,
                               interval=20)


anim.save(out_path, writer='imagemagick', fps=30)
