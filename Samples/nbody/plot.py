# coding: utf-8
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from progress_lib import progress_bar_init

import os
import time
import re

# Get file list
data_dir_path = "./Output"

regexp = r'out_[012][0-9]*\.csv$'
file_list = [f for f in os.listdir(data_dir_path) if re.search(regexp, f)]
file_list.sort()

# Setup plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

axes = fig.gca()
axes.set_xlim([-15, 15])
axes.set_ylim([-15, 15])
axes.set_zlim([-15, 15])

axes.set_axis_off()
ax.set_axis_bgcolor((0.15, 0.15, 0.15))

azimuth = 0
ax.view_init(elev=10., azim=azimuth)

# Setup progress bar
progress_bar = progress_bar_init(len(file_list)-1)

# Start
for file_name in file_list:
    start = time.time()

    file_name = os.path.join(data_dir_path, file_name)

    raw_data = np.loadtxt(file_name)

    disk1, disk2, bulge1, bulge2 = np.split(raw_data, [2730, 5460, 6826])

    disk1 = disk1[::5, :]
    disk2 = disk2[::5, :]
    bulge1 = bulge1[::5, :]
    bulge2 = bulge2[::5, :]

    scat1 = ax.scatter(disk1[:, 0], disk1[:, 1], disk1[:, 2],
                       c="darkcyan", s=2, lw=0)

    scat2 = ax.scatter(bulge1[:, 0], bulge1[:, 1], bulge1[:, 2],
                       c="paleturquoise", s=2, lw=0)

    scat3 = ax.scatter(disk2[:, 0], disk2[:, 1], disk2[:, 2],
                       c="darkolivegreen", s=2, lw=0)

    scat4 = ax.scatter(bulge2[:, 0], bulge2[:, 1], bulge2[:, 2],
                       c="olive", s=2, lw=0)

    # azimuth += 2
    # azimuth %= 360
    # ax.view_init(elev=10., azim=azimuth)

    fig.savefig(file_name + ".png")
    scat1.remove()
    scat2.remove()
    scat3.remove()
    scat4.remove()

    end = time.time()
    progress_bar(end - start)
