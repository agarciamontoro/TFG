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

regexp = r'out_[0-3][0-9]*\.csv$'
file_list = [f for f in os.listdir(data_dir_path) if re.search(regexp, f)]
file_list.sort()

# Setup plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

axes = fig.gca()
axes.set_xlim([-40, 20])
axes.set_ylim([-40, 20])
axes.set_zlim([-40, 20])

axes.set_axis_off()
ax.set_axis_bgcolor((0.15, 0.15, 0.15))

# Setup progress bar
progress_bar = progress_bar_init(len(file_list)-1)

# Start
for file_name in file_list:
    start = time.time()

    file_name = os.path.join(data_dir_path, file_name)

    raw_data = np.loadtxt(file_name)

    disk1, disk2, bulge1, bulge2 = np.split(raw_data, [2730, 5460, 6826])

    milky_way = np.concatenate((disk1, bulge1))
    andromeda = np.concatenate((disk2, bulge2))

    x = milky_way[:, 0]
    y = milky_way[:, 1]
    z = milky_way[:, 2]

    scat1 = ax.scatter(x, y, z, c="w", s=2.5, lw=0)

    x = andromeda[:, 0]
    y = andromeda[:, 1]
    z = andromeda[:, 2]

    scat2 = ax.scatter(x, y, z, c="b", s=2.5, lw=0)

    fig.savefig(file_name + ".png")
    scat1.remove()
    scat2.remove()

    end = time.time()
    progress_bar(end - start)
