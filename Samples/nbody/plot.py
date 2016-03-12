# coding: utf-8
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import glob

file_list = glob.glob("./out_03*.csv")
file_list.sort()

for file_name in file_list:
    raw_data = np.loadtxt(file_name)
    milky_way = raw_data[:16384:24, :]
    andromeda = raw_data[16384:16384*2:24, :]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    axes = plt.gca()
    axes.set_xlim([-1, 1])
    axes.set_ylim([-1, 1])
    axes.set_zlim([-40, 20])

    x = milky_way[:, 0]
    y = milky_way[:, 1]
    z = milky_way[:, 2]

    ax.scatter(x, y, z, c="r", s=2.5, lw=0)

    x = andromeda[:, 0]
    y = andromeda[:, 1]
    z = andromeda[:, 2]

    ax.scatter(x, y, z, c="b", s=2.5, lw=0)

    # A.x = x[0:1707]
    # A_x = x[0:1707]
    # B_x = x[1707:]
    # A_y = y[0:1707]
    # B_y = y[1707:]
    # A_z = z[0:1707]
    # B_z = z[1707:]

    fig.savefig(file_name + ".png")
    print(file_name + " converted to image")
