# coding: utf-8
import numpy as np

data = np.loadtxt("./out.csv")
x = data[::24, 0]
y = data[::24, 1]
z = data[::24, 2]

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

A.x = x[0:1707]
A_x = x[0:1707]
B_x = x[1707:]
A_y = y[0:1707]
B_y = y[1707:]
A_z = z[0:1707]
B_z = z[1707:]

ax.scatter(A_x, A_y, A_z, s=0.1, c="red")
ax.scatter(B_x, B_y, B_z, s=0.1, c="blue")
