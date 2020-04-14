from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import cv2
import numpy as np

fig = pyplot.figure()
ax = Axes3D(fig)


mask = cv2.imread('mask.png')
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY) 


img = cv2.imread('pic.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = cv2.resize(img, (200, 200))
w, h = img.shape

x_vals = []
y_vals = []
z_vals = []

i = 0
for x in range(w):
    for y in range(h):
        v = img[x][y]
        t = mask[x][y]

        if v > 70 and t:
            y_vals.append(y)
            x_vals.append(x)
            z_vals.append(v) 
        i += 1



ax.scatter(x_vals, y_vals, z_vals, s=2)
pyplot.show()
