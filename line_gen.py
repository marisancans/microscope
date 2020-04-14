import numpy as np
import matplotlib
import cv2
from dataclasses import dataclass
import random

@dataclass
class Rotor:
    radius: float
    coef: float
    reverse: bool

    def __post_init__(self):
        self.t = 0

    def __call__(self, pos):
        theta = self.t * np.pi * 2 * self.coef
        if self.reverse:
            theta = -theta
        x = self.radius * np.sin(theta)
        y = self.radius * np.cos(theta)
        self.t += 1

        xc, yc = pos
        x += xc 
        y += yc 
        return x, y


# params
s = 1000
r = 50



pos = (int(s/2), int(s/2))




blank = np.zeros((s, s, 3))

# rots = []
# rot = Rotor(80, 0.00007) 
# rots.append(rot)
# rot = Rotor(40, 0.00018) 
# rots.append(rot)
# rot = Rotor(23, 0.002532) 
# rots.append(rot)
# rot = Rotor(18, 0.002342) 
# rots.append(rot)


# offset
coords = []

rand_pos = lambda a, b : (random.randint(a, b), random.randint(a, b))

for _ in range(1000):
    rots = []
    r_max = 100
    for i in range(25):
        radius = random.uniform(10, r_max)
        # r_max *= 0.99
        coef = random.uniform(0.01, 0.001)
        reverse = random.getrandbits(1)
        rot = Rotor(radius, coef, reverse) 
        rots.append(rot)

    a = rand_pos(0, s)
    b = rand_pos(0, s)
    # a = (0, 0)
    # b = (s, s)
    points_on_line = np.linspace(a, b, 100) # 100 samples on the line
    points_on_line = points_on_line.tolist()
    points_on_line = [(int(x), int(y)) for x, y in points_on_line]
    points_on_line = [pos for x in range(1000)]

    for pos in points_on_line:
        img = blank.copy()
        p = pos

        for i, rot in enumerate(rots):
            x, y = rot(p)
            img = cv2.line(img, p, (int(x), int(y)), (1, 0, 0), 2)
            p = (int(x), int(y))
            
        coords.append([x, y])

        for prev, curr in zip(coords, coords[1:]):
            if prev == 0 or curr == 0:
                continue

            ax, ay = prev
            bx, by = curr
            a = (int(ax), int(ay))
            b = (int(bx), int(by))
            img = cv2.line(img, a, b, (1, 1, 1), 3)

        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        cv2.imshow("img", img)
        cv2.waitKey(1)

        while len(coords) > 100:
            coords.pop(0)
    coords.append(0)
