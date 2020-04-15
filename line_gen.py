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

@dataclass
class RangeC: #C - container
    start: float
    end: float

    def __call__(self):
        return self.start, self.end

@dataclass
class SquiggleGen:
    size: float = 1000
    n_rotors: int = 25
    n_tail: int = 100
    rotor_radius: RangeC = RangeC(10, 100)
    rotor_coef: RangeC = RangeC(0.01, 0.001)
    debug: bool = True

    def gen_rots(self):
        rots = []
        for i in range(self.n_rotors):
            radius = random.uniform(*self.rotor_radius())
            coef = random.uniform(*self.rotor_coef())
            reverse = random.getrandbits(1)
            rot = Rotor(radius, coef, reverse) 
            rots.append(rot)
        return rots

    def gen_imgs(self, n_samples, pos):
        blank = np.zeros((self.size, self.size, 3))
        rand_pos = lambda a, b : (random.randint(a, b), random.randint(a, b))
        
        for _ in range(n_samples):
            rots = self.gen_rots()
            coords = []

            a = rand_pos(0, self.size)
            b = rand_pos(0, self.size)
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
                    ax, ay = prev
                    bx, by = curr
                    a = (int(ax), int(ay))
                    b = (int(bx), int(by))
                    img = cv2.line(img, a, b, (1, 1, 1), 3)

                cv2.namedWindow("img", cv2.WINDOW_NORMAL)
                cv2.imshow("img", img)
                cv2.waitKey(1)

                while len(coords) > self.n_tail:
                    coords.pop(0)



pos = (500, 500)


sg = SquiggleGen()
sg.gen_imgs(100, pos)

# offset




