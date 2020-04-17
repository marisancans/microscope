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
class RandRange:
    start: float
    end: float

    def __call__(self):
        return random.uniform(self.start, self.end)

@dataclass
class SquiggleGen:
    size: float = 1000
    n_rotors: int = 25
    n_points: int = 100
    rotor_radius: RandRange = RandRange(10, 50)
    rotor_coef: RandRange = RandRange(0.01, 0.001)
    debug: bool = True

    def pt_in_img(self, img, pt):
        w, h = img.shape
        x, y = pt
        a = 0 < x < w
        b = 0 < y < h
        return a and b

    def gen_rots(self):
        rots = []
        for i in range(self.n_rotors):
            radius = self.rotor_radius()
            coef = self.rotor_coef()
            reverse = random.getrandbits(1)
            rot = Rotor(radius, coef, reverse) 
            rot.t = RandRange(1, 1000)()
            rots.append(rot)
        return rots

    def calc_rots(self, img, rots, pos):
        out_of_bb = False
        
        for i, rot in enumerate(rots):
            x, y = rot(pos)
            pos = (int(x), int(y))

            if not self.pt_in_img(img, pos):
                out_of_bb = True
                return out_of_bb, pos
        return out_of_bb, pos

    def gen_imgs(self, n_samples, pos):
        blank = np.zeros((self.size, self.size))
        rand_pos = lambda a, b : (random.randint(a, b), random.randint(a, b))

        samples = []
        coords = []
        
        while len(samples) < n_samples:
            rots = self.gen_rots()
            co = []

            a = rand_pos(0, self.size)
            b = rand_pos(0, self.size)
            # a = (0, 0)
            # b = (s, s)
            points_on_line = np.linspace(a, b, 100) # 100 samples on the line
            points_on_line = points_on_line.tolist()
            points_on_line = [(int(x), int(y)) for x, y in points_on_line]

            points_on_line = [pos for x in range(self.n_points)]

            for pos in points_on_line:
                img = blank.copy()
                out_of_bb, p = self.calc_rots(img, rots, pos)

                if out_of_bb:
                    print("Out", id(img))
                    break
                    
                co.append(p)
            
            if len(co) < self.n_points:
                continue

            for prev, curr in zip(co, co[1:]):
                ax, ay = prev
                bx, by = curr
                a = (int(ax), int(ay))
                b = (int(bx), int(by))
                img = cv2.line(img, a, b, (1), 5)


            if self.debug:
                cv2.namedWindow("img", cv2.WINDOW_NORMAL)
                cv2.imshow("img", img)
                cv2.waitKey(1)


            samples.append(img)
            coords.append(co)
        return samples, coords




