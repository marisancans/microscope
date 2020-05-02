import numpy as np
import matplotlib
import cv2
from dataclasses import dataclass, field
from typing import List
import random
import math
from helpers import pt_in_bb

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
class Layer:
    img: np.array
    mask: np.array
    label_id: int
    coords: list = field(default_factory=list)

# uses furier series
@dataclass
class SquiggleGen:
    size: float = 500
    n_rotors: int = RandRange(5, 10)
    n_points: int = 100
    n_try_points: int = 1000 # behind image borders
    rotor_radius: RandRange = RandRange(150, 250)
    rotor_coef: RandRange = RandRange(0.01, 0.001)
    spawn_radius: int = 500
    debug: bool = True

    def gen_rots(self):
        rots = []
        for i in range(int(self.n_rotors())):
            radius = self.rotor_radius()
            coef = self.rotor_coef()
            reverse = random.getrandbits(1)
            rot = Rotor(radius, coef, reverse) 
            rot.t = RandRange(1, 1000)()
            rots.append(rot)
        return rots

    def calc_rots(self, rots, pos):
        for i, rot in enumerate(rots):
            x, y = rot(pos)
            pos = (int(x), int(y))
        return  pos

    def gen_img(self, n_layers, pos):
        blank = np.zeros((self.size, self.size))
        bb = (0, 0, self.size, self.size)
        layers = []

        # pos generation
        # random angle
        alpha = 2 * math.pi * random.random()

        # random radius
        r = self.spawn_radius * math.sqrt(random.random())
        # calculating coordinates
        x_pos = r * math.cos(alpha) + self.size // 2
        y_pos = r * math.sin(alpha) + self.size // 2
        pos = (x_pos, y_pos)

        # Layer
        while len(layers) < n_layers:
            layer = blank.copy()
            mask = blank.copy()
            coords = []
            rots = self.gen_rots()
            in_bounds = False
            line_color = random.uniform(0.1, 0.9)
            
            # Line generation
            prev_p = None
            for i in range(self.n_try_points):
                p = self.calc_rots(rots, pos)
                
                if pt_in_bb(bb, p):
                    if prev_p:
                        coords.append(prev_p)
                    coords.append(p)
                    in_bounds = True
                    break # we found a point in bounds
                prev_p = p
            
            # retry
            if not in_bounds:
                continue

            # now we try to collect n points
            prev_p = None
            for i in range(self.n_points):
                if prev_p:
                    coords.append(prev_p)

                p = self.calc_rots(rots, pos)
                coords.append(p)    
                prev_p = p   

                if not pt_in_bb(bb, p):
                    break

            line_width = 5
            for prev, curr in zip(coords, coords[1:]):
                ax, ay = prev
                bx, by = curr
                a = (int(ax), int(ay))
                b = (int(bx), int(by))
                layer = cv2.line(layer, a, b, (line_color), line_width)
                mask = cv2.line(mask, a, b, (1), line_width)

            # if len(layers):
            #     layer = cv2.circle(layer, (5, 5), 10, (1), -1)
            # else:
            #     layer = cv2.circle(layer, (25, 25), 10, (1), -1)

            l = Layer(img=layer, mask=mask, label_id=len(layers), coords=coords)

            # check overlapping with previous masks
            for prev in layers:
                mask_bwx = cv2.bitwise_and(prev.mask, l.mask)
                sub = prev.mask - mask_bwx
                prev.mask = sub

            layers.append(l)

        combined_img = blank.copy()
        foreground_mask = blank.copy()
        
        for i, l in enumerate(layers):
            combined_img += l.img
            foreground_mask += l.mask
        # cv2.namedWindow("img", cv2.WINDOW_GUI_NORMAL)
        # cv2.imshow("img", combined_img)
        # cv2.waitKey(0)
        
        # clip back to 0..1
        combined_img = np.clip(combined_img, 0, 1.0)
        foreground_mask = np.clip(foreground_mask, 0, 1.0)

        return layers, combined_img, foreground_mask




