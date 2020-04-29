import numpy as np
import matplotlib
import cv2
from dataclasses import dataclass, field
from typing import List
import random
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
    n_rotors: int = 3
    n_points: int = 100
    rotor_radius: RandRange = RandRange(5, 15)
    rotor_coef: RandRange = RandRange(0.01, 0.001)
    debug: bool = True

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

    def calc_rots(self, rots, pos):
        out_of_bb = False
        bb = (0, 0, self.size, self.size)
        
        for i, rot in enumerate(rots):
            x, y = rot(pos)
            pos = (int(x), int(y))

            if not pt_in_bb(bb, pos):
                out_of_bb = True
                return out_of_bb, pos
        return out_of_bb, pos

    def gen_img(self, n_layers, pos):
        blank = np.zeros((self.size, self.size))
        layers = []

        # Layer
        while len(layers) < n_layers:
            layer = blank.copy()
            coords = []
            rots = self.gen_rots()
            
            # Line generation
            while len(coords) < self.n_points:
                out_of_bb, p = self.calc_rots(rots, pos)

                if out_of_bb:
                    # print("Out", id(layer))
                    break
                    
                coords.append(p)
            
            if out_of_bb:
                continue
            
            for prev, curr in zip(coords, coords[1:]):
                ax, ay = prev
                bx, by = curr
                a = (int(ax), int(ay))
                b = (int(bx), int(by))
                layer = cv2.line(layer, a, b, (1), 2)
                mask = layer.copy()

            # if len(layers):
            #     layer = cv2.circle(layer, (5, 5), 10, (1), -1)
            # else:
            #     layer = cv2.circle(layer, (25, 25), 10, (1), -1)

            mask = layer.copy()

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
        cv2.namedWindow("img", cv2.WINDOW_GUI_NORMAL)
        cv2.imshow("img", combined_img)
        cv2.waitKey(1)
        
        # clip back to 0..1
        combined_img = np.clip(combined_img, 0, 1.0)
        foreground_mask = np.clip(foreground_mask, 0, 1.0)

        return layers, combined_img, foreground_mask




