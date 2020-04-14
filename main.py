# import cv2
# import numpy as np
# import imgui_datascience
# from PIL import Image

# from unet import UNet

# from torch.utils.data.dataset import Dataset
# from torch.utils.data import DataLoader
# import torch

# class SynthDataset(Dataset):
#     def __init__(self):
#         super(SynthDataset).__init__()

#     def __len__(self):
#         return 1


#     def __getitem__(self, idx):
#         img = np.zeros((400, 300))

        


#         img_t = torch.tensor(img)
#         img_t = img_t.permute(2, 0, 1) # (H, W, C) --> (C, H, W)    
#         return img_t


# model = UNet()
# dataset = SynthDataset()
# dataloader = DataLoader(dataset=dataset, num_workers=0, batch_size=4)

# # while True:
# #     for batch_t in dataloader:
# #         x=0



# # cv2.imshow('img', img)
# # cv2.waitKey(1)


# # img = np.zeros((400, 300), dtype=np.uint8)
# orignal = cv2.imread("pic.png")
# orignal = cv2.cvtColor(orignal, cv2.COLOR_BGR2GRAY)





from __future__ import absolute_import

import sys

import pygame
import OpenGL.GL as gl

from imgui.integrations.pygame import PygameRenderer
import imgui
from OpenGL.GL import *
import cv2
from dataclasses import dataclass
import numpy as np

@dataclass
class ImageWidget:
    img_np: np.array
    width: int = 0
    height: int = 0

    def __post_init__(self):
        self.texture_id = 0
        self.loadImage(self.img_np)
        

    def redraw(self, img_np):
        self.img_np = img_np
        self.loadImage(self.img_np)
        self.draw()

    def draw(self):
        imgui.image(self.texture_id, self.width, self.height)
        print(self.texture_id)

    # make sure the image is RGBA 
    def loadImage(self, img_np):
        if not self.width or not self.height:
            self.width, self.height, channels = self.img_np.shape

        textureData = img_np.tobytes()

        if self.texture_id:
            glDeleteTextures(1, self.texture_id)

        self.texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.width, self.height, 0, GL_RGBA,
                    GL_UNSIGNED_BYTE, textureData)


def main():
    pygame.init()
    size = 800, 600

    pygame.display.set_mode(size, pygame.DOUBLEBUF | pygame.OPENGL | pygame.RESIZABLE)

    imgui.create_context()
    impl = PygameRenderer()

    io = imgui.get_io()
    io.display_size = size

    img = cv2.imread("pic.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgba = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    img_w = ImageWidget(img_rgba)

    center = int(255/2)
    sides = 10

    while 1:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

            impl.process_event(event)

        imgui.new_frame()

        imgui.begin("Custom window", True)

        changed_center, center = imgui.slider_int("center", center, min_value=0, max_value=255, format="%.0d")
        changed_sides, sides = imgui.slider_int("sides", sides, min_value=0, max_value=255, format="%.0d")

        imgui.text(f"Changed: {changed_center}, center: {center}, sides: {sides}")

        if changed_center or changed_sides:
            i = img.copy()

            idx = i > center + sides
            i[idx] = 0

            idx = i < center - sides
            i[idx] = 0

            i = cv2.cvtColor(i, cv2.COLOR_BGR2BGRA)

            img_w.redraw(i)
        else:
            img_w.draw()

        

        

       
        

        imgui.end()

        # note: cannot use screen.fill((1, 1, 1)) because pygame's screen
        #       does not support fill() on OpenGL sufraces
        gl.glClearColor(1, 1, 1, 1)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        imgui.render()
        impl.render(imgui.get_draw_data())

        pygame.display.flip()


if __name__ == "__main__":
    main()







