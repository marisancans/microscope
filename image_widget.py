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