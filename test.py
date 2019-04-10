#!/usr/bin/env python3

import imageio
import matplotlib.pyplot as plt
import numpy as np

from haar import *

if __name__ == "__main__":
    cam = imageio.imread("imageio:camera.png")
    imageio.imsave("cam.png", cam)
 
    cam = imageio.imread("cam.png").astype(float)
    xform_cam = fast_2d_haar_transform(cam)#.astype(np.uint8)

    imageio.imsave("cam_xform.png", xform_cam)

