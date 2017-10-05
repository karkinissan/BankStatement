import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from helpermethods import *

# filename = "hsbc_statement.png"
# filenames = ["hsbc_camera.jpg"]
# filename = "mandiri_statement.png"
# filename = "mandiri_camera.jpg"
# filename = "bca_statement.png"
# filename = "bca_camera.jpg"
filenames = ["hsbc_camera.jpg","mandiri_camera.jpg","bca_camera.jpg"]
for filename in filenames:
    pic = cv2.imread(filename)

    # img = cv2.imread(filename, 0)
    ratio = pic.shape[0]/pic.shape[1]
    pic = cv2.resize(pic,(1280,int(1280*ratio)),interpolation = cv2.INTER_CUBIC)
    img_x, index = findLogo(pic)
    bankName = index_to_name(index)
    print(bankName)
    # img_x = cv2.resize(img_x,(1280,int(1280*ratio)),interpolation = cv2.INTER_CUBIC)
    cv2.imshow('img_x', img_x)
    # plt.imshow(img_x, 'img_x'),plt.show()
    cv2.waitKeyEx()
    cv2.destroyAllWindows()

