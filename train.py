import cv2
import os
import numpy as np
from helpermethods import *
import pickle

# filename1="box.png"
filenames = ["hsbc.png", "mandiri.png", "bca.png"]
# filename1 = "puma_logo.png"
temp_array = []
for filename in filenames:
    img1 = cv2.imread(filename, 0)
    # img1 = extract_logo_and_add_border(img1)
    # img1 = cv2.resize(img1, (512, 512))

    sift = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    # print("kp1", kp1, "des1", des1)

    # Store and Retrieve keypoint features

    temp = pickle_keypoints(kp1, des1, img1.shape[0], img1.shape[1])
    temp_array.append(temp)
pickle.dump(temp_array, open("keypoints_database.p", "wb"))

# Retrieve Keypoint Features
keypoints_database = pickle.load(open("keypoints_database.p", "rb"))
print(len(keypoints_database))
kp2, desc2,height,width = unpickle_keypoints(keypoints_database[0])
print(height,width)
