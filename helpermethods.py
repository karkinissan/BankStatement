import cv2
import numpy as np
import pickle
import time


def calculateMaxAndMinDistance(dst):
    z = np.array([])
    y = dst.reshape(-1).reshape(-1, 2)
    z = np.append(z, (np.linalg.norm(np.array(y[0]) - np.array(y[1]))))
    z = np.append(z, (np.linalg.norm(np.array(y[0]) - np.array(y[3]))))
    z = np.append(z, (np.linalg.norm(np.array(y[1]) - np.array(y[2]))))
    z = np.append(z, (np.linalg.norm(np.array(y[2]) - np.array(y[3]))))
    return z.max(), z.min()


def findLogo(image):
    # extract_page(image)
    stra = ""
    masksum = 0
    M_topLeft_det = 0
    area = 0
    perimeter = 0
    start_time = time.time()
    MIN_MATCH_COUNT = 10
    sift = cv2.xfeatures2d.SIFT_create()
    img2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kp1s = []
    des1s = []
    h1s = []
    w1s = []
    # Retrieve Keypoint Features
    keypoints_database = pickle.load(open("keypoints_database.p", "rb"))
    for kp_des in keypoints_database:
        kp, des, h, w = unpickle_keypoints(kp_des)
        kp1s.append(kp)
        des1s.append(des)
        h1s.append(h)
        w1s.append(w)
    # for i in range(0, len(kp1[0])):
    #     print (kp1[0][i].pt)
    kp2, des2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = None
    good = []
    goodlen = 0
    j = 0
    for i in range(0, len(kp1s)):
        matches_itr = flann.knnMatch(des1s[i], des2, k=2)
        # store all the good matches as per Lowe's ratio test.
        good_itr = []
        for m, n in matches_itr:
            if m.distance < 0.7 * n.distance:
                good_itr.append(m)
        print("len good itr", len(good_itr))
        if len(good_itr) > goodlen:
            goodlen = len(good_itr)
            matches = matches_itr
            good = good_itr
            j = i

    print(j)
    kp1 = kp1s[j]
    des1 = des1s[j]
    h1 = h1s[j]
    w1 = w1s[j]
    print("{} matches found".format(len(matches)))
    print("Time:", time.time() - start_time)
    if len(kp2) != 0 and des2 is not None:
        # print("Time:", time.time() - start_time)
        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()

            # h, w = img1.shape
            # h = h
            # w = w
            pts = np.float32([[0, 0], [0, h1 - 1], [w1 - 1, h1 - 1], [w1 - 1, 0]]).reshape(-1, 1, 2)
            if M is not None:
                for i in matchesMask:
                    masksum = masksum + i
                # print(masksum)
                # print("Masksum", masksum)
                # print (M)
                # Check if the homography is good.
                # If the determinant of the top left 2x2 matrix is >0, then it is good
                M_topLeft = M[0:2:1, 0:2:1]
                M_topLeft_det = np.linalg.det(M_topLeft)
                dst = cv2.perspectiveTransform(pts, M)
                # Calculate relative gap between the longest and shortest sides
                (longest_side, shortest_side) = calculateMaxAndMinDistance(dst)
                relative_gap = (longest_side - shortest_side) / longest_side
                # Find the area of the box of the 4 points
                area = cv2.contourArea(dst.reshape((-1, 1, 2)).astype(np.int32))

                # if masksum > 15 and M_topLeft_det > 0 and area > 100 and relative_gap < 0.8:
                #     print("Drawing")
                # for i in dst:
                #     for j in i:
                #         stra = stra + str(j[0]) + " " + str(j[1])
                #     stra = stra + " "
                image = cv2.polylines(image, [np.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)
                # font = cv2.FONT_HERSHEY_SIMPLEX
                # img2 = cv2.putText(img2, str(masksum) + " " +
                #                    str(M_topLeft_det) + " " +
                #                    str(area) + " " + str(relative_gap),
                #                    (0, 300), font, 1, (0, 255, 255), 2, cv2.LINE_AA)
                # else:
                #     print("Not Drawing")
        else:
            print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
            matchesMask = None
    print("Time:", time.time() - start_time)
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # img2 = cv2.putText(img2, str(len(good)), (0, 300), font, 1, (0, 255, 255), 2, cv2.LINE_AA)

    # img2 = cv2.putText(img2, stra, (0, 400), font, 0.5, (0, 255, 255), 2, cv2.LINE_AA)

    return image, j


def extract_page(image):
    area = 0
    largest_contour = None
    img_orig = image.copy()
    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contours = contours[1:len(contours)]
    for cnt in contours:
        cnt_area = cv2.contourArea(cnt)
        if area < cnt_area:
            area = cnt_area
            largest_contour = cnt
    x, y, w, h = cv2.boundingRect(largest_contour)
    img = cv2.rectangle(img_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return img
    # print (x,y,w,h)
    # image = img_orig[x:w, y:h]
    # cv2.imshow("image",img)
    # cv2.waitKeyEx()
    # cv2.destroyAllWindows()

def pickle_keypoints(keypoints, descriptors, height, width):
    i = 0
    temp_array = []
    for point in keypoints:
        temp = (point.pt, point.size, point.angle, point.response, point.octave,
                point.class_id, descriptors[i])
        i = i + 1
        temp_array.append(temp)
    temp_array.append(height)
    temp_array.append(width)
    return temp_array


def unpickle_keypoints(array):
    keypoints = []
    descriptors = []
    filenames = []
    height = 0
    width = 0
    for point in array[0:(len(array) - 2)]:
        temp_feature = cv2.KeyPoint(x=point[0][0], y=point[0][1], _size=point[1], _angle=point[2], _response=point[3],
                                    _octave=point[4], _class_id=point[5])
        temp_descriptor = point[6]
        keypoints.append(temp_feature)
        descriptors.append(temp_descriptor)
    height = array[(len(array) - 2)]
    width = array[(len(array) - 1)]
    return keypoints, np.array(descriptors), height, width


def index_to_name(i):
    return {
        0: "HSBC",
        1: "Mandiri",
        2: "BCA"
    }.get(i, "Unknown")
