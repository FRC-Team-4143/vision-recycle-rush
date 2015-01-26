# -*- coding: utf-8 -*-
from __future__ import division
import cv2
from cv2 import cv
import math
from matplotlib import pyplot as plt
import numpy as np

VIEW_ANGLE = 64 # View angle fo camera, 49.4 for Axis m1011, 64 for m1013, 51.7 for 206, 52 for HD3000 square, 60 for HD3000 640x480
TOTE_WIDTH = 26.9 # in
TOTE_DEPTH = 16.9 # in
TOTE_HEIGHT = 12.1 # in
RATIO_THRESHOLD = 0.1 # percent difference from actual ratio to calculated ratio


def calc_distance(target_width, target_width_px, total_width_px):
    """Calculates the distance to the target (units equal to those passed in)."""
    return target_width * total_width_px / (2 * target_width_px * math.tan(VIEW_ANGLE * math.pi / 180))

def calc_angle_x(center_x, res_x):
    """Calculates the angle to rotate to center the target (degrees)."""
    normal_x = (center_x - res_x / 2) / res_x / 2
    return normal_x * VIEW_ANGLE / 2

def nothing(x):
    """Stub function for sliders."""
    pass

def hsv_filter(image):
    low_h = cv2.getTrackbarPos("LowH", "Control")
    high_h = cv2.getTrackbarPos("HighH", "Control")
    low_s = cv2.getTrackbarPos("LowS", "Control")
    high_s = cv2.getTrackbarPos("HighS", "Control")
    low_v = cv2.getTrackbarPos("LowV", "Control")
    high_v = cv2.getTrackbarPos("HighV", "Control")
    lower = np.array([low_h, low_s, low_v])
    upper = np.array([high_h, high_s, high_v])
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, lower, upper), (low_h, high_h, low_s, high_s, low_v, high_v)

def open_close(image, size=5):
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, np.ones((size, size)))
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, np.ones((size, size)))

def ratio_score(ratio, actual):
    factor = ratio / actual
    if abs(round(factor)) > 0.01:
        return abs((factor - round(factor)) / round(factor))
    return 100

def ratio(width, height):
    ratio = height / width
    long_ratio = TOTE_HEIGHT / TOTE_WIDTH
    short_ratio = TOTE_HEIGHT / TOTE_DEPTH

    long_score = ratio_score(ratio, long_ratio)
    short_score = ratio_score(ratio, short_ratio)

    is_long = long_score < short_score
    if is_long:
        best_score = long_score
        stacks = round(ratio / long_ratio)
    else:
        best_score = short_score
        stacks = round(ratio / short_ratio)

    return is_long, best_score, stacks

def calc_bbox(contour):
    rect = cv2.minAreaRect(shape)
    x, y = rect[0]
    if rect[2] < -45:
        h, w = rect[1]
    else:
        w, h = rect[1]
    box = cv.BoxPoints(rect)
    box = np.int0(box)
    return x, y, w, h, box

def main():
    cv2.namedWindow('Control', cv2.WINDOW_AUTOSIZE)
    cv.CreateTrackbar("LowH", "Control", 0, 255, nothing)
    cv.CreateTrackbar("HighH", "Control", 0, 255, nothing)
    cv.CreateTrackbar("LowS", "Control", 0, 255, nothing)
    cv.CreateTrackbar("HighS", "Control", 0, 255, nothing)
    cv.CreateTrackbar("LowV", "Control", 0, 255, nothing)
    cv.CreateTrackbar("HighV", "Control", 0, 255, nothing)

    cv2.setTrackbarPos("LowH", "Control", 0)
    cv2.setTrackbarPos("HighH", "Control", 40)
    cv2.setTrackbarPos("LowS", "Control", 60)
    cv2.setTrackbarPos("HighS", "Control", 255)
    cv2.setTrackbarPos("LowV", "Control", 50)
    cv2.setTrackbarPos("HighV", "Control", 255)

    img = cv2.imread('SampleImages/normal_wtargets.jpg', cv2.CV_LOAD_IMAGE_COLOR)
    filters = ()

    while cv2.waitKey(30) != 27:
        img_copy = img.copy()
        filtered, filters = hsv_filter(img_copy)
        filtered = open_close(filtered)
        clone = filtered.copy()
        contours, hierarchy = cv2.findContours(clone, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for shape in contours:
            x, y, w, h, box = calc_bbox(shape)
            is_long, score, num_totes = ratio(w, h)
            if score > RATIO_THRESHOLD:
                continue
            width = TOTE_WIDTH if is_long else TOTE_DEPTH
            distance = calc_distance(width, w, img_copy.shape[1])
            angle = calc_angle_x(x, img_copy.shape[1])
            cv2.drawContours(img_copy,[box],0,(0,255,0),2)
            cv2.putText(img_copy, 'Distance: {:.3}"'.format(distance),(int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))
            cv2.putText(img_copy, 'Angle: {:.3} deg'.format(angle),(int(x), int(y)+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))
            cv2.putText(img_copy, 'Totes: {} '.format(int(num_totes)),(int(x), int(y)+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))

        cv2.drawContours(img_copy, contours, -1, (0, 0, 255))
        cv2.imshow("Original", img_copy)
        cv2.imshow("Filtered", filtered)

    print("Hue: {0[0]} to {0[1]}\n"
          "Saturation: {0[2]} to {0[3]}\n"
          "Value: {0[4]} to {0[5]}".format(filters))

if __name__ == '__main__':
    main()
