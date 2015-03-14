# -*- coding: utf-8 -*-
from __future__ import division
import cv2
from cv2 import cv
import math
from matplotlib import pyplot as plt
import numpy as np
import socket
from filters import low_h, high_h, low_s, high_s, low_v, high_v

VIEW_ANGLE = 60 # View angle fo camera, 49.4 for Axis m1011, 64 for m1013, 51.7 for 206, 52 for HD3000 square, 60 for HD3000 640x480
TOTE_WIDTH = 26.9 # in
TOTE_DEPTH = 16.9 # in
TOTE_HEIGHT = 12.1 # in
TAPE_WIDTH = 7 # in
RATIO_THRESHOLD = 0.2 # percent difference from actual ratio to calculated ratio
IP = "20.20.41.43"
PORT = 4143


def calc_distance(target, target_px, total_px):
    """Calculates the distance to the target (units equal to those passed in)."""
    return target * total_px / (2 * target_px * math.tan(VIEW_ANGLE / 2 * math.pi / 180))

def calc_angle_x(center_x, res_x):
    """Calculates the angle to rotate to center the target (degrees)."""
    normal_x = (center_x - res_x / 2) / res_x / 2
    return normal_x * VIEW_ANGLE / 2

def nothing(x):
    """Stub function for sliders."""
    pass

def hsv_filter(image, low_h, high_h, low_s, high_s, low_v, high_v):
    lower = np.array([low_h, low_s, low_v])
    upper = np.array([high_h, high_s, high_v])
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, lower, upper)

def open_close(image, size=10):
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
    rect = cv2.minAreaRect(contour)
    x, y = rect[0]
    if rect[2] < -45:
        h, w = rect[1]
    else:
        w, h = rect[1]
    box = cv.BoxPoints(rect)
    box = np.int0(box)
    return x, y, w, h, box

def main(args):
    global low_h
    global low_s
    global low_v
    global high_h
    global high_s
    global high_v
    if args.test:
        cv2.namedWindow('Control', cv2.WINDOW_AUTOSIZE)
        cv.CreateTrackbar("LowH", "Control", 0, 255, nothing)
        cv.CreateTrackbar("HighH", "Control", 0, 255, nothing)
        cv.CreateTrackbar("LowS", "Control", 0, 255, nothing)
        cv.CreateTrackbar("HighS", "Control", 0, 255, nothing)
        cv.CreateTrackbar("LowV", "Control", 0, 255, nothing)
        cv.CreateTrackbar("HighV", "Control", 0, 255, nothing)

        cv2.setTrackbarPos("LowH", "Control", low_h)
        cv2.setTrackbarPos("HighH", "Control", high_h)
        cv2.setTrackbarPos("LowS", "Control", low_s)
        cv2.setTrackbarPos("HighS", "Control", high_s)
        cv2.setTrackbarPos("LowV", "Control", low_v)
        cv2.setTrackbarPos("HighV", "Control", high_v)

    if args.filename:
        img = cv2.imread(args.filename, cv2.CV_LOAD_IMAGE_COLOR)
    elif args.camera:
        cap = cv2.VideoCapture(int(args.camera) if len(args.camera) < 3 else args.camera)

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    while True:
        if args.filename:
            img_copy = img.copy()
        elif args.camera:
            ret, img_copy = cap.read()

        if args.test:
            low_h = cv2.getTrackbarPos("LowH", "Control")
            high_h = cv2.getTrackbarPos("HighH", "Control")
            low_s = cv2.getTrackbarPos("LowS", "Control")
            high_s = cv2.getTrackbarPos("HighS", "Control")
            low_v = cv2.getTrackbarPos("LowV", "Control")
            high_v = cv2.getTrackbarPos("HighV", "Control")

        filtered = hsv_filter(img_copy, low_h, high_h, low_s, high_s, low_v, high_v)
        filtered = open_close(filtered)
        clone = filtered.copy()
        contours, hierarchy = cv2.findContours(clone, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        targets = list()
        for shape in contours:
            x, y, w, h, box = calc_bbox(shape)
            area = w * h
            if area < 0.05 * img_copy.shape[0] * img_copy.shape[1]:
                continue
            distance = calc_distance(TAPE_WIDTH, w, img_copy.shape[1])
            angle = calc_angle_x(x, img_copy.shape[1])
            targets.append((area, distance, angle, x, x+h))
            if not args.novideo:
                cv2.drawContours(img_copy,[box],0,(0,255,0),2)
                cv2.putText(img_copy, 'Distance: {:.3}"'.format(distance),(int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))
                cv2.putText(img_copy, 'Angle: {:.3} deg'.format(angle),(int(x), int(y)+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))

        if targets > 2:
            targets.sort(key=lambda tup: tup[0], reverse=True)
        elif targets < 2:
            continue

        if targets[0][2] < targets[1][2]:
            mid_px = (targets[1][3] - targets[0][4]) / 2. + targets[0][4]
        else:
            mid_px = (targets[0][3] - targets[1][4]) / 2. + targets[1][4]

        s.sendto(str(mid_px), (IP, PORT))

        if not args.novideo:
            cv2.drawContours(img_copy, contours, -1, (0, 0, 255))
            cv2.imshow("Original", img_copy)
            cv2.imshow("Filtered", filtered)

        if not args.novideo:
            key = cv2.waitKey(30)
            if args.test and (key == 83 or key == 115):
                save = True
                break
            if key == 27:
                save = False
                break

    if args.camera:
        cap.release()

    if args.test and save:
        with open("filters.py", 'w') as f:
            f.write("low_h = {}\n".format(low_h))
            f.write("high_h = {}\n".format(high_h))
            f.write("low_s = {}\n".format(low_s))
            f.write("high_s = {}\n".format(high_s))
            f.write("low_v = {}\n".format(low_v))
            f.write("high_v = {}\n".format(high_v))

    cv2.destroyAllWindows()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("-f", "--filename",
                        help="path to an image to process")
    action.add_argument("-c", "--camera",
                        help="camera number or address to process")
    parser.add_argument("-t", "--test", action="store_true",
                        help="test mode to help pick parameters")
    parser.add_argument("-v", "--novideo", action="store_true",
                        help="do not show images")
    args = parser.parse_args()
    main(args)
