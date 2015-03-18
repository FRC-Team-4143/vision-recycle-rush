#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
from collections import namedtuple
import cv2
from cv2 import cv
import math
import numpy as np
import socket
import filters

VIEW_ANGLE = 60 # View angle fo camera, 49.4 for Axis m1011, 64 for m1013, 51.7 for 206, 52 for HD3000 square, 60 for HD3000 640x480
TOTE_WIDTH = 26.9 # in
TOTE_DEPTH = 16.9 # in
TOTE_HEIGHT = 12.1 # in
TAPE_WIDTH = 7 # in
RATIO_THRESHOLD = 0.2 # percent difference from actual ratio to calculated ratio
IP = "10.41.43.2"
IP2 = "10.4.13.2"
PORT = 4143
SCREENWIDTH = 640
MIDSCREEN = SCREENWIDTH / 2
Y_DIFF = 100  # found boxes must be within this many pixels for tote find
AREA_THRESHOLD = 0.01 # percent of image area
EDGE_THRESHOLD = 5 # number of px from edge to call it filled up

Target = namedtuple("Target", "area left right center")


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
    x,y,w,h = cv2.boundingRect(contour)
    return x, y, w, h

def main(args):
    if args.test:
        cv2.namedWindow('Control', cv2.WINDOW_AUTOSIZE)
        cv.CreateTrackbar("LowH", "Control", 0, 255, nothing)
        cv.CreateTrackbar("HighH", "Control", 0, 255, nothing)
        cv.CreateTrackbar("LowS", "Control", 0, 255, nothing)
        cv.CreateTrackbar("HighS", "Control", 0, 255, nothing)
        cv.CreateTrackbar("LowV", "Control", 0, 255, nothing)
        cv.CreateTrackbar("HighV", "Control", 0, 255, nothing)

        cv2.setTrackbarPos("LowH", "Control", filters.low_h)
        cv2.setTrackbarPos("HighH", "Control", filters.high_h)
        cv2.setTrackbarPos("LowS", "Control", filters.low_s)
        cv2.setTrackbarPos("HighS", "Control", filters.high_s)
        cv2.setTrackbarPos("LowV", "Control", filters.low_v)
        cv2.setTrackbarPos("HighV", "Control", filters.high_v)

    if args.filename:
        img = cv2.imread(args.filename, cv2.CV_LOAD_IMAGE_COLOR)
    elif args.camera:
        cap = cv2.VideoCapture(int(args.camera) if len(args.camera) < 3 else args.camera)
    else:
        raise RuntimeError("Please supply an image or a camera.")

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    while True:
        if args.test:
            filters.low_h = cv2.getTrackbarPos("LowH", "Control")
            filters.high_h = cv2.getTrackbarPos("HighH", "Control")
            filters.low_s = cv2.getTrackbarPos("LowS", "Control")
            filters.high_s = cv2.getTrackbarPos("HighS", "Control")
            filters.low_v = cv2.getTrackbarPos("LowV", "Control")
            filters.high_v = cv2.getTrackbarPos("HighV", "Control")

        if args.filename:
            img_copy = img.copy()
        elif args.camera:
            ret, img_copy = cap.read()

        filtered = hsv_filter(img_copy, filters.low_h, filters.high_h,
                              filters.low_s, filters.high_s, filters.low_v,
                              filters.high_v)
        filtered = open_close(filtered)
        contours, hierarchy = cv2.findContours(filtered.copy(),
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)

        targets = list()
        for shape in contours:
            area = cv2.contourArea(shape)
            if area < AREA_THRESHOLD * img_copy.shape[0] * img_copy.shape[1]:
                continue
            x, y, w, h = calc_bbox(shape)
            targets.append(Target(area, x, x+w, (x+w/2, y+h/2)))
            if not args.novideo:
                distance = calc_distance(TAPE_WIDTH, w, img_copy.shape[1])
                angle = calc_angle_x(x, img_copy.shape[1])
                cv2.rectangle(img_copy,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.putText(img_copy, 'Distance: {:.3}"'.format(distance),
                            (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0,0,0))
                cv2.putText(img_copy, 'Angle: {:.3} deg'.format(angle),
                            (int(x), int(y)+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0,0,0))

        mid_x = 1.0
        mid_y = 0
        if len(targets) > 1:
            targets.sort(key=lambda tup: tup.area, reverse=True)
            if abs(targets[0].center[1] - targets[1].center[1]) <= Y_DIFF:
                if targets[0].left < targets[1].left:
                    if targets[0].left < EDGE_THRESHOLD and targets[1].right > SCREENWIDTH - EDGE_THRESHOLD:
                        mid_x = MIDSCREEN
                    else:
                        mid_x = (targets[1].left - targets[0].right) / 2 + targets[0].right
                else:
                    if targets[1].left < EDGE_THRESHOLD and targets[0].right > SCREENWIDTH - EDGE_THRESHOLD:
                        mid_x = MIDSCREEN
                    else:
                        mid_x = (targets[0].left - targets[1].right) / 2 + targets[1].right
                mid_y = sum(i.center[1] for i in targets[:2]) / 2
            else:  # if y test fails just send center of biggest box
                mid_x, mid_y = targets[0].center
                # print "Y mismatch"
            mid_x = mid_x - MIDSCREEN
        elif len(targets) == 1:  # only one target found. send center of it
            mid_x, mid_y = targets[0].center
            mid_x = mid_x - MIDSCREEN


        #print mid_x
        try:
            s.sendto(str(mid_x), (IP, PORT))
        except:
            try:
                s.sendto(str(mid_x), (IP2, PORT))
            except:
                pass

        if not args.novideo:
            cv2.circle(img_copy, (int(mid_x+MIDSCREEN), int(mid_y)), 10, (0,0,255), -1)
            cv2.drawContours(img_copy, contours, -1, (0, 0, 255))
            cv2.imshow("Original", img_copy)
            cv2.imshow("Filtered", filtered)
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
            f.write("low_h = {}\n".format(filters.low_h))
            f.write("high_h = {}\n".format(filters.high_h))
            f.write("low_s = {}\n".format(filters.low_s))
            f.write("high_s = {}\n".format(filters.high_s))
            f.write("low_v = {}\n".format(filters.low_v))
            f.write("high_v = {}\n".format(filters.high_v))

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
