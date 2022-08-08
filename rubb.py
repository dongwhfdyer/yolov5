"""
@file hough_lines.py
@brief This program demonstrates line finding with the Hough transform
"""
import os
import shutil
import sys
import math
import time

import cv2 as cv2
import numpy as np


def get_intersection_pts(two_lines):
    (pt1, pt2) = two_lines[0]
    (pt3, pt4) = two_lines[1]
    A = np.array([
        [pt1[1] - pt2[1], pt2[0] - pt1[0]],
        [pt3[1] - pt4[1], pt4[0] - pt3[0]]
    ])
    b = np.array([
        [pt1[1] * pt2[0] - pt1[0] * pt2[1]],
        [pt3[1] * pt4[0] - pt3[0] * pt4[1]]
    ])
    x, y = np.linalg.solve(A, b)
    return (x, y)


def line_detect_one_img(filename):
    ## [load]

    # Loads an image
    src = cv2.imread(cv2.samples.findFile(filename), cv2.IMREAD_GRAYSCALE)

    dst = cv2.Canny(src, 50, 200, None, 3)

    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

    lines = cv2.HoughLines(dst, 1, np.pi / 180, 80, None, 0, 0)
    center_x = (480, 480)
    center_radius_thres = 300
    simi_angle_thres = 40
    if lines is not None:
        pts_lines = get_pts_lines(lines)
    else:
        return None
    ##########nhuk#################################### angle inspection
    print("######################################## angle inspection")
    angle_approve_dict = {}
    for i, (pt1, pt2) in enumerate(pts_lines):
        # calculate the distance between the line and the center of the image

        # calculate the angle of the line
        angle = int(math.atan2(pt2[1] - pt1[1], pt2[0] - pt1[0]) * (180 / math.pi))
        if angle_approve_dict == {}:
            angle_approve_dict[str(angle)] = [(pt1, pt2)]
        else:
            # if the angle is too close to the previous angle, ignore it
            for an in angle_approve_dict:
                int_an = int(an)
                if abs(angle - int_an) < simi_angle_thres or 180 - abs(angle - int_an) < simi_angle_thres:
                    angle_approve_dict[an].append((pt1, pt2))
                    break
            else:
                angle_approve_dict[str(angle)] = [(pt1, pt2)]
        print("angle_approve_dict:", angle_approve_dict)
    ##########nhuk####################################

    ##########nhuk#################################### get the line with the smallest distance to the center
    angle_approved_dist_max = []
    print("######################################## dist min after angle inspection")
    for an in angle_approve_dict:
        dist_line2center_list = [get_dist_line2center(center_x, pt1, pt2) for (pt1, pt2) in angle_approve_dict[an]]
        print(dist_line2center_list)
        print(np.argmin(dist_line2center_list))
        angle_approved_dist_max.append(angle_approve_dict[an][np.argmin(dist_line2center_list)])
    ##########nhuk####################################

    ##########nhuk#################################### remove the line that is too far from the center
    print("######################################## remove the line that is too far from the center")
    angle_approved_dist_max_2 = angle_approved_dist_max.copy()
    for (pt1, pt2) in angle_approved_dist_max_2:
        print("##############################")
        if len(angle_approved_dist_max) == 2:
            break
        dist_line2center = get_dist_line2center(center_x, pt1, pt2)
        if dist_line2center > center_radius_thres:
            del angle_approved_dist_max[angle_approved_dist_max.index((pt1, pt2))]
        print("pt1:", pt1)
        print("pt2:", pt2)
        print("dist_line2center:", dist_line2center)
        print(angle_approved_dist_max)

    if len(angle_approved_dist_max) != 2:
        print("line number is not 2, the actual line number is:", len(angle_approved_dist_max))
        time.sleep(5)
        return None

    # get the intersection point of the two lines
    inter_pts = get_intersection_pts(angle_approved_dist_max)
    print("inter_pts:", inter_pts)

    ##########nhuk####################################
    for (pt1, pt2) in angle_approved_dist_max:
        dist_line2center_list = [get_dist_line2center(center_x, pt1, pt2) for (pt1, pt2) in angle_approved_dist_max]
        cv2.line(cdst, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.circle(cdst, (int(inter_pts[0]), int(inter_pts[1])), 10, (0, 255, 255), -1)

    # linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)
    #
    # if linesP is not None:
    #     for i in range(0, len(linesP)):
    #         l = linesP[i][0]
    #         cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)

    # cv2.imshow("Source", src)
    cv2.circle(cdst, center_x, center_radius_thres, (0, 0, 255), )
    cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    cv2.waitKey(0)
    # cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)

    return cdst


def get_pts_lines(lines):
    pts_lines = []
    for i in range(0, len(lines)):
        if_draw = True
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        pts_lines.append((pt1, pt2))

    return pts_lines


def get_dist_line2center(center_x, pt1, pt2):
    dist_pt1_center = math.sqrt((pt1[0] - center_x[0]) ** 2 + (pt1[1] - center_x[1]) ** 2)
    distance_pt2_center = math.sqrt((pt2[0] - center_x[0]) ** 2 + (pt2[1] - center_x[1]) ** 2)
    distance_pt1_pt2 = math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
    # Heron's formula
    s = (dist_pt1_center + distance_pt2_center + distance_pt1_pt2) / 2
    area = math.sqrt(s * (s - dist_pt1_center) * (s - distance_pt2_center) * (s - distance_pt1_pt2))
    dist_line_center = 2 * area / distance_pt1_pt2
    return dist_line_center


if __name__ == "__main__":

    data_path = r"d:\ANewspace\code\yolov5_new\datasets\shibie\result"
    out_dir = r"d:\ANewspace\code\yolov5_new\datasets\shibie\result\line_detected"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    for img_name in os.listdir(data_path):
        print("########################################")
        line_detected_img = line_detect_one_img(os.path.join(data_path, img_name))
        if line_detected_img is not None:
            cv2.imwrite(os.path.join(out_dir, img_name), line_detected_img)
