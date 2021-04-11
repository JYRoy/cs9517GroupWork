#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Junyu Zhang
# Zid: z5304897
# Description: the main processing is shown in the main function, I also wrote some functions as utils, such as get_features, sigmoid, get_hog_features, etc.
# Note that I use the code of reading JSON file in the given repo. I declare it in the reference part in the report as [5].

import random
import os
import time

import cv2
import copy
import json

import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.measurements import label
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.utils import shuffle


class VeloEval(object):
    '''
    the class of read from JSON file in the given repo. I declare it in the reference part in the report as [5].
    '''

    @staticmethod
    def load_json_file(file_list):
        data_list = []
        for file_name in file_list:
            with open(file_name) as f:
                raw_data = json.load(f)
            data_list.append(raw_data)
        return data_list

    @staticmethod
    def transform_annotation(raw_data_list):
        anno_list = []
        for raw_data in raw_data_list:
            data = []
            for instance in raw_data:
                instance["bbox"] = np.array([[instance["bbox"]["top"],
                                              instance["bbox"]["left"],
                                              instance["bbox"]["bottom"],
                                              instance["bbox"]["right"]]])
                data.append(instance)
            anno_list.append(data)
        return anno_list

    @staticmethod
    def load_annotation(file_list):
        raw_data_list = VeloEval.load_json_file(file_list)
        anno_list = VeloEval.transform_annotation(raw_data_list)
        print("Finished loading {0:d} annotations.".format(len(anno_list)))
        return anno_list


def sigmoid(x):
    '''
    sigmoid
    :param x: 
    :return: sigmoid(x)
    '''
    return 1 / (1 + np.exp(-x))


def get_hog_features(img):
    '''
    extract features from three channels
    :param img: input image
    :return: the feature vector of the input image
    '''
    features0 = hog(img[:, :, 0], orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False,
                    feature_vector=True)
    features1 = hog(img[:, :, 1], orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False,
                    feature_vector=True)
    features2 = hog(img[:, :, 2], orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False,
                    feature_vector=True)
    features = np.concatenate((features0, features1, features2))
    return features


def get_color_histgram_features(img):
    '''
    extract features from three channels
    :param img: input image
    :return: the feature vector of the input image
    '''
    channel0 = np.histogram(img[:, :, 0], bins=32, range=(0, 256))
    channel1 = np.histogram(img[:, :, 1], bins=32, range=(0, 256))
    channel2 = np.histogram(img[:, :, 2], bins=32, range=(0, 256))
    features = np.concatenate((channel0[0], channel1[0], channel2[0]))
    return features


def get_spatial_features(img):
    '''
    extract features from three channels
    :param img: input image
    :return: the feature vector of the input image
    '''
    s0 = cv2.resize(img[:, :, 0], (32, 32)).ravel()
    s1 = cv2.resize(img[:, :, 1], (32, 32)).ravel()
    s2 = cv2.resize(img[:, :, 2], (32, 32)).ravel()
    features = np.concatenate((s0, s1, s2))
    return features


def get_features(img):
    '''
    extract features through three ways at the same time and then concatenate to the final feature vector
    :param img: input image
    :return: the final feature vector of the input image
    '''
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    hog_features = get_hog_features(img)
    color_features = get_color_histgram_features(img)
    spatial_features = get_spatial_features(img)
    features = np.concatenate((hog_features, color_features, spatial_features))
    return features


def search_cars(img, svc, x_scaler):
    '''
    search and detect cat in the image
    :param img: input image
    :param svc: classifier
    :param x_scaler: the scaler
    :return: labelled image, detected bounding boxes
    '''
    draw_img = copy.deepcopy(img)
    x_scales = [ [300, 500], [400, 600], [400, 700]]
    window__scales = [[64, 64], [64, 64], [128, 128]]
    strides = [ 16, 32, 64]
    test_bbox = []
    judge = False
    for i in range(len(x_scales)):
        temp = x_scales[i]
        stride = strides[i]
        x_window = window__scales[i][0]
        y_window = window__scales[i][1]
        x_steps = (temp[1] - temp[0]) // stride
        y_steps = 1280 // stride
        for row in range(x_steps):
            for col in range(y_steps):
                top = row * stride + temp[0]
                bottom = top + x_window
                left = col * stride
                right = left + y_window
                test_roi = img[top:bottom, left:right]
                test_roi = cv2.resize(test_roi, (64, 64))
                test_features = get_features(test_roi)
                test_features = np.array(test_features).reshape(1, -1)
                test_features = x_scaler.transform(test_features)
                result = svc.predict(test_features)
                # logits = sigmoid(svc.decision_function(test_features))
                if result == 1:
                    judge = True
                    tmp = [top, bottom, left, right]
                    cv2.rectangle(draw_img, (left, top), (right, bottom), (255, 0, 0), 2)
                    test_bbox.append(tmp)
    if judge == False:
        test_bbox.append([0, 0, 0, 0])
    return draw_img, test_bbox


def reduce_box(img, bbox_list, threshold=3):
    '''
    reduce the overlap boxes by heat map
    :param img: input image
    :param bbox_list: the detected bounding boxes
    :param threshold: the threshold to reduce the bounding boxes
    :return: the preprocessed heatmap
    '''
    heatmap = np.zeros_like(img[:, :]).astype(np.float64)

    for box in bbox_list:
        top = box[0]
        bottom = box[1]
        left = box[2]
        right = box[3]
        heatmap[top:bottom, left:right] += 1
    plt.imshow(heatmap)
    plt.savefig("heatmap.jpg")

    heatmap[heatmap <= threshold] = 0
    heatmap = np.clip(heatmap, 0, 255)
    return heatmap


def draw_labeled_bboxes(img, labels):
    '''
    draw the boxes on the image
    :param img: input image
    :param labels: the labels generated by the heat map
    :return: the labelled image
    '''
    for i in range(1, labels[1] + 1):
        nonzero = (labels[0] == i).nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])
        bbox = ((np.min(nonzero_x), np.min(nonzero_y)), (np.max(nonzero_x), np.max(nonzero_y)))
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    return img


def get_distance_metrics(groud_truth_bbox, test_bboxes):
    '''
    calculate distance between ground truth bounding boxes and detected bouding boxes
    :param groud_truth_bbox: ground truth bounding boxes
    :param test_bboxes: detected bouding boxes
    :return: the distance matric
    '''
    distance_metrics = [[0 for i in range(len(groud_truth_bbox))] for j in range(len(test_bboxes))]
    temp = groud_truth_bbox
    top_gt = int(temp[0][0])
    left_gt = int(temp[0][1])
    bottom_gt = int(temp[0][2])
    right_gt = int(temp[0][3])
    center_x_gt = (top_gt + bottom_gt) // 2
    center_y_gt = (left_gt + right_gt) // 2
    i = 0
    for test_bbox in test_bboxes:
        top_test = test_bbox[0]
        bottom_test = test_bbox[1]
        left_test = test_bbox[2]
        right_test = test_bbox[3]
        center_x_test = (top_test + bottom_test) // 2
        center_y_tess = (left_test + right_test) // 2
        distance = math.sqrt((center_x_gt - center_x_test) ** 2 + (center_y_gt - center_y_tess) ** 2)
        distance_metrics[i][0] = distance
        i += 1
    min_distance = np.min(distance_metrics[:])
    return distance_metrics, min_distance


def get_overlap_total_ratio_metrics(groud_truth, test_bboxes):
    '''
    the overlap ratio between ground truth bounding boxes and detected bouding boxes
    :param groud_truth: ground truth bounding boxes
    :param test_bboxes: detected bouding boxes
    :return: the overlap ratio metrics
    '''
    ratio_metrics = [[0 for i in range(len(groud_truth))] for j in range(len(test_bboxes))]
    top_groud_truth = int(groud_truth[0][0])
    left_groud_truth = int(groud_truth[0][1])
    bottom_groud_truth = int(groud_truth[0][2])
    right_groud_truth = int(groud_truth[0][3])
    i = 0
    for test_bbox in test_bboxes:
        top_test_bbox = int(test_bbox[0])
        bottom_test_bbox = int(test_bbox[1])
        left_test_bbox = int(test_bbox[2])
        right_test_bbox = int(test_bbox[3])
        diff_bottom = abs(bottom_test_bbox - bottom_groud_truth)
        diff_top = abs(top_test_bbox - top_groud_truth)
        diff_left = abs(left_test_bbox - left_groud_truth)
        diff_right = abs(right_test_bbox - right_groud_truth)
        area_1 = diff_bottom * diff_left
        area_2 = diff_top * diff_right
        max_area = (abs(min(top_groud_truth, top_test_bbox) - max(bottom_groud_truth, bottom_test_bbox))) * (
            abs(min(left_groud_truth, left_test_bbox) - max(right_groud_truth, right_test_bbox)))
        total_area = max_area - area_1 - area_2
        area_ground_truth = abs(top_groud_truth - bottom_groud_truth) * abs(left_groud_truth - right_groud_truth)
        area_test_bbox = abs(top_test_bbox - bottom_test_bbox) * abs(left_test_bbox - right_test_bbox)
        overlap_area = area_ground_truth + area_test_bbox - total_area
        ratio_metrics[i][0] = overlap_area / total_area
        i += 1
    max_ratio = np.max(ratio_metrics[:])
    return ratio_metrics, max_ratio


if __name__ == '__main__':
    '''
    Read images and json files
    '''
    path = "benchmark_velocity_train/clips"
    folder_path = list(map(int, os.listdir(path)))
    folder_path.sort()

    img_list = []
    for i in folder_path:
        temp_path = os.path.join(path, str(i) + "/imgs/040.jpg")
        img = cv2.imread(temp_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_list.append(img)

    annotations_list = [os.path.join(path, str(x), "annotation.json") for x in folder_path]
    ground_truth = VeloEval.load_annotation(annotations_list)

    '''
    Get the regions with cars and without cars
    '''
    img_number = len(annotations_list)
    X_roi, X_nocar, y_roi, y_nocar = [], [], [], []
    for i in range(img_number):
        temp = ground_truth[i][0]['bbox']
        labeled_img = img_list[i]
        top = int(temp[0][0])
        left = int(temp[0][1])
        bottom = int(temp[0][2])
        right = int(temp[0][3])
        # with cars
        roi_img = labeled_img[top:bottom, left:right]
        roi_img = cv2.resize(roi_img, (64, 64))
        X_roi.append(roi_img)
        y_roi.append(1)
        # no cars
        nocar_img = copy.deepcopy(labeled_img)
        nocar_img[top:bottom, left:right] = 0
        # if i % 2 == 0:
        top_no_car = random.randint(200, 700 - 64)
        left_no_car = random.randint(0, 1280 - 64)
        bottom_no_car = top_no_car + 64
        right_no_car = left_no_car + 64
        roi_img_no_car = nocar_img[top_no_car:bottom_no_car, left_no_car:right_no_car]
        # roi_img_no_car = cv2.resize(roi_img_no_car, (64, 64))
        X_nocar.append(roi_img_no_car)
        y_nocar.append(0)
        top_no_car = random.randint(200, 700 - 64)
        left_no_car = random.randint(0, 1280 - 64)
        bottom_no_car = top_no_car + 64
        right_no_car = left_no_car + 64
        roi_img_no_car = nocar_img[top_no_car:bottom_no_car, left_no_car:right_no_car]
        # roi_img_no_car = cv2.resize(roi_img_no_car, (64, 64))
        X_nocar.append(roi_img_no_car)
        y_nocar.append(0)
        top_no_car = random.randint(200, 700 - 32)
        left_no_car = random.randint(0, 1280 - 32)
        bottom_no_car = top_no_car + 32
        right_no_car = left_no_car + 32
        roi_img_no_car = nocar_img[top_no_car:bottom_no_car, left_no_car:right_no_car]
        roi_img_no_car = cv2.resize(roi_img_no_car, (64, 64))
        X_nocar.append(roi_img_no_car)
        y_nocar.append(0)
        top_no_car = random.randint(200, 700 - 128)
        left_no_car = random.randint(0, 1280 - 128)
        bottom_no_car = top_no_car + 128
        right_no_car = left_no_car + 128
        roi_img_no_car = nocar_img[top_no_car:bottom_no_car, left_no_car:right_no_car]
        roi_img_no_car = cv2.resize(roi_img_no_car, (64, 64))
        X_nocar.append(roi_img_no_car)
        y_nocar.append(0)


    print("car images: ", len(X_roi))
    print("no car images: ", len(X_nocar))

    '''
    Generate feature extraction vector
    '''
    features_roi_list, features_nocar_list = [], []
    for i in range(len(X_roi)):
        img_features = get_features(X_roi[i])
        features_roi_list.append(img_features)
    for i in range(len(X_nocar)):
        img_features = get_features(X_nocar[i])
        features_nocar_list.append(img_features)

    '''
    Divide training data and test data
    '''
    # features = np.r_[features_roi_list, features_nocar_list].astype(np.float64)
    features = np.vstack((features_roi_list, features_nocar_list)).astype(np.float64)
    # labels = np.r_[y_roi, y_nocar]
    labels = np.hstack((np.ones(len(y_roi)), np.zeros(len(y_nocar))))
    x_scaler = StandardScaler().fit(features)
    features = x_scaler.transform(features)
    features, labels = shuffle(features, labels, random_state=123)
    Xtrain, Xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.2, random_state=123)
    print("Feature vector length: ", len(features[0]))
    '''
    Generate classifier
    '''
    t = time.time()
    svc = svm.LinearSVC()
    svc.fit(Xtrain, ytrain)
    print("Training time: ", time.time() - t)
    '''
    Calculate the accuracy of the classifier
    '''
    train_accuracy = svc.score(Xtrain, ytrain)
    test_accuracy = svc.score(Xtest, ytest)
    print("train_accuracy: ", train_accuracy)
    print("test_accuracy: ", test_accuracy)

    # Test on one image
    # test_img_path = "benchmark_velocity_test/clips//imgs/040.jpg"
    # test_img = cv2.imread(test_img_path)
    # test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    # out_img, box_list = search_cars(test_img, svc, x_scaler)
    # plt.imshow(out_img)
    # plt.show()
    # '''
    # Search and detect car in the testing data set
    # '''
    test_path = "benchmark_velocity_test/clips"
    test_folder_path = list(map(int, os.listdir(test_path)))
    test_folder_path.sort()

    test_img_list = []
    for i in test_folder_path:
        temp_path = os.path.join(test_path, str(i) + "/imgs/040.jpg")
        img = cv2.imread(temp_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        test_img_list.append(img)
    test_img_bbox = []
    for test_img in test_img_list:
        out_img, box_list = search_cars(test_img, svc, x_scaler)
        test_img_bbox.append(box_list)
        heat = np.zeros_like(test_img[:, :, 0]).astype(np.float)
        heatmap = reduce_box(out_img, box_list, 2)
        labels = label(heatmap)
        draw_img = draw_labeled_bboxes(np.copy(test_img), labels)
        fig = plt.figure(figsize=(30, 20))
        plt.subplot(121)
        plt.imshow(out_img)
        plt.subplot(122)
        plt.imshow(draw_img, cmap='hot')
        fig.tight_layout()
        plt.savefig("result.jpg")
        plt.show()

    '''
    Evaluation
    '''
    test_annotations_list = [os.path.join(test_path, str(x), "annotation.json") for x in test_folder_path]
    test_ground_truth = VeloEval.load_annotation(test_annotations_list)

    for i in range(len(test_ground_truth)):
        # distance_metrics
        distance_metrics, min_distance = get_distance_metrics(test_ground_truth[i][0]['bbox'], test_img_bbox[i])
        print("distance_metrics", distance_metrics)
        print("min_distance", min_distance)
        # overlap ratios
        ratio_metrics, max_ratio = get_overlap_total_ratio_metrics(test_ground_truth[i][0]['bbox'], test_img_bbox[i])
        print("======")
        print("ratio_metrics", ratio_metrics)
        print("max_ratio", max_ratio)
