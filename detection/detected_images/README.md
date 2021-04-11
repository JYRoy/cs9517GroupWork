# Introduction

This folder has the detected training images and detected testing images labels. Since it is a big file with images, it only includes the labels.

# Format

## Folder Organisation

- detected_images
  - detected_train
    - labels
  - detected_test
    - labels

## Naming

 Labels and detected images have same way of naming, {the number of clips}_{the number of images}.{the type of files}. Each labelled images has corresponding txt file which records the bounding boxes. 

For example:

- 1_00.txt: the labels for the first image of the first clips
- 1_040.txt: the labels for the last image of the first clips

## The Format of the txt

Each txt records the bounding boxes in the image. Each bounding box uses one line to record the arguments values. Each line has five columes. The first line means the object class of the bounding box. in this task, it only has one class, car, the class number is zero. The second column is the x center, the third column is the y center, the fourth column is the width, the fifth column is the height. Noting that box coordinates are normalized from 0 - 1 by the size of the images (1280 * 720).

![](https://user-images.githubusercontent.com/26833433/91506361-c7965000-e886-11ea-8291-c72b98c25eec.jpg)