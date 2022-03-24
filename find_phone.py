# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 17:06:39 2022

@author: Yash

Given the location of the training set, this script can evaluate the performance
of the model on the entire training set
"""

# import the necessary packages
# import config
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import numpy as np
import mimetypes
import argparse
import imutils
import cv2
import os
import re

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("input", help="path to the image to be tested")
args = vars(ap.parse_args())

# determine the input file type, but assume that we're working with
# single input image
# imagePaths = [args["input"]]

# if the file type is a text file, then we need to process *multiple*
# images
filenames = open(args["input"]).read().strip().split("\n")
imagePaths = []
test_targets = []

# loop over the filenames
for f in filenames:
    f = f.split(" ")
    # construct the full path to the image filename and then
    # update our image paths list
    p = os.path.sep.join(["find_phone_task_4/find_phone", f[0]])
    imagePaths.append(p)

    Xt = f[1]
    Yt = f[2]
    test_targets.append((Xt, Yt))

test_targets = np.array(test_targets, dtype="float32")
        
# load our trained model from disk
print("[INFO] loading model weights...")
model = load_model("training_output/model_parameters.h5")

counter = 0
score = 0

# loop over the images that we'll be testing using our regression model
for imagePath in imagePaths:
    # load the input image (in Keras format) from disk and preprocess
    # it, scaling the pixel intensities to the range [0, 1]
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
   
    # predict XY coordinates and true coordinates of phone in the input image
    preds = model.predict(image)[0]
    (X, Y) = preds
    (Xt, Yt) = test_targets[counter]
    
    if abs(X-Xt) <= 0.05 and abs(Y-Yt) <= 0.05:
        score += 1
   
   	# load the input image (in OpenCV format), and grab its dimensions
    image = cv2.imread(imagePath)
    (h, w) = image.shape[:2]
   
   	# scale the predicted and true coordinates based on the image
   	# dimensions
    X = int(X * w)
    Y = int(Y * h)
    
    Xt = int(Xt * w)
    Yt = int(Yt * h)
    
    counter += 1
       
    # plot the predicted coordinate and a circle of 0.05 normalized radius
    cv2.circle(image, (X, Y), radius=5, color=(255, 255, 255), thickness=-1)
    cv2.circle(image, (Xt, Yt), radius=16, color=(255, 255, 255), thickness=1)
       
    #show the output image
    # cv2.imshow("Prediction", image)
    # cv2.waitKey(0)
    
print(score)
accuracy = score / test_targets.shape[0]
print(accuracy)