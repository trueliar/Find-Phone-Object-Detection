# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 16:22:06 2022

@author: Yash
"""

# import config
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import apply_affine_transform
from tensorflow.image import random_saturation
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math as m
import numpy as np
import cv2
import os
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("input", help="path to the folder with labeled images")
args = vars(ap.parse_args())

# define the base path to the input dataset and then use it to derive
# the path to the images directory and annotation CSV file
BASE_PATH = args["input"]
IMAGES_PATH = BASE_PATH
ANNOTS_PATH = os.path.sep.join([IMAGES_PATH, "labels.txt"])

# define the path to the base output directory
BASE_OUTPUT = "training_output"

# define the path to the output serialized model, model training plot,
# and testing image filenames
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "model_parameters.h5"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
TEST_FILENAMES = os.path.sep.join([BASE_OUTPUT, "test_images.txt"])
TEST_LABELS = os.path.sep.join([BASE_OUTPUT, "test_labels.txt"])

# initialize our initial learning rate, number of epochs to train
# for, and the batch size
learning_rate = 1e-3
NUM_EPOCHS = 20
BATCH_SIZE = 32

# load the contents of the labels.txt file
print("[INFO] loading dataset...")
rows = open(ANNOTS_PATH).read().strip().split("\n")

# initialize the list of data (images), our target output predictions
# (XY coordinates), along with the filenames of the
# individual images
data = []
targets = []
filenames = []

# loop over the rows
for row in rows:
    # break the row into the filename and XY coordinates
    row = row.split(" ")
    (filename, X, Y) = row
    
    # derive the path to the input image, load the image (in OpenCV
	# format), and grab its dimensions
    imagePath = os.path.sep.join([IMAGES_PATH, filename])
    image = cv2.imread(imagePath)
    (h, w) = image.shape[:2]

	# Convert to floating point numbers
    X = float(X)
    Y = float(Y)
    
    # load the image and preprocess it
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)

	# update our list of data, targets, and filenames
    data.append(image)
    targets.append((X, Y))
    filenames.append(filename)
    
    filename_aug = filename.split(".")
    
    # flip vertically
    image_flip = cv2.flip(image, 0)
   
   	# update our list of data, targets, and filenames
    data.append(image_flip)
    targets.append((X, 1-Y))
    filenames.append(filename_aug[0]+"-1."+filename_aug[1])
    
    # flip horizontally
    image_flip = cv2.flip(image, 1)
    # update our list of data, targets, and filenames
    data.append(image_flip)
    targets.append((1-X, Y))
    filenames.append(filename_aug[0]+"-2."+filename_aug[1])
    
    # adjust saturation by a random factor
    image_sat = random_saturation(image, 5, 10)
    
    # update our list of data, targets, and filenames
    data.append(image_sat)
    targets.append((X, Y))
    filenames.append(filename_aug[0]+"-3."+filename_aug[1])
   
    # flip horizontally and vertically
    image_flip = cv2.flip(image, -1)
    # update our list of data, targets, and filenames
    data.append(image_flip)
    targets.append((1-X, 1-Y))
    filenames.append(filename_aug[0]+"-4."+filename_aug[1])
    
    # rotate image by a small random number
    th = np.random.randn() * m.pi / 180
    image_rot = apply_affine_transform(image, theta=th, fill_mode='nearest', order=1)
    Xr = m.cos(th) * X + m.sin(th) * Y
    Yr = -m.sin(th) * X + m.cos(th) * Y
    
    # update our list of data, targets, and filenames
    data.append(image_rot)
    targets.append((Xr, Yr))
    filenames.append(filename_aug[0]+"-5."+filename_aug[1])
    
# convert the data and targets to NumPy arrays, scaling the input
# pixel intensities from the range [0, 255] to [0, 1]
data = np.array(data, dtype="float32") / 255.0
targets = np.array(targets, dtype="float32")

# partition the data into training and testing splits using 90% of
# the data for training and the remaining 10% for testing
split = train_test_split(data, targets, filenames, test_size=0.10)

# unpack the data split
(trainImages, testImages) = split[:2]
(trainTargets, testTargets) = split[2:4]
(trainFilenames, testFilenames) = split[4:]

print(trainImages.shape)

# write the testing filenames to disk so that we can use then
# when evaluating/testing our regressor
print("[INFO] saving testing filenames...")
f = open(TEST_FILENAMES, "w")
f.write("\n".join(testFilenames))
f.close()

# write the testing labels to disk so that we can use then
# when evaluating/testing our regressor
print("[INFO] saving testing labels...")
f = open(TEST_LABELS, "w")
f.write("\n".join(str(label) for label in testTargets))
f.close()

# load the VGG16 network, ensuring the head FC layers are left off
vgg = VGG16(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# freeze all VGG layers so they will *not* be updated during the
# training process
vgg.trainable = False

# flatten the max-pooling output of VGG
flatten = vgg.output
flatten = Flatten()(flatten)

# construct a fully-connected layer header to output the predicted
# XY coordinates
bboxHead = Dense(128, activation="linear")(flatten)
bboxHead = BatchNormalization(momentum=0.99, epsilon=1e-4)(bboxHead)
bboxHead = ReLU()(bboxHead)
bboxHead = Dense(64, activation="linear")(bboxHead)
bboxHead = BatchNormalization(momentum=0.99, epsilon=1e-4)(bboxHead)
bboxHead = ReLU()(bboxHead)
bboxHead = Dense(32, activation="linear")(bboxHead)
bboxHead = BatchNormalization(momentum=0.99, epsilon=1e-4)(bboxHead)
bboxHead = ReLU()(bboxHead)
bboxHead = Dense(4, activation="linear")(bboxHead)
bboxHead = Dense(2, activation="sigmoid")(bboxHead)

# construct the model we will fine-tune for regression
model = Model(inputs=vgg.input, outputs=bboxHead)

# initialize the optimizer, compile the model, and show the model summary
opt = Adam(lr=learning_rate)
model.compile(loss="mse", optimizer=opt)
print(model.summary())

# train the network for regression
print("[INFO] training the model...")
H = model.fit(
	trainImages, trainTargets,
	validation_data=(testImages, testTargets),
	batch_size=BATCH_SIZE,
	epochs=NUM_EPOCHS,
	verbose=1)

# serialize the model to disk
print("[INFO] saving model weights...")
model.save(MODEL_PATH, save_format="h5")

# plot the model training history
N = NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.title("MSE Loss on Training Set")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(PLOT_PATH)