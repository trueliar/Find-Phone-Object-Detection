# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 16:11:45 2022

@author: Yash
"""

# import the necessary packages
import os

# define the base path to the input dataset and then use it to derive
# the path to the images directory and annotation CSV file
BASE_PATH = "find_phone_task_4"
IMAGES_PATH = os.path.sep.join([BASE_PATH, "find_phone"])
ANNOTS_PATH = os.path.sep.join([IMAGES_PATH, "labels.txt"])

# define the path to the base output directory
BASE_OUTPUT = "Output"

# define the path to the output serialized model, model training plot,
# and testing image filenames
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "detector.h5"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
TEST_FILENAMES = os.path.sep.join([BASE_OUTPUT, "test_images.txt"])
TEST_LABELS = os.path.sep.join([BASE_OUTPUT, "test_labels.txt"])

# initialize our initial learning rate, number of epochs to train
# for, and the batch size
learning_rate = 1e-3
NUM_EPOCHS = 20
BATCH_SIZE = 32