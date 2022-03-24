# Find-Phone-Object-Detection
Given a set of training data, this code trains a DL model that can find the XY location of a phone lying on the floor 

The training data is of the following type: a set of ~ 100 images, each of which has a phone lying on the floor. The ground truth is given in the labels.txt file which contains normalized coordinates of the phone in each picture. Each line is of the following format:  
*img_path, x(coordinate of the phone), y(coordinate of the phone)*

## config.py 
Defines all paths and hyperparameters like the learning rate, batch size, no. of epochs.  
## train_phone_finder.py 
Takes the training images as input and trains the model.  
## find_phone.py 
Loads the trained model, and given a test image, finds the normalized coordinates of the phone in that image. The uploaded version has been specifically developed to calculate the training accuracy on the entire training set, but you can easily tweak it to find the location of the phone in any test image.
