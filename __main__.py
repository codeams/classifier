#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 16:36:09 2018

@author: codeams
"""
import cv2
import numpy as np
import os
import glob
import mahotas as mt
from sklearn.svm import LinearSVC


train_path = 'dataset/train'

train_names = os.listdir(train_path)

train_features = []
train_labels = []



def extract_features(image):

    textures = mt.features.haralick(image)

    ht_mean = textures.mean(axis=0)
    
    flatten_hu_moments_vector = cv2.HuMoments(cv2.moments(image)).flatten()
    
    features = (ht_mean, flatten_hu_moments_vector)
    
    return np.concatenate(features).ravel()



for train_name in train_names:
    if train_name == ".DS_Store":
        train_names.remove(train_name)
        
        
        
print "[STATUS] Started extracting Haralick features"

print train_names



i = 1

for train_name in train_names:

    cur_path = train_path + "/" + train_name

    cur_label = train_name

    i = 1


    for file in glob.glob(cur_path + "/*.jpg"):
        
        print "About to process image - {} in {}".format(i, cur_label)
        
        image = cv2.imread(file)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        features = extract_features(gray)
        
        train_features.append(features)
        
        train_labels.append(cur_label)
        
        i += 1

        
        

print "Training features: {}".format(np.array(train_features).shape)

print "Training labels: {}".format(np.array(train_labels).shape)



print "[STATUS] Creating the classifier.."

clf_svm = LinearSVC(random_state=9)



print "[STATUS] Fitting data/label to model.."

clf_svm.fit(train_features, train_labels)







# loop over the test images
test_path = "dataset/test"

for file in glob.glob(test_path + "/*.jpg"):

	# read the input image
	image = cv2.imread(file)

	# convert to grayscale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# extract haralick texture from the image
	features = extract_features(gray)

	# evaluate the model and predict label
	prediction = clf_svm.predict(features.reshape(1, -1))[0]

	# show the label
	cv2.putText(image, prediction, (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)
	print "Prediction - {}".format(prediction)

	# display the output image
	#cv2.imshow("Test_Image", image)
	#cv2.waitKey(0)
    
    
    



