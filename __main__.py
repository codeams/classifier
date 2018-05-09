#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 16:36:09 2018

@author: codeams
"""

# Third party modules
import numpy
from sklearn.svm import LinearSVC

# Project modules
from extractor import extract_vectors
from validator import validate

### Extract data ###
train_data = extract_vectors()
train_features = train_data['features']
train_labels = train_data['labels']
print "Training features: {}".format(numpy.array(train_features).shape)
print "Training labels: {}".format(numpy.array(train_labels).shape)

### Train the classifier ###
print "[STATUS] Creating the classifier.."
svm_classifier = LinearSVC(random_state=9)

print "[STATUS] Fitting data/label to model.."
svm_classifier.fit(train_features, train_labels)

### Validate the classifier ###
results = validate(svm_classifier)

print "[RESULTS]"
print "{} validations".format(results['total_validations'])
print "{} successful".format(results['success_counter'])
print "{}% precise".format(results['precision'])
