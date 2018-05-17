# -*- coding: utf-8 -*-
"""
Created on Wed May  2 16:36:09 2018

@author: Erick A. Montañez
"""

import sys
sys.path.append('images')

# Third party modules
import numpy
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

# Project modules
from extractor import extract_vectors
from validator import validate

# Extract data
train_features, train_labels = extract_vectors()
print "Training features: {}".format(numpy.array(train_features).shape)
print "Training labels: {}".format(numpy.array(train_labels).shape)

# Train the classifier
print "[STATUS] Creating the classifier.."
# classifier = LinearSVC(random_state=9)
classifier = KNeighborsClassifier(n_neighbors=5)

print "[STATUS] Fitting data/label to model.."
classifier.fit(train_features, train_labels)

# Validate the classifier
results = validate(classifier)

print "[RESULTS]"
print "{} validations".format(results['total_validations'])
print "{} successful".format(results['success_counter'])
print "{}% precise".format(results['precision'])
