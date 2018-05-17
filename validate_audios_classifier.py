
import sys
sys.path.append('audios')

# Third party modules
import numpy
from sklearn.svm import LinearSVC

# Project modules
from extractor import extract_vectors
from validator import validate

# Extract data
train_features, train_labels = extract_vectors()
print "Training features: {}".format(numpy.array(train_features).shape)
print "Training labels: {}".format(numpy.array(train_labels).shape)

# Train the classifier
print "[STATUS] Creating the classifier.."
svm_classifier = LinearSVC(random_state=42)

# Needed to check feature length when the audio
# length wasn't the same and I was getting a "valueType"
# error
#
# for train_feature in train_features:
#    print len(train_feature)
# exit()

print "[STATUS] Fitting data/label to model.."
svm_classifier.fit(train_features, train_labels)

# Validate the classifier
results = validate(svm_classifier)

print "[RESULTS]"
print "{} validations".format(results['total_validations'])
print "{} successful".format(results['success_counter'])
print "{}% precise".format(results['precision'])
