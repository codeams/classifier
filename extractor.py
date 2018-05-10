
import cv2
import glob
import numpy
import os

import config
from features import entropy, color_histogram, haralick, hu_moments, vector
from utils import remove_system_files


def extract_features(image):
    if config.SCALE_IMAGE:
        cv2.resize(image, (45, 45))

    features = (
        # haralick(image),
        color_histogram(image),
        hu_moments(image),
        entropy(image),
        vector(image),
        ()  # This allows the return statement to concatenate even
            # if we're only using one descriptor
    )
    return numpy.concatenate(features).ravel()


def extract_vectors():
    train_features = []
    train_labels = []

    train_names = os.listdir(config.TRAIN_PATH)
    remove_system_files(train_names)

    print "[STATUS] Started extracting features"
    print train_names

    for train_name in train_names:
        cur_path = config.TRAIN_PATH + "/" + train_name
        cur_label = train_name

        for file in glob.glob(cur_path + "/*." + config.IMAGES_EXTENSION):
            print "About to process image in {}".format(cur_label)
            image = cv2.imread(file)
            features = extract_features(image)
            train_features.append(features)
            train_labels.append(cur_label)

    return {
        'features': train_features,
        'labels': train_labels
    }
