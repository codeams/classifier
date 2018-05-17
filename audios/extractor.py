
import glob
import numpy
import os

import config
from features import spectral_bandwidth, peaks_indexes, peaks_count
from utils import remove_system_files


def extract_features(file_path):
    features = (
        peaks_indexes(file_path),
        peaks_count(file_path),
        spectral_bandwidth(file_path),
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

        for file_path in glob.glob(cur_path + "/*." + config.FILES_EXTENSION):
            print "About to process audio in {}".format(cur_label)
            features = extract_features(file_path)

            train_features.append(features)
            train_labels.append(cur_label)

    return train_features, train_labels
