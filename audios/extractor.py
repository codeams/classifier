
import glob
import numpy
import os
import librosa
from scipy.io import wavfile

import config
from features import peaks_indexes, peaks_count, spectral_bandwidth
from utils import remove_system_files


def extract_features(file_path):
    # print spectral_bandwidth(file_path)[0]
    # exit()

    features = (
        spectral_bandwidth(file_path)[0],
        ()  # This allows the return statement to concatenate even
            # if we're only using one descriptor
    )
    return numpy.hstack(features)
    # return numpy.concatenate(features).ravel()


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
            # sample_rate, data = wavfile.read(file_path)
            features = extract_features(file_path)

            train_features.append(features)
            train_labels.append(cur_label)

    return train_features, train_labels
