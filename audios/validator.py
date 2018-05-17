
import glob
from os import listdir
from scipy.io import wavfile

import config
from extractor import extract_features
from utils import remove_system_files


def validate(classifier):
    success_counter = 0
    fails_counter = 0

    validate_names = listdir(config.VALIDATE_PATH)
    remove_system_files(validate_names)

    print "[STATUS] Started validating classifier effectiveness"
    print validate_names
    print "Prediction: predicted label / real label"

    for validate_name in validate_names:
        current_path = config.VALIDATE_PATH + '/' + validate_name
        current_label = validate_name

        for file_path in glob.glob(current_path + '/*.' + config.FILES_EXTENSION):
            features = extract_features(file_path)
            prediction = classifier.predict(features.reshape(1, -1))[0]

            if config.PRINT_PREDICTIONS:
                print "Prediction: {} / {}".format(prediction, current_label)

            if prediction == current_label:
                success_counter += 1
            else:
                fails_counter += 1
                if config.DEBUG_MODE:
                    print "Debug mode does nothing on this repo."
    return {
        'success_counter': success_counter,
        'fails_counter': fails_counter,
        'total_validations': success_counter + fails_counter,
        'precision': (success_counter / float(success_counter + fails_counter)) * 100
    }
