
import glob, cv2, os

import config
from extractor import extract_features
from utils import remove_system_files

def validate(classifier):
    success_counter = 0
    fails_counter = 0

    validate_names = os.listdir(config.VALIDATE_PATH)
    remove_system_files(validate_names)
    
    print "[STATUS] Started validating classifier effectiveness"
    print validate_names

    for validate_name in validate_names:
        current_path = config.VALIDATE_PATH + '/' + validate_name
        current_label = validate_name

        for file in glob.glob(current_path + '/*.' + config.IMAGES_EXTENSION):
            image = cv2.imread(file)
            features = extract_features(image)
            prediction = classifier.predict(features.reshape(1, -1))[0]

            # cv2.putText(image, prediction, (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)
            # print "Prediction: {} / {}".format(prediction, current_label)

            if (prediction == current_label):
                success_counter += 1
            else:
                fails_counter += 1

    return {
        'success_counter': success_counter,
        'fails_counter': fails_counter,
        'total_validations': success_counter + fails_counter,
        'precision': (success_counter / float(success_counter + fails_counter)) * 100
    }
