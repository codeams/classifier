
import glob, cv2

import config
from extractor import extract_features

def validate(classifier):
    ### Test the classifier ###
    test_path = config.TEST_PATH

    # loop over the test images
    for file in glob.glob(test_path + "/*.jpg"):
        # read the input image
        image = cv2.imread(file)

        # Extract the selected features from the image
        features = extract_features(image)

        # evaluate the model and predict label
        prediction = classifier.predict(features.reshape(1, -1))[0]

        # show the label
        cv2.putText(image, prediction, (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)
        print "Prediction - {}".format(prediction)
