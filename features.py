
import cv2
import imutils
import mahotas

from preprocessors import grayscale, image_to_feature_vector


def vector(image):
    return image_to_feature_vector(image)


def color_histogram(image, bins=(8, 8, 8)):
    # extract a 3D color histogram from the HSV color space using
    # the supplied number of `bins` per channel
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
        [0, 180, 0, 256, 0, 256])

    # handle normalizing the histogram if we are using OpenCV 2.4.X
    if imutils.is_cv2():
        hist = cv2.normalize(hist)

    # otherwise, perform "in place" normalization in OpenCV 3 (I
    # personally hate the way this is done
    else:
        cv2.normalize(hist, hist)

    # return the flattened histogram as the feature vector
    return hist.flatten()


def entropy(image):
    entropy = mahotas.features.haralick(grayscale(image))
    entropy_mean = entropy.mean(axis=0)
    return entropy_mean


def haralick(image):
    textures = mahotas.features.haralick(grayscale(image))
    textures_mean = textures.mean(axis=0)
    return textures_mean


def hu_moments(image):
    moments = cv2.moments(grayscale(image))
    hu_moments = cv2.HuMoments(moments)
    hu_moments_vector = hu_moments.flatten()
    return hu_moments_vector
