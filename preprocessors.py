
import cv2

def grayscale(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale_image

def gaussian_blur(image):
    return cv2.GaussianBlur(image,(5,5),0)

def image_to_feature_vector(image, size=(32, 32)):
    return cv2.resize(image, size).flatten()
