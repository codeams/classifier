
import cv2
from matplotlib import pyplot


def harvest(file_path):
    image = cv2.imread(file_path)
    edged = cv2.Canny(image, 10, 250)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    _, contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        min_size = 150
        x, y, w, h = cv2.boundingRect(contour)
        if w > min_size and h > min_size:
            fruit = image[y : y + h, x : x + w]
            pyplot.imshow(fruit)
            pyplot.show()
