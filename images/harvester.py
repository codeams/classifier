
import cv2
from matplotlib import pyplot
from classifier import direct_classify


def look_for(fruit_name, file_path):
    fruits = harvest(file_path)
    matches = []

    for fruit in fruits:
        prediction = direct_classify(fruit)
        if prediction == fruit_name:
            matches.append(fruit)

    return matches


def harvest(file_path):
    image = cv2.imread(file_path)
    edged = cv2.Canny(image, 10, 250)
    pyplot.imshow(edged)
    # pyplot.show()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    pyplot.imshow(closed)
    # pyplot.show()

    _, contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fruits = []

    for contour in contours:
        min_size = 150
        x, y, w, h = cv2.boundingRect(contour)
        if w > min_size and h > min_size:
            fruit = image[y: y + h, x: x + w]
            color_fruit = cv2.cvtColor(fruit, cv2.COLOR_BGR2RGB)
            fruits.append(color_fruit)
            # pyplot.imshow(color_fruit)
            # pyplot.show()

    return fruits


if __name__ == '__main__':
    harvest('../trial.jpg')

