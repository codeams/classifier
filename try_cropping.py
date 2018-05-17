
import sys
sys.path.append('images')

'''
import cv2 as cv
from matplotlib import pyplot

from classifier import get_classifier, direct_classify, classify

images_classifier = get_classifier()
audio_prediction = 'banana'

contours_image_matched = []
image_path = 'trial.jpg'
image = cv.imread(image_path, 0)

blur = cv.GaussianBlur(image, (5, 5), 0)
ret, thresh = cv.threshold(blur, 127, 255, 0)

im2, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
order_contours = sorted(contours, key=cv.contourArea, reverse=True) # [:4]

filtered_images = []

for index_contour in range(0, len(order_contours) - 1):
    if index_contour < 1:
        continue

    x, y, w, h = cv.boundingRect(order_contours[index_contour])

    min_size = 100
    if w > min_size and h > min_size:
        print w
        print h

        crop_img = blur[y - 3:y + h + 3, x - 3:x + w + 3]
        image_resized = cv.resize(crop_img, (80, 100))
        filtered_images.append(image_resized)


for filtered_image in filtered_images:
    filtered_image_clean = cv.cvtColor(filtered_image, cv.COLOR_GRAY2RGB)
    prediction = direct_classify(filtered_image_clean)
    print prediction
    pyplot.imshow(filtered_image_clean)
    pyplot.show()
'''


'''
import cv2
import numpy as np
from matplotlib import pyplot

from classifier import get_classifier, direct_classify, classify

image = cv2.imread('trial.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, bw = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# find connected components
connectivity = 8
nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity, cv2.CV_32S)
sizes = stats[1:, -1]
nb_components = nb_components - 1
min_size = 250  # threshhold value for objects in scene
img2 = np.zeros((image.shape), np.uint8)
for i in range(0, nb_components):
    # use if sizes[i] >= min_size: to identify your objects
    if sizes[i] >= min_size:
        color = np.random.randint(255, size=3)
        # draw the bounding rectangele around each object
        cv2.rectangle(img2, (stats[i][0],stats[i][1]),(stats[i][0]+stats[i][2],stats[i][1]+stats[i][3]), (0,255,0), 2)
        img2[output == i + 1] = color

        pyplot.imshow(img2)
        pyplot.show()
'''

'''
import cv2
from matplotlib import pyplot

# reading the image
image = cv2.imread("trial.jpg")
edged = cv2.Canny(image, 10, 250)
# cv2.imshow("Edges", edged)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.waitKey(1)

# applying closing function
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
# cv2.imshow("Closed", closed)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.waitKey(1)

# finding_contours
im2, cnts, hierarchy = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    if w > 150 and h > 150:
        new_img = image[y:y + h, x:x + w]
        pyplot.imshow(new_img)
        pyplot.show()
    # peri = cv2.arcLength(c, True)
    # approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    # cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
# cv2.imshow("Output", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.waitKey(1)
'''

from classifier import direct_classify
from harvester import harvest
from matplotlib import pyplot
fruits = harvest('trial.jpg')

audio_prediction = 'apple'

matches = []

for fruit in fruits:
    prediction = direct_classify(fruit)
    if prediction == audio_prediction:
        matches.append(fruit)

for match in matches:
    pyplot.imshow(match)
    pyplot.show()
