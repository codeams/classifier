
import cv2
from sklearn.svm import LinearSVC

from extractor import extract_vectors, extract_features


def get_classifier():
    train_features, train_labels = extract_vectors()
    svm_classifier = LinearSVC(random_state=9)
    svm_classifier.fit(train_features, train_labels)
    return svm_classifier


def classify(file_path, classifier=get_classifier()):
    image = cv2.imread(file_path)
    features = extract_features(image)
    prediction = classifier.predict(features.reshape(1, -1))[0]
    return prediction


if __name__ == '__main__':
    print classify('uploads/avocado.jpg')
