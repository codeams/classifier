
from sklearn.svm import LinearSVC
# from sklearn.neighbors import KNeighborsClassifier

from extractor import extract_vectors, extract_features


def get_classifier():
    train_features, train_labels = extract_vectors()
    classifier = LinearSVC(random_state=9)
    # classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(train_features, train_labels)
    return classifier


def classify(file_path, classifier=get_classifier()):
    features = extract_features(file_path)
    prediction = classifier.predict(features.reshape(1, -1))[0]
    return prediction


if __name__ == '__main__':
    print classify('uploads/avocado.jpg')
