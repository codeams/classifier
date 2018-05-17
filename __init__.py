
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

from classifier import get_classifier, classify


app = Flask(__name__)
classifier = get_classifier()

@app.route('/')
def index_route():
    return render_template('index.html')


@app.route('/classify', methods=['POST'])
def classify_route():
    image = request.files['image']
    image_name = secure_filename(image.filename)
    image.save('uploads/' + image_name)

    prediction = classify('uploads/' + image_name, classifier=classifier)
    return render_template('classify.html', prediction=prediction)
