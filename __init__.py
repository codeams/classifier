
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

from .images.classifier import get_classifier as get_images_classifier
from .images.classifier import classify as classify_image
from translator import translate_label


app = Flask(__name__)
classifier = get_images_classifier()


@app.route('/', methods=['GET'])
def index_route():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def classify_route():
    image = request.files['image']
    image_name = secure_filename(image.filename)
    image.save('uploads/' + image_name)

    prediction = classify_image('uploads/' + image_name, classifier=classifier)
    prediction = translate_label(prediction)
    return render_template('index.html', prediction=prediction)
