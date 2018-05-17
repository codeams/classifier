
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

from .images.classifier import get_classifier as get_images_classifier
from .images.classifier import classify as classify_image
from .audios.classifier import get_classifier as get_audios_classifier
from .audios.classifier import classify as classify_audio
from translator import translate_label


app = Flask(__name__)
images_classifier = get_images_classifier()
audios_classifier = get_audios_classifier()


@app.route('/', methods=['GET'])
def index_route():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def classify_route():
    audio = request.files['audio']
    audio_name = secure_filename(audio.filename)
    audio.save('uploads/' + audio_name)
    prediction = classify_audio('uploads/' + audio_name, classifier=audios_classifier)

    # image = request.files['image']
    # image_name = secure_filename(image.filename)
    # image.save('uploads/' + image_name)
    # prediction = classify_image('uploads/' + image_name, classifier=classifier)
    prediction = translate_label(prediction)
    return render_template('index.html', prediction=prediction)
