
import base64
import cv2 as cv
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

from .images.harvester import look_for
from .audios.classifier import classify as classify_audio
from translator import translate_label


app = Flask(__name__)


@app.route('/', methods=['GET'])
def index_route():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def classify_route():
    audio = request.files['audio']
    audio_name = secure_filename(audio.filename)
    audio.save('uploads/' + audio_name)
    prediction = classify_audio('uploads/' + audio_name)

    harvestable_image = request.files['image']
    harvestable_image_name = secure_filename(harvestable_image.filename)
    harvestable_path = 'uploads/' + harvestable_image_name
    harvestable_image.save(harvestable_path)
    matches = look_for(prediction, harvestable_path)

    matches_amount = len(matches)
    encoded_images = []

    for match in matches:
        match = cv.cvtColor(match, cv.COLOR_BGR2RGB)
        _, buffer = cv.imencode('.jpg', match)
        processed_string = base64.b64encode(buffer)
        encoded_images.append(processed_string)

    prediction = translate_label(prediction)
    return render_template('index.html', prediction=prediction, matches_amount=matches_amount, images=encoded_images)

