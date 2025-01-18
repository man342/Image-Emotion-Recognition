from flask import Flask, render_template, request
from keras.models import model_from_json
from keras_preprocessing.image import load_img
import numpy as np

app = Flask(__name__)

# Load the model
with open('models/emotiondetector.json', 'r') as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)
model.load_weights('models/emotiondetector.h5')

# Define emotion labels
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded', 400

    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    img = load_img(file, target_size=(48, 48), color_mode='grayscale')
    img_array = np.array(img).reshape(1, 48, 48, 1) / 255.0
    prediction = model.predict(img_array)
    emotion = labels[np.argmax(prediction)]
    return f'Predicted Emotion: {emotion}'

if __name__ == '__main__':
    app.run(debug=True)
