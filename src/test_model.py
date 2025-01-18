from keras.models import model_from_json
from keras_preprocessing.image import load_img
import numpy as np


json_file = open('models/emotiondetector.json', 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights('models/emotiondetector.h5')


labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


image_path = 'data/test/some_image.png'  # Example image path
img = load_img(image_path, grayscale=True)
img = np.array(img).reshape(1, 48, 48, 1) / 255.0
prediction = model.predict(img)
print(f'Predicted emotion: {labels[np.argmax(prediction)]}')
