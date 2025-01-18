from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras_preprocessing.image import load_img
import os
import pandas as pd
import numpy as np

TRAIN_DIR = 'data/train'
TEST_DIR = 'data/test'

# Create dataframes for images and labels
def create_dataframe(dir):
    image_paths = []
    labels = []
    for label in os.listdir(dir):
        for imagename in os.listdir(os.path.join(dir, label)):
            image_paths.append(os.path.join(dir, label, imagename))
            labels.append(label)
    return pd.DataFrame({'image': image_paths, 'label': labels})

# Preprocess images
def preprocess_images(image_paths):
    features = []
    for image in image_paths:
        img = load_img(image, grayscale=True)
        img = np.array(img).reshape(48, 48, 1)
        features.append(img)
    return np.array(features) / 255.0


train_df = create_dataframe(TRAIN_DIR)
test_df = create_dataframe(TEST_DIR)

x_train = preprocess_images(train_df['image'])
x_test = preprocess_images(test_df['image'])


from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

y_train = to_categorical(le.fit_transform(train_df['label']))
y_test = to_categorical(le.transform(test_df['label']))

# Build CNN model
model = Sequential([
    Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.4),
    Conv2D(256, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.4),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(7, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50, batch_size=128)


model_json = model.to_json()
with open('models/emotiondetector.json', 'w') as json_file:
    json_file.write(model_json)
model.save('models/emotiondetector.h5')
