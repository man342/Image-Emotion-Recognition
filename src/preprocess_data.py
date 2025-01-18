import os
import pandas as pd
import numpy as np
from keras_preprocessing.image import load_img

# Function to create a DataFrame with image paths and labels
def create_dataframe(directory):
    image_paths = []
    labels = []
    for label in os.listdir(directory):
        label_path = os.path.join(directory, label)
        if os.path.isdir(label_path):
            for image_file in os.listdir(label_path):
                image_paths.append(os.path.join(label_path, image_file))
                labels.append(label)
    return pd.DataFrame({'image': image_paths, 'label': labels})

# Function to preprocess images: resize, normalize, and reshape
def preprocess_images(image_paths, image_size=(48, 48)):
    features = []
    for image_path in image_paths:
        img = load_img(image_path, target_size=image_size, color_mode='grayscale')
        img_array = np.array(img).reshape(*image_size, 1) / 255.0
        features.append(img_array)
    return np.array(features)

# Split data into training and testing datasets (if needed)
def split_data(df, test_ratio=0.2):
    test_size = int(len(df) * test_ratio)
    test_data = df.sample(n=test_size, random_state=42)
    train_data = df.drop(test_data.index)
    return train_data, test_data
