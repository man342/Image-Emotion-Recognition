# Image Emotion Recognition

This repository contains the implementation and documentation for the project **Image Emotion Recognition**. It uses deep learning techniques, particularly Convolutional Neural Networks (CNNs), to classify facial expressions into various emotions such as Happy, Sad, Angry, Fear, Surprise, Disgust, and Neutral. Additionally, a Flask-based web application is provided for real-time interaction with the trained model.


## Features
- **Emotion Classification**: Classify images into seven emotion categories: Happy, Sad, Angry, Fear, Surprise, Disgust, and Neutral.
- **Deep Learning Model**: Built using Convolutional Neural Networks (CNNs) trained on the FER-2013 dataset.
- **Web Application**: Real-time emotion detection using a user-friendly Flask web interface.
- **Visualization**: Accuracy and loss graphs provided to analyze model performance.

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/image-emotion-recognition.git
cd image-emotion-recognition
```


### 2. Dataset Setup
The project uses the **FER-2013** dataset, which can be downloaded from [Kaggle](https://www.kaggle.com/msambare/fer2013). Place the dataset in the `data/` directory as follows:
```plaintext
data/
├── train/
└── test/
```

### 3. Train the Model
To train the CNN model, run the following script:
```bash
python src/train_model.py
```

### 4. Test the Model
To test the model on the test dataset, use:
```bash
python src/test_model.py
```

### 5. Run the Web Application
Start the Flask web app for real-time emotion detection:
```bash
python src/web_app.py
```
Visit `http://127.0.0.1:5000` in your browser to upload images and view predictions.

## Results
The results, including accuracy and loss graphs, are saved in the `results/` directory. Example:
```plaintext
results/
├── accuracy_graph.png
├── loss_graph.png
```

- **Accuracy Graph**:
  ![Accuracy](results/accuracy_graph.png)

- **Loss Graph**:
  ![Loss](results/loss_graph.png)

## Key Scripts

### **`src/train_model.py`**
- Trains the CNN model using the FER-2013 dataset.
- Saves the trained model as `emotiondetector.json` and `emotiondetector.h5` in the `models/` directory.

### **`src/test_model.py`**
- Loads the trained model and evaluates it on the test dataset.
- Outputs the classification performance.

### **`src/preprocess_data.py`**
- Handles preprocessing tasks, such as resizing, normalization, and data splitting.

### **`src/web_app.py`**
- Provides a Flask web interface for uploading images and predicting emotions.
- Accepts images via a simple HTML form and returns predictions in real-time.

### **`templates/index.html`**
- HTML template for the Flask web interface.
- Allows users to upload images and view emotion predictions.

## Future Enhancements
- Add real-time webcam-based emotion detection.
- Implement more advanced CNN architectures, such as ResNet or Inception.
- Improve the web interface with better design and usability.


## References
- [FER-2013 Dataset on Kaggle](https://www.kaggle.com/msambare/fer2013)
- Deep learning frameworks: [TensorFlow](https://www.tensorflow.org/), [Keras](https://keras.io/).
