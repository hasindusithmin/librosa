import io
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import numpy as np
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
import librosa

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load the save model
model = tf.keras.models.load_model('keras_model.h5')


def extract_features(audio_data, sample_rate):
    # Placeholder for the features
    features = []
    # Extract features using librosa or any other library
    # Append the extracted features to the list
    return np.array(features)

@app.post("/")
async def upload_file(file: UploadFile):
    content = await file.read()
    # Load the audio data from the BytesIO object
    audio_data, sample_rate = librosa.load(io.BytesIO(content), sr=None)
    # Extract features from the audio data
    features = extract_features(audio_data, sample_rate)
    # Reshape the features to match the model's input shape
    features = np.expand_dims(features, axis=0)
    # Predict the class of the audio
    label = ['class_1','class_2','class_3']
    prediction = model.predict(features)
    return list(zip(label, prediction.tolist()[0]))


