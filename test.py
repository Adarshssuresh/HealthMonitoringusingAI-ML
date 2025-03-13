from fastapi import FastAPI
import numpy as np
import tensorflow as tf
from pydantic import BaseModel

model = tf.keras.models.load_model("heart_monitor_lstm.h5")
scaler_mean = np.load("scaler.npy")
print(scaler_mean)


app = FastAPI()
latest_prediction = {"status": "Waiting...", "confidence": 0.0}

class HeartData(BaseModel):
    heart_rate: float
    spo2: float

@app.post("/predict")
def predict_heart_condition(data: HeartData):
    global latest_prediction
    input_data = np.array([[data.heart_rate, data.spo2]]) - scaler_mean
    input_data = input_data.reshape(1, 1, 2)

    prediction = model.predict(input_data)[0][0]
    result = "Abnormal" if prediction > 0.5 else "Normal"

    latest_prediction = {"status": result, "confidence": round(prediction * 100, 1)}
    return latest_prediction

@app.get("/get_latest_prediction")
def get_latest_prediction():
    return latest_prediction
