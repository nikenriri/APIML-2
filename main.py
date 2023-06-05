import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI,File,UploadFile
import uvicorn
from PIL import Image
from io import BytesIO

#Define Function
labels = ['Jamur Enoki', 'Jamur Shimeji Coklat', 'Jamur Shimeji Putih', 'Jamur Tiram']

def process(file)-> Image.Image:
    image = image = Image.open(BytesIO(file))
    return image

def predict(image: Image.Image):
    loaded_model = tf.keras.models.load_model('YangJamurJamuraja_v2.h5')
    image = tf.image.resize(image, (224,224))
    image = (cv2.cvtColor(image.numpy().astype(np.uint8),cv2.COLOR_BGR2RGB))
    image = np.expand_dims(image/255,0)
    hasil = loaded_model.predict(image)
    idx = hasil.argmax()
    return labels[idx]


#FASTAPI
app = FastAPI()

@app.post("/predict/image")
async def predict_fastapi(file: UploadFile = File(...)):
    image = process(await file.read())
    prediction = predict(image)
    return prediction

uvicorn.run(app, host='0.0.0.0', port=3000)