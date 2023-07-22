from fastapi import FastAPI, File ,UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests
from fastapi.middleware.cors import CORSMiddleware
import warnings
warnings.filterwarnings('ignore')

app=FastAPI()

# endpoint ="http:localhost:3000"
origins=[
    "http://localhost:",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,                      
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
MODEL =tf.keras.models.load_model("/Users/RAVJOT SINGH RAYAT/Desktop/5th sem/5th project/saved_models/3")

CLASS_NAMES=["bacterial_blight", "curl_virus", "fussarium_wilt", "healthy"]

@app.get("/ping")
async def ping():
    return "Hello, i am alive"

def read_file_as_image(data) -> np.ndarray:
    image=np.array(Image.open(BytesIO(data)))
    return image

# @app.post("/predict/")
# async def predict(
#     file: UploadFile
# ):
#     return {"filename": file.filename}


@app.post("/predict/")
async def predict(
    file: UploadFile=File(...)
    ):
    print("route entered!")
    print ({"filename": file.filename})
    image = read_file_as_image( await file.read())
    img_batch = np.expand_dims(image,0)
    
    predictions =MODEL.predict(img_batch)
    pass

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class':predicted_class,
        'confidence':float(confidence)
        }

if __name__ == "__main__":
    uvicorn.run(app,host='localhost',port=8000)
