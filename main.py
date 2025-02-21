from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
from io import BytesIO

app = FastAPI()

@app.get("/")
async def home():
    return {"message": "AI API đang chạy trên Render!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

    # 🔥 XỬ LÝ AI TẠI ĐÂY 🔥
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Chuyển ảnh thành grayscale

    _, encoded_img = cv2.imencode(".jpg", gray)
    return {"result": encoded_img.tobytes()}  # Trả ảnh đã xử lý
