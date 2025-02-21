from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
import uvicorn
import os

app = FastAPI()  # Định nghĩa app trước khi dùng

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

# Chạy ứng dụng khi script được chạy trực tiếp
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Render yêu cầu lấy PORT từ env
    uvicorn.run(app, host="0.0.0.0", port=port)
