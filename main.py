from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
import mediapipe as mp
import uvicorn
import os

# Khởi tạo FastAPI
app = FastAPI()

# Khởi tạo MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

@app.get("/")
async def home():
    return {"message": "AI API đang chạy trên Render!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Đọc file ảnh
    image_bytes = await file.read()
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

    # Chuyển ảnh sang RGB vì MediaPipe yêu cầu
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Xử lý bằng MediaPipe
    results = hands.process(image_rgb)
    
    # Đếm số ngón tay giơ lên
    fingers_up = 0
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            fingers = [4, 8, 12, 16, 20]  # Các điểm mốc ngón tay

            for i in range(1, 5):  # Duyệt từ ngón trỏ đến út
                if hand_landmarks.landmark[fingers[i]].y < hand_landmarks.landmark[fingers[i] - 2].y:
                    fingers_up += 1
            
            # Kiểm tra ngón cái (so sánh theo trục x thay vì y)
            if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x:
                fingers_up += 1

    return {"fingers_up": fingers_up}

# Chạy API
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  
    uvicorn.run(app, host="0.0.0.0", port=port)
