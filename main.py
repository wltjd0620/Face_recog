import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
from facenet_pytorch import MTCNN
import torch.nn.functional as F
import numpy as np
import time
from datetime import datetime
from collections import deque

# ===============================================================
# [설정]
VIDEO_SOURCE = 0  # 웹캠
CLASS_NAMES = ['jisung', 'unknown'] 
AUTHORIZED_USERS = ['jisung']
CONFIDENCE_THRESHOLD = 0.80

MODEL_PATH = "./model/20251209_052410/face_model.pth"
# ===============================================================

def run_dashboard():
    # 1. 장치 및 모델 준비
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("시스템 초기화 중...")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
    except:
        print("모델 경로를 확인해주세요")
        return

    mtcnn = MTCNN(keep_all=True, device=device)
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    # 로그 저장용 큐
    access_logs = deque(maxlen=5)
    
    # [추가] 마지막으로 로그를 남긴 시간을 기억할 변수 (초기화)
    last_log_time = time.time()

    layout_width = 1000
    layout_height = 480

    print("대시보드 실행 (종료: q)")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.resize(frame, (640, 480))
        
        # 전체 캔버스 그리기
        dashboard = np.zeros((layout_height, layout_width, 3), dtype=np.uint8)
        dashboard[0:480, 0:640] = frame

        # 오른쪽 정보창 배경
        ui_x_start = 640
        cv2.rectangle(dashboard, (ui_x_start, 0), (layout_width, layout_height), (30, 30, 30), -1)

        # AI 인식
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        boxes, _ = mtcnn.detect(pil_img)

        current_status = "STANDBY"
        status_color = (100, 100, 100)
        target_name = "Scanning..."
        prob_val = 0.0

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [int(b) for b in box]
                face_img = pil_img.crop((max(0,x1), max(0,y1), min(640,x2), min(480,y2)))
                
                try:
                    input_tensor = preprocess(face_img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        probs = F.softmax(outputs, dim=1)
                        max_prob, idx = torch.max(probs, 1)
                        prob_val = max_prob.item()
                        pred_name = CLASS_NAMES[idx.item()]

                    if prob_val < CONFIDENCE_THRESHOLD:
                        pred_name = "unknown"
                    
                    if pred_name in AUTHORIZED_USERS:
                        current_status = "ACCESS GRANTED"
                        status_color = (0, 255, 0)
                        box_color = (0, 255, 0)
                        target_name = pred_name.upper()
                    else:
                        current_status = "ACCESS DENIED"
                        status_color = (0, 0, 255)
                        box_color = (0, 0, 255)
                        target_name = "UNKNOWN"

                    cv2.rectangle(dashboard, (x1, y1), (x2, y2), box_color, 2)
                    
                    # -------------------------------------------------------
                    # [핵심 수정] 로그 기록 속도 조절 (Throttling)
                    # -------------------------------------------------------
                    curr_time = time.time()
                    # 1. 마지막 로그 기록보다 1초 이상 지났는지 확인
                    if curr_time - last_log_time >= 1.0:
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        log_text = f"[{timestamp}] {target_name}"
                        
                        # 2. 내용이 이전 로그와 다를 때만 기록 (선택 사항) 
                        #    또는 그냥 1초마다 무조건 기록하고 싶으면 if문 빼도 됨
                        if not access_logs or access_logs[-1] != log_text:
                             access_logs.append(log_text)
                             last_log_time = curr_time # 시간 갱신!

                except Exception as e:
                    pass
        
        # ---------------------------------------------------------
        # [UI 그리기]
        # ---------------------------------------------------------
        # 1. 헤더
        cv2.putText(dashboard, "AI SECURITY SYSTEM", (ui_x_start + 20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.line(dashboard, (ui_x_start + 20, 50), (layout_width - 20, 50), (100, 100, 100), 1)

        # 2. 상태 배너
        cv2.rectangle(dashboard, (ui_x_start + 20, 70), (layout_width - 20, 130), status_color, -1)
        cv2.putText(dashboard, current_status, (ui_x_start + 35, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)

        # 3. 인식 정보
        cv2.putText(dashboard, "DETECTED USER:", (ui_x_start + 20, 170), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.putText(dashboard, target_name, (ui_x_start + 20, 200), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        # 4. 신뢰도 게이지
        cv2.putText(dashboard, f"CONFIDENCE: {prob_val*100:.1f}%", (ui_x_start + 20, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.rectangle(dashboard, (ui_x_start + 20, 250), (layout_width - 20, 270), (50, 50, 50), -1)
        bar_width = int((layout_width - 20 - (ui_x_start + 20)) * prob_val)
        cv2.rectangle(dashboard, (ui_x_start + 20, 250), (ui_x_start + 20 + bar_width, 270), status_color, -1)

        # 5. 접속 로그
        cv2.putText(dashboard, "ACCESS LOG:", (ui_x_start + 20, 310), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        y_log = 335
        for log in access_logs:
            log_color = (0, 255, 0) if "JISUNG" in log or "MINJI" in log else (0, 0, 255)
            cv2.putText(dashboard, log, (ui_x_start + 20, y_log), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, log_color, 1)
            y_log += 25

        # 6. 하단 정보
        cv2.line(dashboard, (ui_x_start + 20, 440), (layout_width - 20, 440), (100, 100, 100), 1)
        cv2.putText(dashboard, "Model: ResNet18 | GPU: ON", (ui_x_start + 20, 465), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

        cv2.imshow('AI Face Dashboard', dashboard)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_dashboard()