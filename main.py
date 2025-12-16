import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from facenet_pytorch import MTCNN
import torch.nn.functional as F
import numpy as np
import time
from datetime import datetime
from collections import deque, Counter

# ===============================================================
# [1] 시스템 설정 (Configuration)
# ===============================================================
VIDEO_SOURCE = 0  # 웹캠 번호
CLASS_NAMES = ['jisung', 'richard', 'unknown'] 
AUTHORIZED_USERS = ['jisung', 'richard'] # 승인된 사용자 목록

# [중요] 모델 경로 (본인 경로에 맞게 수정 필수!)
MODEL_PATH = "./model/20251215_053604/face_model.pth"

# ---------------------------------------------------------------
# [2] 안정화 알고리즘 설정 (Hyperparameters)
# ---------------------------------------------------------------
# A. 다수결 버퍼 (Flickering 방지)
# 최근 10프레임(약 0.3초)의 결과를 모아서 다수결로 판단
BUFFER_SIZE = 10 

# B. 이중 임계값 (Hysteresis Locking)
# 진입 장벽: 처음 인식될 때는 85% 이상이어야 함 (엄격)
# 유지 장벽: 한번 락(Lock)이 걸리면 50%까지 떨어져도 유지 (관대)
HIGH_THRESHOLD = 0.85 
LOW_THRESHOLD = 0.50

# C. 오인식 방지 (Safety Breakers)
# 텔레포트 방지: 얼굴이 0.1초 만에 100픽셀 이상 움직이면 초기화
MAX_MOVE_DISTANCE = 100 
# ===============================================================

def run_dashboard():
    # 1. 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 시스템 시작... (Device: {device})")
    
    # 2. 전처리 정의 (ResNet 입력 규격)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 3. 모델 로드
    print("🛠️ 모델 로딩 중...")
    try:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        print("✅ 모델 로드 완료!")
    except Exception as e:
        print(f"🚨 모델 로드 실패: {e}")
        print(f"경로를 확인하세요: {MODEL_PATH}")
        return

    # 4. MTCNN (얼굴 감지기)
    mtcnn = MTCNN(keep_all=True, device=device)
    
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    
    # -----------------------------------------------------------
    # [상태 변수 초기화]
    # -----------------------------------------------------------
    # UI 레이아웃
    layout_width, layout_height = 1000, 480
    
    # 로그 관련
    access_logs = deque(maxlen=5)
    last_log_time = time.time()
    
    # 알고리즘 관련
    prediction_buffer = deque(maxlen=BUFFER_SIZE) # 최근 결과 저장소
    current_locked_user = None # 현재 락온된 사용자
    prev_center = None # 이전 프레임 얼굴 중심점 (이동 감지용)

    print("🎥 대시보드 실행 (종료하려면 'q'를 누르세요)")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 프레임 리사이즈 (처리 속도 및 UI 규격 통일)
        frame = cv2.resize(frame, (640, 480))
        
        # -------------------------------------------------------
        # [UI] 배경 그리기
        # -------------------------------------------------------
        dashboard = np.zeros((layout_height, layout_width, 3), dtype=np.uint8)
        dashboard[0:480, 0:640] = frame # 왼쪽: 카메라
        
        # 오른쪽 정보창 배경 (Dark Grey)
        ui_x_start = 640
        cv2.rectangle(dashboard, (ui_x_start, 0), (layout_width, layout_height), (30, 30, 30), -1)

        # -------------------------------------------------------
        # [AI] 얼굴 인식 및 로직 처리
        # -------------------------------------------------------
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        
        # 얼굴 감지
        boxes, _ = mtcnn.detect(pil_img)

        # 기본값 (아무도 없을 때)
        final_decision_name = "Scanning..."
        final_prob = 0.0
        current_status = "STANDBY"
        status_color = (100, 100, 100)
        target_text_ui = "SCANNING..."

        if boxes is not None:
            # 가장 큰 얼굴 하나만 처리 (Focusing)
            box = boxes[0] 
            x1, y1, x2, y2 = [int(b) for b in box]
            
            # [안전장치 1] 텔레포트 감지 (위치 급변 확인)
            current_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            if prev_center is not None:
                dist = np.sqrt((current_center[0] - prev_center[0])**2 + (current_center[1] - prev_center[1])**2)
                if dist > MAX_MOVE_DISTANCE: 
                    # 사람이 휙 바뀌었다고 판단 -> 초기화
                    prediction_buffer.clear()
                    current_locked_user = None
            prev_center = current_center

            # 얼굴 이미지 잘라내기 (Crop)
            face_img = pil_img.crop((max(0,x1), max(0,y1), min(640,x2), min(480,y2)))
            
            try:
                # ResNet 추론
                input_tensor = preprocess(face_img).unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probs = F.softmax(outputs, dim=1)
                    max_prob, idx = torch.max(probs, 1)
                    
                    raw_prob = max_prob.item()
                    raw_name = CLASS_NAMES[idx.item()]

                # ---------------------------------------------------
                # [핵심 로직] 락킹 & 히스테리시스 & ID 검증
                # ---------------------------------------------------
                
                # [안전장치 2] ID 변경 감지 (락 상태인데, 다른 사람 확률이 높음)
                if current_locked_user is not None:
                    if raw_name != current_locked_user and raw_prob > 0.6:
                        current_locked_user = None
                        prediction_buffer.clear()
                
                # 임계값 결정 (락 상태면 관대하게, 아니면 엄격하게)
                threshold = LOW_THRESHOLD if current_locked_user else HIGH_THRESHOLD
                
                # 1차 판단
                detected_name = raw_name if raw_prob >= threshold else "unknown"

                # 버퍼 저장 및 다수결 투표
                prediction_buffer.append(detected_name)
                
                if len(prediction_buffer) > 0:
                    most_common_name, count = Counter(prediction_buffer).most_common(1)[0]
                    # 과반수 이상 동의 시 채택
                    if count >= (len(prediction_buffer) // 2):
                        final_decision_name = most_common_name
                    else:
                        final_decision_name = "unknown"
                else:
                    final_decision_name = "unknown"

                # 락 상태 갱신
                if final_decision_name in AUTHORIZED_USERS:
                    current_locked_user = final_decision_name
                elif final_decision_name == "unknown":
                    current_locked_user = None # Unknown이면 락 해제

                # 시각화용 변수 할당
                final_prob = raw_prob
                
                # 얼굴 박스 그리기
                box_color = (0, 255, 0) if final_decision_name in AUTHORIZED_USERS else (0, 0, 255)
                cv2.rectangle(dashboard, (x1, y1), (x2, y2), box_color, 2)

            except Exception:
                pass
        else:
            # 얼굴 없으면 모든 상태 초기화
            prediction_buffer.clear()
            current_locked_user = None
            prev_center = None

        # -------------------------------------------------------
        # [상태 결정]
        # -------------------------------------------------------
        if final_decision_name in AUTHORIZED_USERS:
            current_status = "ACCESS GRANTED"
            status_color = (0, 255, 0) # Green
            target_text_ui = final_decision_name.upper()
        elif final_decision_name == "unknown":
            current_status = "ACCESS DENIED"
            status_color = (0, 0, 255) # Red
            target_text_ui = "UNKNOWN"
        else:
            # Scanning...
            pass

        # -------------------------------------------------------
        # [로그 기록] (1초 Throttling)
        # -------------------------------------------------------
        if current_status != "STANDBY":
            curr_time = time.time()
            if curr_time - last_log_time >= 1.0:
                timestamp = datetime.now().strftime("%H:%M:%S")
                log_text = f"[{timestamp}] {target_text_ui}"
                
                # 중복 로그 방지 (선택사항)
                if not access_logs or access_logs[-1] != log_text:
                    access_logs.append(log_text)
                    last_log_time = curr_time

        # -------------------------------------------------------
        # [UI 그리기] 오른쪽 정보 패널
        # -------------------------------------------------------
        # 1. 헤더
        cv2.putText(dashboard, "AI SECURITY SYSTEM", (ui_x_start + 20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.line(dashboard, (ui_x_start + 20, 50), (layout_width - 20, 50), (100, 100, 100), 1)

        # 2. 상태 배너 (크게)
        cv2.rectangle(dashboard, (ui_x_start + 20, 70), (layout_width - 20, 130), status_color, -1)
        cv2.putText(dashboard, current_status, (ui_x_start + 35, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)

        # 3. 인식된 사용자 이름
        cv2.putText(dashboard, "DETECTED USER:", (ui_x_start + 20, 170), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.putText(dashboard, target_text_ui, (ui_x_start + 20, 200), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        # 4. 신뢰도 게이지 (Bar Chart)
        cv2.putText(dashboard, f"CONFIDENCE: {final_prob*100:.1f}%", (ui_x_start + 20, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        # 게이지 배경
        cv2.rectangle(dashboard, (ui_x_start + 20, 250), (layout_width - 20, 270), (50, 50, 50), -1)
        # 게이지 값
        bar_width = int((layout_width - 20 - (ui_x_start + 20)) * final_prob)
        cv2.rectangle(dashboard, (ui_x_start + 20, 250), (ui_x_start + 20 + bar_width, 270), status_color, -1)

        # 5. 접속 로그 (색상 자동 적용)
        cv2.putText(dashboard, "ACCESS LOG:", (ui_x_start + 20, 310), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        y_log = 335
        for log in access_logs:
            # AUTHORIZED_USERS에 포함된 이름이 로그에 있으면 초록색, 아니면 빨간색
            if any(user.upper() in log for user in AUTHORIZED_USERS):
                log_color = (0, 255, 0)
            else:
                log_color = (0, 0, 255)

            cv2.putText(dashboard, log, (ui_x_start + 20, y_log), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, log_color, 1)
            y_log += 25

        # 6. 하단 시스템 정보
        cv2.line(dashboard, (ui_x_start + 20, 440), (layout_width - 20, 440), (100, 100, 100), 1)
        cv2.putText(dashboard, "Model: ResNet18 | GPU: ON", (ui_x_start + 20, 465), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

        # 화면 출력
        cv2.imshow('AI Face Dashboard', dashboard)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_dashboard()