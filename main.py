import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from facenet_pytorch import MTCNN
import torch.nn.functional as F
import numpy as np
import time

# ===============================================================
# [1] 사용자 설정 (여기만 수정하면 됩니다!)
# ===============================================================
# 1. 사용할 영상 소스
# - 웹캠 사용 시: 0  (숫자 0)
# - 파일 사용 시: 'test_video.mp4' (문자열)
VIDEO_SOURCE = 0

# 2. 클래스 이름 ( train.py 돌릴 때 폴더 순서(알파벳순)와 똑같아야 함!)
# 예: dataset 폴더에 jisung, minji, unknown이 있다면 -> ['jisung', 'minji', 'unknown']
CLASS_NAMES = ["jisung", "unknown"]

# 3. 문 열어줄 사람 명단
AUTHORIZED_USERS = ["jisung"]

# 4. 확신 기준 (이 점수보다 낮으면 모르는 사람 취급)
# 0.7 (70%) ~ 0.8 (80%) 추천
CONFIDENCE_THRESHOLD = 0.8

# 5. 모델 파일 경로
MODEL_PATH = "./model/20251209_052410/face_model.pth"

# ===============================================================


def run_inference():
    print("------------------------------------------------")
    print("얼굴 인식 시스템 가동 (Inference Mode)")
    print(f"타겟: {CLASS_NAMES}")
    print("------------------------------------------------")

    # 1. 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"1. 시스템 장치: {device}")

    # 2. 데이터 전처리 (⚠️ 중요: 학습 때 썼던 증강(회전, 반전)은 다 빼고 정석만 남김)
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # 3. 모델 로드
    print("2. AI 모델(ResNet18) 로딩 중...")
    try:
        # 껍데기 만들기
        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES))

        # 가중치 불러오기 (CPU/GPU 호환성 처리)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model = model.to(device)

        # [핵심] 평가 모드로 전환 (Dropout, BatchNorm 고정)
        model.eval()
        print("   -> 모델 로딩 성공!")
    except Exception as e:
        print(f"모델 로딩 실패! 경로와 클래스 개수를 확인하세요.\n{e}")
        return

    # 4. 얼굴 감지기 (MTCNN)
    # keep_all=True: 화면에 있는 모든 사람 다 찾기
    mtcnn = MTCNN(keep_all=True, device=device)

    # 5. 카메라 켜기
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("카메라(또는 파일)를 열 수 없습니다.")
        return

    print("🟢 [Start] 화면에 얼굴을 비춰주세요. (종료: 'q' 키)")

    prev_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # FPS 계산
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # OpenCV(BGR) -> PIL(RGB) 변환
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        # 1. 얼굴 위치 찾기 (Detection)
        boxes, _ = mtcnn.detect(pil_img)

        if boxes is not None:
            for box in boxes:
                # 좌표 정수 변환 및 예외 처리
                x1, y1, x2, y2 = [int(b) for b in box]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                # 얼굴 영역이 너무 작으면 패스 (노이즈 방지)
                if (x2 - x1) < 20 or (y2 - y1) < 20:
                    continue

                # 2. 얼굴 자르기 (Crop)
                face_img = pil_img.crop((x1, y1, x2, y2))

                try:
                    # 3. 전처리 및 AI 예측
                    input_tensor = preprocess(face_img).unsqueeze(0).to(device)

                    with torch.no_grad():  # 계산 기록 끄기 (속도 향상)
                        outputs = model(input_tensor)
                        probs = F.softmax(outputs, dim=1)  # 확률로 변환 (0~1)
                        max_prob, idx = torch.max(probs, 1)

                        prob_val = max_prob.item()
                        pred_name = CLASS_NAMES[idx.item()]

                    # 4. 결과 판독 (Thresholding)
                    if prob_val < CONFIDENCE_THRESHOLD:
                        # 확률이 낮으면 모르는 사람으로 간주
                        final_name = "Unknown"
                        color = (0, 0, 255)  # 빨강 (Red)
                        status_text = f"UNKNOWN ({prob_val*100:.1f}%)"
                    else:
                        # 확률이 높을 때
                        if pred_name in AUTHORIZED_USERS:
                            final_name = pred_name
                            color = (0, 255, 0)  # 초록 (Green)
                            status_text = (
                                f"OPEN: {pred_name.upper()} ({prob_val*100:.1f}%)"
                            )
                        elif pred_name == "unknown":
                            final_name = "Unknown"
                            color = (0, 0, 255)  # 빨강
                            status_text = f"UNKNOWN ({prob_val*100:.1f}%)"
                        else:
                            final_name = pred_name
                            color = (0, 0, 255)
                            status_text = f"DENIED: {pred_name} ({prob_val*100:.1f}%)"

                    # 5. 화면에 그리기
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    # 글자 배경 검은색 박스 (가독성 UP)
                    cv2.rectangle(
                        frame,
                        (x1, y1 - 35),
                        (x1 + len(status_text) * 18, y1),
                        color,
                        -1,
                    )
                    cv2.putText(
                        frame,
                        status_text,
                        (x1 + 5, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                    )

                except Exception as e:
                    pass  # 얼굴 처리 중 에러 나면 무시하고 다음 프레임

        # FPS 표시
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        # 화면 출력
        cv2.imshow("AI Face Security System", frame)

        # 'q' 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("🔴 프로그램 종료")


if __name__ == "__main__":
    run_inference()
