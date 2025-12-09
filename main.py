import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import models, transforms
from PIL import Image
from facenet_pytorch import MTCNN
import torch.nn.functional as F

# ===============================================================
# [1] 환경 설정 (이 부분만 본인 상황에 맞게 수정하세요)
# ===============================================================

# 학습할 때 사용한 클래스 이름 (알파벳 순서대로 적어야 합니다!)
# 예: dataset 폴더 안에 ['jisung', 'minji', 'unknown'] 폴더가 있다면 그 순서 그대로.
CLASS_NAMES = ['jisung', 'unknown'] 

# 문을 열어줄 사람 목록
AUTHORIZED_USERS = ['jisung']

# 몇 % 이상 확신할 때만 문을 열어줄지 (0.0 ~ 1.0)
# unknown이 포함되어 있으므로 0.8(80%) 이상 추천
CONFIDENCE_THRESHOLD = 0.85 

# 모델 파일 경로
MODEL_PATH = '/workspace/face_recog/model/face_model_20251202_060528.pth'

# 영상 소스 (파일 경로 또는 0)
VIDEO_SOURCE = '/workspace/face_recog/test/Image.jpg' 
# ===============================================================

# 1. 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 실행 장치: {device}")

# 2. 데이터 전처리 (학습할 때와 똑같이 맞춰야 함)
# ResNet은 224x224 크기를 좋아합니다.
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. 모델 불러오기
print("모델 로딩 중...")
# (1) 뼈대 만들기
model = models.resnet18(weights=None) # 껍데기만 가져옴
num_ftrs = model.fc.in_features
# (2) 마지막 층 개수 맞추기 (학습한 클래스 개수만큼)
model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES)) 
# (3) 저장된 가중치(Brain) 심기
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
except RuntimeError as e:
    print(f"❌ 에러 발생! 학습된 클래스 개수와 CLASS_NAMES 개수가 다릅니다.\n에러 내용: {e}")
    exit()

model = model.to(device)
model.eval() # 평가 모드로 전환 (Dropout 등 비활성화)

# 4. 얼굴 감지기 (MTCNN) 로드
mtcnn = MTCNN(keep_all=True, device=device)

# 5. 영상 처리 시작
cap = cv2.VideoCapture(VIDEO_SOURCE)

# 영상 저장 설정
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
four