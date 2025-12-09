import cv2
import os
from facenet_pytorch import MTCNN
import torch
import sys


# ==========================================
# 설정: 저장할 라벨 이름 ( 이름이나 unknown )
LABEL = 'jisung' 
VIDEO_SOURCE = '/workspace/face_recog/dataset_make_video/jskim3.mp4' # 웹캠이면 0, 파일이면 '파일명.mp4'
SAVE_COUNT = 300 # 저장할 사진 개수 -> frame 단위

FRAME_INTERVAL = 10 # n프레임마다 1장씩 저장
# 영상길이 1분 미만 : 5 ~ 10
# 영상길이 3분 이상 : 30 ( 1초에 1장 저장 )
# ==========================================

print("------------------------------------------------------------------------------------")
print(f"데이터 수집 시작: {VIDEO_SOURCE} -> {LABEL} 라벨 폴더")
print("------------------------------------------------------------------------------------")

# 장치 설정
# GPU 체크
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"장치 확인 완료:{device}")

# mtcnn 로드
print("모델 로딩중 . . .")
mtcnn = MTCNN(keep_all=False, device=device) # 얼굴 하나만 찾기
print("모델 로딩 성공")

# 저장경로 설정
save_path = f'./dataset/{LABEL}'
os.makedirs(save_path, exist_ok=True)

# 파일명 접두사 추출
base_name = os.path.basename(VIDEO_SOURCE) 
video_name = os.path.splitext(base_name)[0]
print(f"파일명 규칙: {video_name}_숫자.jpg")

print(f"영상 소스({VIDEO_SOURCE}) 여는 중...")
cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    print(f"\n[error] 영상을 열 수 없습니다!")
    print(f"원인 1: 폴더 안에 '{VIDEO_SOURCE}' 파일이 없는 경우")
    print(f"원인 2: 파일명 오타 (me.mp4 vs me.MOV 등)")
    print(f"원인 3: 도커에서 웹캠(0)을 쓰려고 한 경우 (도커 실행 시 장치 연결 안 함)")
    sys.exit() # 프로그램 강제 종료

print("   -> 영상 열기 성공! 데이터 수집을 시작합니다.") 
saved_count = 0
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame_count += 1
    # 지정된 간격마다 실행
    if frame_count % FRAME_INTERVAL == 0:
        # BGR -> RGB
        # cvtColor : color을 convert(변환)해라
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 파일명 생성
        file_name = f"{video_name}_{saved_count}.jpg"
        final_save_path = f"{save_path}/{file_name}"
        
        # 저장 시도
        result = mtcnn(frame_rgb, save_path=final_save_path)
        
        if result is not None:
            saved_count += 1
            # 10장마다 로그 출력 (너무 빠르면 정신없으니까)
            if saved_count % 10 == 0:
                print(f"수집 중... 총 {saved_count}장 모음 (파일명: {file_name})")


cap.release()
print(f"수집 완료. 총 {saved_count}장의 데이터를 확보했습니다.")