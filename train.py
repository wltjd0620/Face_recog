import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import os
import time
import datetime # 모델 이름 시간 지정

# ====================================================================
# 설정
DATA_DIR = '/workspace/face_recog/dataset' # 데이터 폴더 경로

# 모델 이름
now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
MODEL_SAVE_PATH = f'/workspace/face_recog/model/face_model_{now}.pth'

BATCH_SIZE = 32 # 한번에 학습할 사진 개수
LEARNING_RATE = 0.001 # 학습 속도
NUM_EPOCHS = 15 # 반복 학습 횟수
TRAIN_SPLIT_RATIO = 0.8 # 데이터 나누기 비율 (80% 학습, 20% 검증)
# ====================================================================


def train_model():
    print("-------------------------")
    print("학습 시작")
    print("-------------------------")
    
    # 장치 설정 ( GPU 우선 )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"학습 장치: {device}")

    # 데이터 전처리 및 증강
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)), # ResNet 기본 입력 크기
        
        # 데이터 증강
        transforms.RandomHorizontalFlip(p=0.5), # 50% 확률로 좌우 반전
        transforms.RandomRotation(15), # -15 ~ 15도 살짝 회전
        transforms.ColorJitter(brightness=0.2,contrast=0.2), # 밝기/ 대비 변화
        transforms.RandomGrayscale(p=0.1), # 10% 확률로 흑백 사진으로 만듦 (색깔 의존도 낮춤)

        transforms.ToTensor(), # 텐서로 변환
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 정규화
    ])

    # 데이터 불러오기
    try:
        full_dataset = datasets.ImageFolder(DATA_DIR, transform=data_transforms)
    except Exception as e:
        print(f"에러: 데이터 없음 ({e})")
        return
    
    # 데이터 쪼개기 (Train vs Val)
    train_size = int(TRAIN_SPLIT_RATIO * len(full_dataset))
    val_size = len(full_dataset) - train_size

    # 랜덤으로 섞어서 나누기
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"전체 이미지: {len(full_dataset)}장")
    print(f"학습용(Train): {len(train_dataset)}장")
    print(f"검증용(Val):   {len(val_dataset)}장")

    class_names = full_dataset.classes
    print(f"클래스 목록: {class_names}")

    # 데이터 로더
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False) # 검증 때는 shuffle 불필요

    # 모델 설계
    print("모델 불러오는 중...")
    model = models.resnet18(weights='IMAGENET1K_V1')

    # 모델의 마지막 층(출력층)을 우리 목적에 맞게 교체 ( 우리의 클래스 개수 )
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names)) 

    # 모델 gpu로 이동
    model = model.to(device)

    # 학습 설정 ( 손실함수, 최적화함수 )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 학습 시작
    print(f"\n모델 학습 시작( 총 {NUM_EPOCHS} 회 반복)")
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}')
        print('-' * 10)

        # 각 Epoch마다 '학습(Train)'과 '검증(Val)' 단계를 모두 거침
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() # 학습 모드 (Dropout 켜기, 가중치 업데이트 O)
                dataloader = train_loader
            else:
                model.eval()  # 평가 모드 (Dropout 끄기, 가중치 업데이트 X)
                dataloader = val_loader
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 이전 학습 기록 초기화
            optimizer.zero_grad()
            
            # 학습 단계에서만 그래디언트(기울기) 계산
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # 학습 단계라면 역전파 수행
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            correct += torch.sum(preds == labels.data)
            total += inputs.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct.double() / total * 100

        print(f'{phase.upper()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%')
    
    time_elapsed = time.time() - start_time
    print(f"학습 완료, 소요 시간: {time_elapsed // 60:.0f}분 {time_elapsed % 60:.0f}초")

    # 7. 모델 저장
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("모델이 저장되었습니다.")
    print(f"[중요] main.py의 CLASS_NAMES를 다음 순서로 수정하세요: {class_names}")

if __name__ == "__main__":
    train_model()