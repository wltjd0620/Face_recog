import os
import random
from PIL import Image
from tqdm import tqdm
import torch
# [핵심 추가] 얼굴 감지기 라이브러리
from facenet_pytorch import MTCNN

# ==================================================================
# [설정] 여기만 수정하세요!
# ==================================================================
# 1. Kaggle 원본 데이터 경로 (압축 푼 폴더)
SOURCE_DIR = r'/workspace/face_recog/dataset_make_video/Humans' 

# 2. 저장할 나의 unknown 폴더
DEST_DIR = r'/workspace/face_recog/dataset/unknown'

# 3. 목표 개수 (얼굴을 못 찾는 경우를 대비해 넉넉히 1.5배수로 설정)
# 나중에 폴더에서 300개만 남기고 지우셔도 됩니다.
TARGET_COUNT = 450
# ==================================================================

def preprocess_images_with_crop():
    # 장치 설정 (GPU가 있으면 더 빠름)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 얼굴 전처리 시작 (사용 장치: {device})")

    # [핵심] MTCNN 얼굴 감지기 초기화
    # image_size=224: 잘라낸 얼굴을 자동으로 224x224로 맞춰줌!
    # margin=20: 얼굴 너무 꽉 차게 자르지 말고 여백 좀 주기
    mtcnn = MTCNN(image_size=224, margin=20, keep_all=False, device=device)

    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)
    
    print("🔍 이미지 파일을 찾는 중...")
    all_images = []
    for root, dirs, files in os.walk(SOURCE_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_images.append(os.path.join(root, file))
    
    total_imgs = len(all_images)
    print(f"🧐 발견된 총 이미지: {total_imgs}장")

    if total_imgs == 0: return

    # 랜덤 셔플 (섞어서 앞에서부터 처리)
    random.shuffle(all_images)

    print("✂️ 얼굴 감지 및 자르기(Crop) 시작...")
    
    count = 0
    # TQDM 진행바 설정
    try: iterator = tqdm(all_images)
    except ImportError: iterator = all_images

    for img_path in iterator:
        # 목표 개수 채우면 중단
        if count >= TARGET_COUNT:
            break

        try:
            # 1. 이미지 열기 및 RGB 변환
            img = Image.open(img_path).convert('RGB')
            
            # 2. [핵심] MTCNN으로 얼굴 찾아서 자르기!
            # 이 함수가 알아서 얼굴을 찾고, 잘라서(crop), 224로 리사이징까지 해서 돌려줍니다.
            # 얼굴이 없으면 None을 반환합니다.
            face_tensor = mtcnn(img)

            if face_tensor is not None:
                # 3. 텐서를 다시 이미지로 변환 (저장을 위해)
                # 픽셀 값 범위를 [0, 1]에서 [0, 255]로 되돌림
                face_img = face_tensor.permute(1, 2, 0).mul(255).byte().numpy()
                face_pil = Image.fromarray(face_img)
                
                # 4. 저장
                save_name = f"unknown_{count+1:04d}.jpg"
                save_path = os.path.join(DEST_DIR, save_name)
                face_pil.save(save_path, 'JPEG', quality=95)
                
                count += 1
            else:
                # print(f"스킵: 얼굴 없음 ({img_path})") # 너무 많이 뜨면 주석 처리
                pass
                
        except Exception as e:
            # print(f"에러: {img_path} ({e})")
            pass

    print("------------------------------------------------")
    print(f"✅ 전처리 완료! 총 {count}장의 얼굴을 잘라서 '{DEST_DIR}'에 저장했습니다.")
    print("💡 Tip: 폴더에 들어가서 이상하게 잘린 사진이 없는지 쓱 훑어보고, 개수를 300개 정도로 맞춰주세요.")

if __name__ == '__main__':
    # 혹시 facenet-pytorch가 없다면 설치하라는 안내
    try:
        import facenet_pytorch
    except ImportError:
        print("🚨 에러: 'facenet-pytorch' 라이브러리가 필요합니다.")
        print("👉 실행: pip install facenet-pytorch")
        exit()
        
    preprocess_images_with_crop()