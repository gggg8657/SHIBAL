# 🎯 커스텀 데이터셋 STEAD 사용 가이드

이 디렉토리는 `image_segments.json` 파일의 세그먼트 정보를 활용하여 커스텀 데이터셋으로 STEAD 모델을 훈련하고 테스트하기 위한 모든 파일들을 포함합니다.

## 📁 디렉토리 구조

```
for_custom/
├── saved_models/           # 프리트레인드 모델들
│   ├── 888tiny.pkl        # Tiny 모델 (93KB)
│   └── 913base.pkl        # Base 모델 (6.3MB)
├── segment_parser.py       # 세그먼트 정보 파싱
├── segment_feat_extractor.py  # 특징 추출기
├── segment_dataset.py      # 커스텀 데이터셋 클래스
├── preprocess_segments.py  # 메인 프리프로세싱 스크립트
├── train_custom.py         # 커스텀 훈련 스크립트
├── test_custom.py          # 상세 테스트 스크립트
├── quick_test.py           # 빠른 테스트 스크립트
├── model.py                # STEAD 모델 아키텍처
├── dataset.py              # 기존 데이터셋 (호환성)
├── utils.py                # 유틸리티 함수들
├── option.py               # 설정 옵션
├── requirements.txt         # 필요한 패키지들
└── README_custom.md        # 이 파일
```

## 🚀 빠른 시작

### **1단계: 환경 설정**
```bash
cd for_custom
pip install -r requirements.txt
```

### **2단계: 데이터 준비**
```bash
# 세그먼트 정보 파싱 및 데이터 리스트 생성
python preprocess_segments.py --json_path ../image_segments.json --skip_features

# 특징 추출 (PyTorchVideo 설치 필요)
pip install pytorchvideo torchvision
python preprocess_segments.py --json_path ../image_segments.json
```

### **3단계: 테스트 (프리트레인드 모델)**
```bash
# 빠른 테스트
python quick_test.py

# 상세 테스트
python test_custom.py --model_path saved_models/888tiny.pkl --test_list custom_data/custom_test.txt
```

### **4단계: 커스텀 훈련**
```bash
# 커스텀 데이터로 훈련
python train_custom.py --comment custom_training --max_epoch 30
```

## 📊 세그먼트 정보 구조

`image_segments.json` 파일은 다음과 같은 구조를 가집니다:

```json
{
    "start_time": "143012",
    "start_frame": "273",
    "description": "NORMAL",
    "description_en": "normal",
    "category": "normal",
    "images": [
        "D:\\output_2025\\frame_20250728-143012_273.jpg",
        "D:\\output_2025\\frame_20250728-143012_308.jpg"
    ]
}
```

**지원되는 카테고리:**
- `normal` (정상) → 라벨 0
- `rest` (휴식) → 라벨 0
- `violence` (폭력) → 라벨 1
- `abnormal movement` (비정상 움직임) → 라벨 1
- `baggage movement` (짐 이동) → 라벨 1
- `collapse` (붕괴) → 라벨 1
- `suspicious behavior` (의심스러운 행동) → 라벨 1
- `unknown` (알 수 없음) → 라벨 1

## 🔧 주요 스크립트 설명

### **preprocess_segments.py**
- 세그먼트 정보 파싱
- 훈련/테스트 데이터 리스트 생성
- X3D 모델을 사용한 특징 추출
- 카테고리별 통계 출력

**사용법:**
```bash
# 리스트만 생성
python preprocess_segments.py --skip_features

# 특징까지 추출
python preprocess_segments.py --model_name x3d_l
```

### **train_custom.py**
- 세그먼트 정보를 활용한 커스텀 훈련
- Triplet Loss를 사용한 대조 학습
- 정상/비정상 특징 간 거리 학습

**사용법:**
```bash
python train_custom.py --comment my_custom_model --max_epoch 50 --batch_size 32
```

### **test_custom.py**
- 상세한 성능 분석
- 세그먼트별 성능 지표
- 시각화 결과 저장
- JSON/CSV 형태로 결과 저장

**사용법:**
```bash
python test_custom.py --model_path saved_models/888tiny.pkl --output_dir ./my_results
```

### **quick_test.py**
- 간단한 성능 확인
- 기본적인 ROC AUC 및 정확도
- 빠른 결과 확인용

**사용법:**
```bash
python quick_test.py
```

## 📈 성능 지표

### **전체 성능**
- **ROC AUC**: 이상 탐지의 전반적인 성능
- **PR AUC**: 불균형 데이터에서 더 정확한 성능 측정
- **정확도, 정밀도, 재현율, F1-Score**

### **세그먼트별 성능**
- 각 카테고리별 ROC AUC 및 PR AUC
- 샘플 수 및 정상/비정상 비율
- 개선이 필요한 세그먼트 식별

## 🎨 시각화 결과

테스트 실행 시 다음 파일들이 생성됩니다:

- `custom_test_results.png`: 성능 지표 시각화
- `custom_umap_embedding.png`: 특징 임베딩 시각화
- `custom_detailed_results.json`: 상세 결과 (JSON)
- `custom_test_predictions.csv`: 예측 결과 (CSV)

## ⚙️ 설정 옵션

### **모델 아키텍처**
- `--model_arch tiny`: 경량 모델 (32차원, 빠른 추론)
- `--model_arch fast`: 중간 모델 (32차원, 균형)
- `--model_arch base`: 기본 모델 (192차원, 높은 정확도)

### **훈련 파라미터**
- `--batch_size`: 배치 크기 (기본값: 16)
- `--lr`: 학습률 (기본값: 2e-4)
- `--max_epoch`: 최대 에포크 (기본값: 30)
- `--dropout_rate`: 드롭아웃 비율 (기본값: 0.4)

## 🔍 문제 해결

### **PyTorchVideo 설치 오류**
```bash
pip install pytorchvideo
# 또는
conda install pytorchvideo -c pytorch
```

### **CUDA 메모리 부족**
```bash
# 배치 크기 줄이기
python train_custom.py --batch_size 8

# 모델 아키텍처 변경
python train_custom.py --model_arch tiny
```

### **데이터 로딩 오류**
```bash
# 데이터 리스트 파일 확인
ls -la custom_data/

# 특징 파일 존재 확인
ls -la features/ | head -10
```

## 📝 사용 예시

### **전체 워크플로우**
```bash
# 1. 데이터 준비
python preprocess_segments.py

# 2. 프리트레인드 모델로 테스트
python test_custom.py --model_path saved_models/888tiny.pkl

# 3. 커스텀 훈련
python train_custom.py --comment my_model --max_epoch 50

# 4. 훈련된 모델로 테스트
python test_custom.py --model_path ckpt/2e-4_16_my_model/model29-x3d.pkl
```

### **다양한 모델 테스트**
```bash
# Tiny 모델
python test_custom.py --model_path saved_models/888tiny.pkl --model_arch tiny

# Base 모델
python test_custom.py --model_path saved_models/913base.pkl --model_arch base
```

## 🎯 팁과 트릭

1. **특징 추출**: 처음에는 `--skip_features`로 리스트만 생성하고, 나중에 특징 추출
2. **모델 선택**: 빠른 테스트는 `tiny`, 정확한 결과는 `base` 모델 사용
3. **배치 크기**: GPU 메모리에 따라 조정 (8, 16, 32)
4. **데이터 분할**: 80% 훈련, 20% 테스트로 자동 분할
5. **결과 저장**: 모든 테스트 결과는 자동으로 저장되어 나중에 분석 가능

## 📞 지원

문제가 발생하면 다음을 확인하세요:
1. 필요한 패키지가 모두 설치되었는지
2. 데이터 파일 경로가 올바른지
3. GPU 메모리가 충분한지
4. Python 버전이 호환되는지 (3.8+ 권장)

이제 `image_segments.json`의 풍부한 세그먼트 정보를 활용하여 더 정확한 이상 탐지 모델을 훈련할 수 있습니다! 🚀
