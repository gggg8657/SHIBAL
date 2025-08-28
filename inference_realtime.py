import os
import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import time
import argparse

# 상위 디렉토리 import를 위한 경로 추가
sys.path.append('..')
from model import Model

class LightweightSTEAD(nn.Module):
    """경량화된 STEAD 모델 (실시간 추론용)"""
    
    def __init__(self, model_arch='tiny', dropout=0.4, attn_dropout=0.1, 
                 ff_mult=1, dims=[32, 32], depths=[1, 1]):
        super().__init__()
        
        # 경량화된 Feature Extractor (ResNet18 기반)
        try:
            import torchvision.models as models
            self.feature_extractor = models.resnet18(pretrained=True)
            # 마지막 FC 레이어 제거
            self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])
            self.feature_dim = 512
        except:
            # torchvision이 없는 경우 간단한 CNN
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),  # 64 → 32로 경량화
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),  # 128 → 64로 경량화
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),  # 256 → 128로 경량화
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
            self.feature_dim = 128
        
        # STEAD 모델 (경량화)
        if model_arch == 'base':
            self.stead_model = Model(dropout=dropout, attn_dropout=attn_dropout)
        else:
            # Tiny 모델을 더 작게 만들기
            self.stead_model = Model(dropout=dropout, attn_dropout=attn_dropout,
                                   ff_mult=ff_mult, dims=tuple(dims), depths=tuple(depths))
        
        # Feature 차원 맞추기
        self.feature_adapter = nn.Linear(self.feature_dim, 400)
        
        # 추론 모드로 설정
        self.eval()
    
    def forward(self, x):
        # 이미지 → 특징 추출
        batch_size = x.size(0)
        
        # ResNet의 경우
        if hasattr(self.feature_extractor, 'avgpool'):
            features = self.feature_extractor(x)
            features = features.view(batch_size, -1)
        else:
            features = self.feature_extractor(x)
        
        # 차원 맞추기
        features = self.feature_adapter(features)
        
        # STEAD 모델 입력 형태로 변환
        features = features.unsqueeze(1)
        
        # STEAD 모델 통과
        output = self.stead_model(features)
        
        return output

class RealtimeAnomalyDetector:
    """실시간 이상 탐지기"""
    
    def __init__(self, model_path, model_arch='tiny', device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"[INFO] 디바이스: {self.device}")
        
        # 모델 로드
        self.model = self.load_model(model_path, model_arch)
        self.model.to(self.device)
        self.model.eval()
        
        # 이미지 전처리
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 임계값 설정
        self.threshold = 0.5
        
        print(f"[INFO] 모델 로드 완료: {model_path}")
    
    def load_model(self, model_path, model_arch):
        """모델 로드"""
        try:
            # End-to-End 모델 구조로 로드
            model = LightweightSTEAD(model_arch=model_arch)
            
            # 가중치 로드
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint, strict=False)
            
            return model
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            print("새로운 모델로 시작합니다.")
            return LightweightSTEAD(model_arch=model_arch)
    
    def preprocess_image(self, image_path):
        """이미지 전처리"""
        try:
            if isinstance(image_path, str):
                image = Image.open(image_path).convert('RGB')
            else:
                image = image_path.convert('RGB')
            
            # 전처리
            image_tensor = self.transform(image).unsqueeze(0)
            return image_tensor.to(self.device)
            
        except Exception as e:
            print(f"❌ 이미지 전처리 실패: {e}")
            return None
    
    def detect_anomaly(self, image_path, return_score=False):
        """이상 탐지"""
        start_time = time.time()
        
        # 이미지 전처리
        image_tensor = self.preprocess_image(image_path)
        if image_tensor is None:
            return None
        
        # 추론
        with torch.no_grad():
            output = self.model(image_tensor)
            score = torch.sigmoid(output).item()
        
        # 결과
        is_anomaly = score > self.threshold
        inference_time = time.time() - start_time
        
        if return_score:
            return {
                'is_anomaly': is_anomaly,
                'anomaly_score': score,
                'inference_time': inference_time,
                'threshold': self.threshold
            }
        else:
            return is_anomaly
    
    def batch_detect(self, image_paths, batch_size=4):
        """배치 단위 이상 탐지"""
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            
            # 배치 이미지 전처리
            for path in batch_paths:
                img_tensor = self.preprocess_image(path)
                if img_tensor is not None:
                    batch_images.append(img_tensor)
            
            if not batch_images:
                continue
            
            # 배치 추론
            batch_tensor = torch.cat(batch_images, dim=0)
            
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                scores = torch.sigmoid(outputs).cpu().numpy()
            
            # 결과 처리
            for j, score in enumerate(scores):
                if i + j < len(image_paths):
                    results.append({
                        'image_path': batch_paths[j],
                        'is_anomaly': score > self.threshold,
                        'anomaly_score': score,
                        'threshold': self.threshold
                    })
        
        return results
    
    def set_threshold(self, threshold):
        """임계값 설정"""
        self.threshold = threshold
        print(f"[INFO] 임계값 설정: {threshold}")
    
    def get_model_info(self):
        """모델 정보 반환"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_architecture': str(self.model.__class__.__name__),
            'device': str(self.device)
        }

def main():
    parser = argparse.ArgumentParser(description='실시간 이상 탐지')
    parser.add_argument('--model', type=str, required=True, help='모델 파일 경로')
    parser.add_argument('--image', type=str, help='테스트할 이미지 경로')
    parser.add_argument('--folder', type=str, help='테스트할 이미지 폴더 경로')
    parser.add_argument('--threshold', type=float, default=0.5, help='이상 탐지 임계값')
    parser.add_argument('--arch', type=str, default='tiny', choices=['tiny', 'base'], help='모델 아키텍처')
    parser.add_argument('--batch_size', type=int, default=4, help='배치 크기')
    
    args = parser.parse_args()
    
    # 모델 로드
    detector = RealtimeAnomalyDetector(args.model, model_arch=args.arch)
    detector.set_threshold(args.threshold)
    
    # 모델 정보 출력
    model_info = detector.get_model_info()
    print(f"\n=== 모델 정보 ===")
    print(f"총 파라미터: {model_info['total_parameters']:,}")
    print(f"학습 가능 파라미터: {model_info['trainable_parameters']:,}")
    print(f"모델 아키텍처: {model_info['model_architecture']}")
    print(f"디바이스: {model_info['device']}")
    
    # 단일 이미지 테스트
    if args.image:
        print(f"\n=== 단일 이미지 테스트 ===")
        print(f"이미지: {args.image}")
        
        result = detector.detect_anomaly(args.image, return_score=True)
        if result:
            print(f"이상 여부: {'비정상' if result['is_anomaly'] else '정상'}")
            print(f"이상 점수: {result['anomaly_score']:.4f}")
            print(f"추론 시간: {result['inference_time']*1000:.2f}ms")
    
    # 폴더 테스트
    elif args.folder:
        print(f"\n=== 폴더 테스트 ===")
        print(f"폴더: {args.folder}")
        
        # 이미지 파일 찾기
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_paths = []
        
        for root, dirs, files in os.walk(args.folder):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_paths.append(os.path.join(root, file))
        
        print(f"발견된 이미지: {len(image_paths)}개")
        
        if image_paths:
            # 배치 추론
            results = detector.batch_detect(image_paths, batch_size=args.batch_size)
            
            # 결과 요약
            normal_count = sum(1 for r in results if not r['is_anomaly'])
            anomaly_count = sum(1 for r in results if r['is_anomaly'])
            
            print(f"\n=== 결과 요약 ===")
            print(f"정상: {normal_count}개")
            print(f"비정상: {anomaly_count}개")
            print(f"총 처리: {len(results)}개")
            
            # 상위 비정상 이미지
            anomaly_results = [r for r in results if r['is_anomaly']]
            anomaly_results.sort(key=lambda x: x['anomaly_score'], reverse=True)
            
            if anomaly_results:
                print(f"\n=== 상위 비정상 이미지 ===")
                for i, result in enumerate(anomaly_results[:5]):
                    print(f"{i+1}. {os.path.basename(result['image_path'])}: {result['anomaly_score']:.4f}")
    
    else:
        print("사용법:")
        print("  단일 이미지: --image <이미지경로>")
        print("  폴더 테스트: --folder <폴더경로>")
        print("  예시:")
        print("    python inference_realtime.py --model model.pth --image test.jpg")
        print("    python inference_realtime.py --model model.pth --folder test_images/")

if __name__ == "__main__":
    main()
